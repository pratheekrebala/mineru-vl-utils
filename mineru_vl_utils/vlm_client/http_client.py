import asyncio
import json
import os
import re
from enum import Enum
from typing import Any, AsyncIterable, Iterable, Sequence

import httpx
from httpx_retries import Retry, RetryTransport
from loguru import logger

from .base_client import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT,
    ImageType,
    RequestError,
    SamplingParams,
    ServerError,
    VlmClient,
)
from .utils import (
    aio_image_to_bytes_list_and_format,
    gather_tasks,
    get_image_data_url,
    image_to_bytes_list_and_format,
)


def _get_env(key: str, default: str | None = None) -> str:
    value = os.getenv(key)
    if value not in (None, ""):
        return value
    if default is not None:
        return default
    raise ValueError(f"Environment variable {key} is not set.")


class HTTPMethod(str, Enum):
    HEAD = "HEAD"
    GET = "GET"
    PUT = "PUT"
    DELETE = "DELETE"
    OPTIONS = "OPTIONS"
    TRACE = "TRACE"
    POST = "POST"


class HttpVlmClient(VlmClient):
    def __init__(
        self,
        model_name: str | None = None,
        server_url: str | None = None,
        server_headers: dict[str, str] | None = None,
        prompt: str = DEFAULT_USER_PROMPT,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        sampling_params: SamplingParams | None = None,
        text_before_image: bool = False,
        allow_truncated_content: bool = False,
        max_concurrency: int = 100,
        http_timeout: int = 600,
        connect_timeout: int = 10,
        max_connections: int | None = None,
        max_keepalive_connections: int | None = 20,
        keepalive_expiry: float | None = 5,
        http_batch_size: int = 0,
        debug: bool = False,
        max_retries: int = 3,
        retry_backoff_factor: float = 0.5,
        skip_model_name_checking: bool = False,
    ) -> None:
        super().__init__(
            prompt=prompt,
            system_prompt=system_prompt,
            sampling_params=sampling_params,
            text_before_image=text_before_image,
            allow_truncated_content=allow_truncated_content,
        )
        self.max_concurrency = max_concurrency
        self.debug = debug

        if not server_url:
            server_url = _get_env("MINERU_VL_SERVER")
        if server_url.endswith("/"):  # keep server_url if it ends with '/'
            server_url = server_url.rstrip("/")
        else:  # use base_url if it does not end with '/' (backward compatibility)
            server_url = self._get_base_url(server_url)
        self.server_url = server_url

        api_key = os.getenv("MINERU_VL_API_KEY", "").strip()
        if api_key:
            headers = dict(server_headers) if server_headers else {}
            if "Authorization" in headers:
                logger.warning("Overriding existing 'Authorization' header with MINERU_VL_API_KEY from environment variable.")
            headers["Authorization"] = f"Bearer {api_key}"
            self.server_headers = headers
        else:
            self.server_headers = server_headers

        self.http_timeout = http_timeout
        self.connect_timeout = connect_timeout
        self.max_connections = max_connections
        self.max_keepalive_connections = max_keepalive_connections
        self.keepalive_expiry = keepalive_expiry
        self.http_batch_size = int(os.getenv("MINERU_VL_HTTP_BATCH_SIZE", str(http_batch_size or 0)) or 0)
        self.max_retries = max_retries
        self.retry_backoff_factor = retry_backoff_factor

        self._client = self._new_client()
        self._aio_client_sem = asyncio.Semaphore(1)
        self._aio_client_cache: dict[asyncio.AbstractEventLoop, httpx.AsyncClient] = {}

        model_name = model_name or os.getenv("MINERU_VL_MODEL_NAME")
        if model_name:
            if not skip_model_name_checking:
                self._check_model_name(self.server_url, model_name)
            self.model_name = model_name
        else:
            self.model_name = self._get_model_name(self.server_url)

    @property
    def chat_url(self) -> str:
        return f"{self.server_url}/v1/chat/completions"

    @property
    def chat_batch_url(self) -> str:
        return f"{self.server_url}/v1/chat/completions/batch"

    def _new_client(self) -> httpx.Client:
        return httpx.Client(
            headers=self.server_headers,
            timeout=httpx.Timeout(
                connect=self.connect_timeout,
                read=self.http_timeout,
                write=self.http_timeout,
                pool=None,
            ),
            transport=RetryTransport(
                retry=Retry(
                    total=self.max_retries,
                    backoff_factor=self.retry_backoff_factor,
                    allowed_methods=list(HTTPMethod),
                ),
                transport=httpx.HTTPTransport(
                    limits=httpx.Limits(
                        max_connections=self.max_connections,
                        max_keepalive_connections=self.max_keepalive_connections,
                        keepalive_expiry=self.keepalive_expiry,
                    ),
                ),
            ),
        )

    async def _new_aio_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            headers=self.server_headers,
            timeout=httpx.Timeout(
                connect=self.connect_timeout,
                read=self.http_timeout,
                write=self.http_timeout,
                pool=None,
            ),
            transport=RetryTransport(
                retry=Retry(
                    total=self.max_retries,
                    backoff_factor=self.retry_backoff_factor,
                    allowed_methods=list(HTTPMethod),
                ),
                transport=httpx.AsyncHTTPTransport(
                    limits=httpx.Limits(
                        max_connections=self.max_connections,
                        max_keepalive_connections=self.max_keepalive_connections,
                        keepalive_expiry=self.keepalive_expiry,
                    ),
                ),
            ),
        )

    async def _aio_client(self) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        aio_client = self._aio_client_cache.get(loop)
        if aio_client is not None:
            return aio_client
        async with self._aio_client_sem:
            aio_client = self._aio_client_cache.get(loop)
            if aio_client is not None:
                return aio_client
            aio_client = await self._new_aio_client()
            self._aio_client_cache.clear()
            self._aio_client_cache[loop] = aio_client
            return aio_client

    def _get_base_url(self, server_url: str) -> str:
        matched = re.match(r"^(https?://[^/]+)", server_url)
        if not matched:
            raise RequestError(f"Invalid server URL: {server_url}")
        return matched.group(1)

    def _check_model_name(self, base_url: str, model_name: str):
        try:
            response = self._client.get(f"{base_url}/v1/models")
        except httpx.ConnectError:
            raise ServerError(f"Failed to connect to server {base_url}. Please check if the server is running.")
        if response.status_code != 200:
            raise ServerError(
                f"Failed to get model name from {base_url}. Status code: {response.status_code}, response body: {response.text}"
            )
        for model in response.json().get("data", []):
            if model.get("id") == model_name:
                return
        raise RequestError(
            f"Model '{model_name}' not found in the response from {base_url}/v1/models. "
            "Please check if the model is available on the server."
        )

    def _get_model_name(self, base_url: str) -> str:
        try:
            response = self._client.get(f"{base_url}/v1/models")
        except httpx.ConnectError:
            raise ServerError(f"Failed to connect to server {base_url}. Please check if the server is running.")
        if response.status_code != 200:
            raise ServerError(
                f"Failed to get model name from {base_url}. Status code: {response.status_code}, response body: {response.text}"
            )
        models = response.json().get("data", [])
        if not isinstance(models, list):
            raise RequestError(f"No models found in response from {base_url}. Response body: {response.text}")
        if len(models) != 1:
            raise RequestError(
                f"Expected exactly one model from {base_url}, but got {len(models)}. Please specify the model name"
                f" or set the `MINERU_VL_MODEL_NAME` environment variable."
            )
        model_name = models[0].get("id", "")
        if not model_name:
            raise RequestError(f"Model name is empty in response from {base_url}. Response body: {response.text}")
        return model_name

    def build_request_body(
        self,
        system_prompt: str,
        image: Sequence[bytes],
        prompt: str,
        sampling_params: SamplingParams | None,
        image_format: str | None,
        priority: int | None,
    ) -> dict:
        image_urls = [get_image_data_url(im, image_format) for im in image]
        prompt = prompt or self.prompt
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if "<image>" in prompt:
            prompt_parts = prompt.split("<image>", len(image_urls))
            user_messages = []
            for i in range(max(len(prompt_parts), len(image_urls))):
                if i < len(prompt_parts) and prompt_parts[i]:
                    user_messages.append({"type": "text", "text": prompt_parts[i]})
                if i < len(image_urls):
                    user_messages.append({"type": "image_url", "image_url": {"url": image_urls[i]}})
        elif self.text_before_image:
            user_messages = [
                {"type": "text", "text": prompt},
                *({"type": "image_url", "image_url": {"url": image_url}} for image_url in image_urls),
            ]
        else:  # image before text, which is the default behavior.
            user_messages = [
                *({"type": "image_url", "image_url": {"url": image_url}} for image_url in image_urls),
                {"type": "text", "text": prompt},
            ]
        messages.append({"role": "user", "content": user_messages})

        sp = self.build_sampling_params(sampling_params)
        sp_dict = {}
        if sp.temperature is not None:
            sp_dict["temperature"] = sp.temperature
        if sp.top_p is not None:
            sp_dict["top_p"] = sp.top_p
        if sp.top_k is not None:
            sp_dict["top_k"] = sp.top_k
        if sp.presence_penalty is not None:
            sp_dict["presence_penalty"] = sp.presence_penalty
        if sp.frequency_penalty is not None:
            sp_dict["frequency_penalty"] = sp.frequency_penalty
        if sp.repetition_penalty is not None:
            sp_dict["repetition_penalty"] = sp.repetition_penalty
        if sp.no_repeat_ngram_size is not None:
            sp_dict["vllm_xargs"] = {
                "no_repeat_ngram_size": sp.no_repeat_ngram_size,
                "debug": self.debug,
            }
        if sp.max_new_tokens is not None:
            sp_dict["max_completion_tokens"] = sp.max_new_tokens
            sp_dict["max_tokens"] = sp.max_new_tokens  # for compatibility
        sp_dict["skip_special_tokens"] = False

        if self.model_name.lower().startswith("gpt"):
            sp_dict.pop("top_k", None)
            sp_dict.pop("repetition_penalty", None)
            sp_dict.pop("skip_special_tokens", None)

        return {
            "model": self.model_name,
            "messages": messages,
            **({"priority": priority} if priority is not None else {}),
            **sp_dict,
        }

    def get_response_data(self, response: httpx.Response) -> dict:
        if response.status_code != 200:
            raise ServerError(f"Unexpected status code: [{response.status_code}], response body: {response.text}")
        try:
            response_data = response.json()
        except Exception as e:
            raise ServerError(f"Failed to parse response JSON: {e}, response body: {response.text}")
        if not isinstance(response_data, dict):
            raise ServerError(f"Response is not a JSON object: {response.text}")
        return response_data

    def get_response_content(self, response_data: dict) -> str:
        if response_data.get("object") == "error":
            raise ServerError(f"Error from server: {response_data}")
        choices = response_data.get("choices")
        if not (isinstance(choices, list) and choices):
            raise ServerError("No choices found in the response.")
        finish_reason = choices[0].get("finish_reason")
        if finish_reason is None:
            raise ServerError("Finish reason is None in the response.")
        if finish_reason == "length":
            if not self.allow_truncated_content:
                raise RequestError("The response was truncated due to length limit.")
            else:
                logger.warning("The response was truncated due to length limit.")
        elif finish_reason != "stop":
            raise RequestError(f"Unexpected finish reason: {finish_reason}")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise ServerError("Message not found in the response.")
        if "content" not in message:
            raise ServerError("Content not found in the message.")
        content = message["content"]
        if not (content is None or isinstance(content, str)):
            raise ServerError(f"Unexpected content type: {type(content)}.")
        # Allow the end token to be configured via environment variable, falling back to the default.
        # Set MINERU_VLM_END_TOKEN to override or disable stripping (e.g., set to an empty string).
        end_token = os.getenv("MINERU_VLM_END_TOKEN", "<|im_end|>")
        if end_token and isinstance(content, str) and content.endswith(end_token):
            content = content[: -len(end_token)]
        return content or ""

    def get_batch_response_contents(self, response_data: dict, expected_count: int) -> list[str]:
        if response_data.get("object") == "error":
            raise ServerError(f"Error from server: {response_data}")
        choices = response_data.get("choices")
        if not isinstance(choices, list):
            raise ServerError("Choices not found in the batch response.")
        if len(choices) != expected_count:
            raise ServerError(f"Expected {expected_count} choices in batch response, got {len(choices)}.")

        contents: list[str | None] = [None] * expected_count
        for fallback_idx, choice in enumerate(choices):
            if not isinstance(choice, dict):
                raise ServerError(f"Unexpected choice type in batch response: {type(choice)}.")
            idx = choice.get("index", fallback_idx)
            if not isinstance(idx, int) or idx < 0 or idx >= expected_count:
                raise ServerError(f"Invalid choice index in batch response: {idx}")
            contents[idx] = self.get_response_content({"choices": [choice]})

        missing_indices = [idx for idx, content in enumerate(contents) if content is None]
        if missing_indices:
            raise ServerError(f"Missing choices in batch response for indices: {missing_indices}")
        return [content or "" for content in contents]

    @staticmethod
    def _batch_group_key(request_body: dict[str, Any]) -> str:
        batch_shared_body = {key: value for key, value in request_body.items() if key != "messages"}
        return json.dumps(batch_shared_body, sort_keys=True, ensure_ascii=False)

    def _build_batch_request_body(self, request_bodies: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if not request_bodies:
            raise RequestError("Cannot build an empty batch request.")
        batch_body = {key: value for key, value in request_bodies[0].items() if key != "messages"}
        batch_body["messages"] = [body["messages"] for body in request_bodies]
        return batch_body

    async def _aio_predict_batch_chunk(self, request_bodies: Sequence[dict[str, Any]]) -> list[str]:
        request_body = self._build_batch_request_body(request_bodies)

        if self.debug:
            request_text = json.dumps(request_body, ensure_ascii=False)
            if len(request_text) > 4096:
                request_text = request_text[:2048] + "...(truncated)..." + request_text[-2048:]
            logger.debug("Batch request body: {}", request_text)

        client = await self._aio_client()
        response = await client.post(self.chat_batch_url, json=request_body)
        if response.status_code in {404, 405}:
            raise RequestError(
                f"Batch endpoint is not supported by server {self.server_url}. "
                "Disable MINERU_VL_HTTP_BATCH_SIZE or use a vLLM server with /v1/chat/completions/batch."
            )
        response_data = self.get_response_data(response)

        if self.debug:
            logger.debug("Batch response status code: {}", response.status_code)
            logger.debug("Batch response body: {}", response.text)

        return self.get_batch_response_contents(response_data, len(request_bodies))

    async def _aio_batch_predict_vllm_batch_endpoint(
        self,
        images: Sequence[ImageType],
        prompts: Sequence[str],
        sampling_params: Sequence[SamplingParams | None],
        priority: Sequence[int | None],
        semaphore: asyncio.Semaphore,
        use_tqdm: bool = False,
        tqdm_desc: str | None = None,
    ) -> list[str]:
        prepared_images = await gather_tasks(
            tasks=[aio_image_to_bytes_list_and_format(image) for image in images],
            use_tqdm=use_tqdm,
            tqdm_desc=f"{tqdm_desc or 'Prediction'} Preparation",
        )
        request_bodies = [
            self.build_request_body(
                system_prompt=self.system_prompt,
                image=image_bytes,
                prompt=prompt,
                sampling_params=params,
                image_format=image_format,
                priority=prio,
            )
            for (image_bytes, image_format), prompt, params, prio in zip(
                prepared_images,
                prompts,
                sampling_params,
                priority,
            )
        ]

        indexed_outputs: list[str | None] = [None] * len(request_bodies)
        grouped_indices: dict[str, list[int]] = {}
        for idx, request_body in enumerate(request_bodies):
            grouped_indices.setdefault(self._batch_group_key(request_body), []).append(idx)

        async def predict_chunk(chunk_indices: list[int]) -> tuple[list[int], list[str]]:
            async with semaphore:
                chunk_bodies = [request_bodies[idx] for idx in chunk_indices]
                return chunk_indices, await self._aio_predict_batch_chunk(chunk_bodies)

        tasks = []
        batch_size = max(1, self.http_batch_size)
        for indices in grouped_indices.values():
            for start in range(0, len(indices), batch_size):
                tasks.append(predict_chunk(indices[start : start + batch_size]))

        chunk_results = await gather_tasks(tasks=tasks, use_tqdm=use_tqdm, tqdm_desc=tqdm_desc)
        for chunk_indices, chunk_outputs in chunk_results:
            for idx, output in zip(chunk_indices, chunk_outputs):
                indexed_outputs[idx] = output

        missing_indices = [idx for idx, output in enumerate(indexed_outputs) if output is None]
        if missing_indices:
            raise ServerError(f"Missing batch outputs for indices: {missing_indices}")
        return [output or "" for output in indexed_outputs]

    def predict(
        self,
        image: ImageType,
        prompt: str = "",
        sampling_params: SamplingParams | None = None,
        priority: int | None = None,
    ) -> str:
        image, image_format = image_to_bytes_list_and_format(image)

        request_body = self.build_request_body(
            system_prompt=self.system_prompt,
            image=image,
            prompt=prompt,
            sampling_params=sampling_params,
            image_format=image_format,
            priority=priority,
        )

        if self.debug:
            request_text = json.dumps(request_body, ensure_ascii=False)
            if len(request_text) > 4096:
                request_text = request_text[:2048] + "...(truncated)..." + request_text[-2048:]
            logger.debug("Request body: {}", request_text)

        response = self._client.post(self.chat_url, json=request_body)

        if self.debug:
            logger.debug("Response status code: {}", response.status_code)
            logger.debug("Response body: {}", response.text)

        response_data = self.get_response_data(response)
        return self.get_response_content(response_data)

    def batch_predict(
        self,
        images: Sequence[ImageType],
        prompts: Sequence[str] | str = "",
        sampling_params: Sequence[SamplingParams | None] | SamplingParams | None = None,
        priority: Sequence[int | None] | int | None = None,
    ) -> list[str]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        task = self.aio_batch_predict(
            images=images,
            prompts=prompts,
            sampling_params=sampling_params,
            priority=priority,
        )

        if loop is not None:
            return loop.run_until_complete(task)
        else:
            return asyncio.run(task)

    def stream_predict(
        self,
        image: ImageType,
        prompt: str = "",
        sampling_params: SamplingParams | None = None,
        priority: int | None = None,
    ) -> Iterable[str]:
        image, image_format = image_to_bytes_list_and_format(image)

        request_body = self.build_request_body(
            system_prompt=self.system_prompt,
            image=image,
            prompt=prompt,
            sampling_params=sampling_params,
            image_format=image_format,
            priority=priority,
        )
        request_body["stream"] = True

        if self.debug:
            request_text = json.dumps(request_body, ensure_ascii=False)
            if len(request_text) > 4096:
                request_text = request_text[:2048] + "...(truncated)..." + request_text[-2048:]
            logger.debug("Request body: {}", request_text)

        with self._client.stream("POST", self.chat_url, json=request_body) as response:
            for chunk in response.iter_lines():
                chunk = chunk.strip()
                if not chunk.startswith("data:"):
                    continue
                chunk = chunk[5:].lstrip()
                if chunk == "[DONE]":
                    break
                response_data = json.loads(chunk)
                choices = response_data.get("choices") or []
                choice = choices[0] if choices else {}
                delta = choice.get("delta") or {}
                if "content" in delta:
                    yield delta["content"]

    def stream_test(
        self,
        image: ImageType,
        prompt: str = "",
        sampling_params: SamplingParams | None = None,
        priority: int | None = None,
    ) -> None:
        """
        Test the streaming functionality by printing the output.
        """
        print("[Streaming Output]", flush=True)
        for chunk in self.stream_predict(
            image=image,
            prompt=prompt,
            sampling_params=sampling_params,
            priority=priority,
        ):
            print(chunk, end="", flush=True)
        print("\n[End of Streaming Output]", flush=True)

    async def aio_predict(
        self,
        image: ImageType,
        prompt: str = "",
        sampling_params: SamplingParams | None = None,
        priority: int | None = None,
    ) -> str:
        image, image_format = await aio_image_to_bytes_list_and_format(image)

        request_body = self.build_request_body(
            system_prompt=self.system_prompt,
            image=image,
            prompt=prompt,
            sampling_params=sampling_params,
            image_format=image_format,
            priority=priority,
        )

        if self.debug:
            request_text = json.dumps(request_body, ensure_ascii=False)
            if len(request_text) > 4096:
                request_text = request_text[:2048] + "...(truncated)..." + request_text[-2048:]
            logger.debug("Request body: {}", request_text)

        client = await self._aio_client()
        response = await client.post(self.chat_url, json=request_body)
        response_data = self.get_response_data(response)

        if self.debug:
            logger.debug("Response status code: {}", response.status_code)
            logger.debug("Response body: {}", response.text)

        return self.get_response_content(response_data)

    async def aio_batch_predict(
        self,
        images: Sequence[ImageType],
        prompts: Sequence[str] | str = "",
        sampling_params: Sequence[SamplingParams | None] | SamplingParams | None = None,
        priority: Sequence[int | None] | int | None = None,
        semaphore: asyncio.Semaphore | None = None,
        use_tqdm=False,
        tqdm_desc: str | None = None,
    ) -> list[str]:
        if isinstance(prompts, str):
            prompts = [prompts] * len(images)
        if not isinstance(sampling_params, Sequence):
            sampling_params = [sampling_params] * len(images)
        if not isinstance(priority, Sequence):
            priority = [priority] * len(images)

        assert len(prompts) == len(images), "Length of prompts and images must match."
        assert len(sampling_params) == len(images), "Length of sampling_params and images must match."
        assert len(priority) == len(images), "Length of priority and images must match."

        if semaphore is None:
            semaphore = asyncio.Semaphore(self.max_concurrency)

        if self.http_batch_size > 1:
            return await self._aio_batch_predict_vllm_batch_endpoint(
                images=images,
                prompts=prompts,
                sampling_params=sampling_params,
                priority=priority,
                semaphore=semaphore,
                use_tqdm=use_tqdm,
                tqdm_desc=tqdm_desc,
            )

        async def predict_with_semaphore(
            image: ImageType,
            prompt: str,
            sampling_params: SamplingParams | None,
            priority: int | None,
        ):
            async with semaphore:
                return await self.aio_predict(
                    image=image,
                    prompt=prompt,
                    sampling_params=sampling_params,
                    priority=priority,
                )

        return await gather_tasks(
            tasks=[
                predict_with_semaphore(*args)
                for args in zip(
                    images,
                    prompts,
                    sampling_params,
                    priority,
                )
            ],
            use_tqdm=use_tqdm,
            tqdm_desc=tqdm_desc,
        )

    async def aio_batch_predict_as_iter(
        self,
        images: Sequence[ImageType],
        prompts: Sequence[str] | str = "",
        sampling_params: Sequence[SamplingParams | None] | SamplingParams | None = None,
        priority: Sequence[int | None] | int | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> AsyncIterable[tuple[int, str]]:
        if isinstance(prompts, str):
            prompts = [prompts] * len(images)
        if not isinstance(sampling_params, Sequence):
            sampling_params = [sampling_params] * len(images)
        if not isinstance(priority, Sequence):
            priority = [priority] * len(images)

        assert len(prompts) == len(images), "Length of prompts and images must match."
        assert len(sampling_params) == len(images), "Length of sampling_params and images must match."
        assert len(priority) == len(images), "Length of priority and images must match."

        if semaphore is None:
            semaphore = asyncio.Semaphore(self.max_concurrency)

        async def predict_with_semaphore(
            idx: int,
            image: ImageType,
            prompt: str,
            sampling_params: SamplingParams | None,
            priority: int | None,
        ):
            async with semaphore:
                output = await self.aio_predict(
                    image=image,
                    prompt=prompt,
                    sampling_params=sampling_params,
                    priority=priority,
                )
                return (idx, output)

        pending: set[asyncio.Task[tuple[int, str]]] = set()

        for idx, args in enumerate(zip(images, prompts, sampling_params, priority)):
            pending.add(asyncio.create_task(predict_with_semaphore(idx, *args)))

        while len(pending) > 0:
            done, pending = await asyncio.wait(
                pending,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                yield task.result()

    # async def aio_stream_predict(
    #     self,
    #     image: ImageType,
    #     prompt: str = "",
    #     temperature: Optional[float] = None,
    #     top_p: Optional[float] = None,
    #     top_k: Optional[int] = None,
    #     repetition_penalty: Optional[float] = None,
    #     presence_penalty: Optional[float] = None,
    #     no_repeat_ngram_size: Optional[int] = None,
    #     max_new_tokens: Optional[int] = None,
    # ) -> AsyncIterable[str]:
    #     prompt = self.build_prompt(prompt)

    #     sampling_params = self.build_sampling_params(
    #         temperature=temperature,
    #         top_p=top_p,
    #         top_k=top_k,
    #         repetition_penalty=repetition_penalty,
    #         presence_penalty=presence_penalty,
    #         no_repeat_ngram_size=no_repeat_ngram_size,
    #         max_new_tokens=max_new_tokens,
    #     )

    #     if isinstance(image, str):
    #         image = await aio_load_resource(image)

    #     request_body = self.build_request_body(image, prompt, sampling_params)
    #     request_body["stream"] = True

    #     async with httpx.AsyncClient(timeout=self.http_timeout) as client:
    #         async with client.stream(
    #             "POST",
    #             self.server_url,
    #             json=request_body,
    #         ) as response:
    #             pos = 0
    #             async for chunk in response.aiter_lines():
    #                 if not (chunk or "").startswith("data:"):
    #                     continue
    #                 if chunk == "data: [DONE]":
    #                     break
    #                 data = json.loads(chunk[5:].strip("\n"))
    #                 chunk_text = data["text"][pos:]
    #                 # meta_info = data["meta_info"]
    #                 pos += len(chunk_text)
    #                 yield chunk_text
