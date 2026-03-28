"""OpenAI-compatible LLM client supporting multiple providers."""

import json
import logging
from typing import Generator, List, Optional

from openai import OpenAI

from config import settings

logger = logging.getLogger("llm")


class LLMClient:
    """Unified LLM client using OpenAI SDK."""

    PROVIDER_CONFIGS = {
        "cerebras": {
            "base_url": "https://api.cerebras.ai/v1",
            "model": "llama3.1-8b",
        },
        "nvidia": {
            "base_url": "https://integrate.api.nvidia.com/v1",
            "model": "moonshotai/kimi-k2.5",
        },
        "gemini": {
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
            "model": "gemini-2.5-flash",
        },
        "openrouter": {
            "base_url": "https://openrouter.ai/api/v1",
            "model": "openrouter/free",
        },
        "openrouter_meta": {
            "base_url": "https://openrouter.ai/api/v1",
            "model": "meta-llama/llama-3.3-70b-instruct:free",
        },
    }

    def __init__(
        self, provider: Optional[str] = None, fallback_provider: Optional[str] = None
    ):
        self.provider = (provider or settings.llm.provider).lower()
        self.fallback_provider = fallback_provider

        if self.provider not in self.PROVIDER_CONFIGS:
            raise ValueError(
                f"Unknown provider: {self.provider}. Use: {list(self.PROVIDER_CONFIGS.keys())}"
            )

        self.config = dict(self.PROVIDER_CONFIGS[self.provider])
        if settings.llm.model_override:
            self.config["model"] = settings.llm.model_override

        self.api_key = settings.get_api_key(self.provider)
        self.client = OpenAI(
            base_url=self.config["base_url"],
            api_key=self.api_key,
        )

        logger.info(
            f"Initialized provider={self.provider} | model={self.config['model']} | base_url={self.config['base_url']}"
        )

        self.fallback_client = None
        if self.fallback_provider and self.fallback_provider in self.PROVIDER_CONFIGS:
            try:
                fallback_key = settings.get_api_key(self.fallback_provider)
                if fallback_key:
                    self.fallback_client = LLMClient(self.fallback_provider)
                    logger.info(
                        f"Fallback ready: provider={self.fallback_provider} | model={self.fallback_client.config['model']}"
                    )
            except Exception as e:
                logger.warning(
                    f"Fallback provider '{self.fallback_provider}' unavailable: {e}"
                )

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a complete response (non-streaming)."""
        logger.info(
            f"Calling provider={self.provider} | model={self.config['model']} | base_url={self.config['base_url']}"
        )
        try:
            response = self.client.chat.completions.create(
                model=self.config["model"],
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=kwargs.get("temperature", settings.llm.temperature),
                max_tokens=kwargs.get("max_tokens", settings.llm.max_tokens),
            )
            content = response.choices[0].message.content or ""
            if not content and self.fallback_client:
                logger.warning(
                    f"Primary provider={self.provider} returned empty, trying fallback={self.fallback_client.provider}"
                )
                return self.fallback_client.generate(prompt, **kwargs)
            logger.info(
                f"Success from provider={self.provider} | model={self.config['model']}"
            )
            return content
        except Exception as e:
            if self.fallback_client:
                logger.warning(
                    f"Primary provider={self.provider} model={self.config['model']} failed: {e} | "
                    f"Falling back to provider={self.fallback_client.provider} model={self.fallback_client.config['model']}"
                )
                return self.fallback_client.generate(prompt, **kwargs)
            raise Exception(f"LLM generation failed: {e}")

    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """Stream response tokens. Falls back to simulated streaming if provider
        doesn't support true streaming or times out."""
        import logging
        import threading
        import time

        logger = logging.getLogger("llm")

        # Try true streaming with a first-token timeout
        first_token_timeout = getattr(settings.llm, "stream_first_token_timeout", 15)
        tokens_received = []
        stream_error = [None]
        first_token_event = threading.Event()
        stream_done = threading.Event()

        def _stream_in_thread():
            try:
                stream = self.client.chat.completions.create(
                    model=self.config["model"],
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=kwargs.get("temperature", settings.llm.temperature),
                    max_tokens=kwargs.get("max_tokens", settings.llm.max_tokens),
                    stream=True,
                )
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        token = chunk.choices[0].delta.content
                        tokens_received.append(token)
                        first_token_event.set()
            except Exception as e:
                stream_error[0] = e
            finally:
                first_token_event.set()
                stream_done.set()

        thread = threading.Thread(target=_stream_in_thread, daemon=True)
        thread.start()

        # Wait for first token or timeout
        got_token = first_token_event.wait(timeout=first_token_timeout)

        if got_token and not stream_error[0] and tokens_received:
            # True streaming is working - yield accumulated tokens, then wait for more
            logger.info(
                f"True streaming active | provider={self.provider} | "
                f"model={self.config['model']} | first_token in <{first_token_timeout}s"
            )
            yielded = 0
            while not stream_done.is_set() or yielded < len(tokens_received):
                while yielded < len(tokens_received):
                    yield tokens_received[yielded]
                    yielded += 1
                if not stream_done.is_set():
                    time.sleep(0.05)  # Small poll interval
            # Yield any remaining
            while yielded < len(tokens_received):
                yield tokens_received[yielded]
                yielded += 1
        else:
            # Streaming timed out or failed - fallback to non-streaming
            reason = (
                f"error: {stream_error[0]}"
                if stream_error[0]
                else "no tokens within timeout"
            )
            logger.warning(
                f"Streaming fallback | provider={self.provider} | "
                f"model={self.config['model']} | reason={reason} | "
                f"using non-streaming generate()"
            )

            # Get complete response via non-streaming
            full_response = self.generate(prompt, **kwargs)

            if not full_response:
                logger.warning("Non-streaming generate() returned empty response")
                return

            # Simulate streaming by yielding word-by-word
            words = full_response.split(" ")
            for i, word in enumerate(words):
                chunk = word if i == len(words) - 1 else word + " "
                yield chunk

    def get_suggestions(
        self,
        question: str,
        context_snippets: List[str],
        history: Optional[List[dict]] = None,
    ) -> List[str]:
        """Generate follow-up question suggestions using function calling.

        Uses tool_choice to force structured JSON output — guaranteed to return
        a valid list of strings or an empty list on failure.
        """
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "suggest_followups",
                    "description": "Suggest 2-3 natural follow-up questions the user might ask next, based on the conversation context.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "questions": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 2,
                                "maxItems": 3,
                                "description": "Short, natural follow-up questions (max 60 chars each)",
                            }
                        },
                        "required": ["questions"],
                    },
                },
            }
        ]

        # Build messages for the suggestion call
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a legal research assistant. Based on the user's question and the "
                    "available legal context from case law documents, "
                    "suggest 2-3 natural follow-up questions the user might want to ask next. "
                    "Questions should be concise (under 60 characters) and explore different "
                    "angles of the legal topic such as related cases, precedents, or legal principles."
                ),
            }
        ]

        # Add conversation history if available
        if history:
            for msg in history[-6:]:  # Last 3 exchanges
                role = msg.get("role", "user")
                if role in ("user", "assistant"):
                    messages.append(
                        {"role": role, "content": msg.get("content", "")[:500]}
                    )

        # Add context summary and current question
        context_summary = "\n".join(snippet[:300] for snippet in context_snippets[:3])
        messages.append(
            {
                "role": "user",
                "content": f"Available context:\n{context_summary}\n\nCurrent question: {question}",
            }
        )

        try:
            response = self.client.chat.completions.create(
                model=self.config["model"],
                messages=messages,
                tools=tools,
                tool_choice={
                    "type": "function",
                    "function": {"name": "suggest_followups"},
                },
                temperature=0.7,
                max_tokens=200,
            )

            tool_call = response.choices[0].message.tool_calls[0]
            result = json.loads(tool_call.function.arguments)
            suggestions = result.get("questions", [])

            logger.info(
                f"Suggestions generated | provider={self.provider} | "
                f"count={len(suggestions)} | questions={suggestions}"
            )
            return suggestions[:3]

        except Exception as e:
            logger.warning(f"Suggestion generation failed: {e}")
            return []
