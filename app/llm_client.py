import os
import time
import json
import logging
from groq import Groq

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")
        # Hard per-request timeout — protects the evaluator's 30s/call budget.
        # Without this, a hung Groq call has no ceiling of its own.
        self.client = Groq(api_key=api_key, timeout=12.0)
        self.model = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
        logger.info(f"LLM client initialised — model={self.model}")

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        max_retries: int = 1,  # was 2 — /chat makes 2 sequential LLM calls,
        # so retry budget must stay small enough that both calls still fit
        # comfortably inside the evaluator's 30s window.
    ) -> str:
        kwargs: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                response = self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                logger.debug(
                    "LLM response (attempt %d, %d tokens): %.200s…",
                    attempt + 1,
                    response.usage.completion_tokens if response.usage else -1,
                    content,
                )
                return content
            except Exception as exc:
                last_error = exc
                exc_str = str(exc)
                if "429" in exc_str and "rate_limit_exceeded" in exc_str:
                    logger.error(
                        "QUOTA EXHAUSTED: Groq limit hit. Full error: %s", exc_str[:300]
                    )
                    break
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s",
                    attempt + 1, max_retries + 1, exc,
                )
                if attempt < max_retries:
                    time.sleep(0.4)
        raise RuntimeError(
            f"LLM call failed after {max_retries + 1} attempts"
        ) from last_error

    def call_safe_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> dict:
        try:
            raw = self.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                json_mode=True,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return json.loads(raw)
        except (json.JSONDecodeError, RuntimeError) as exc:
            logger.error("call_safe_json failed: %s", exc)
            return {}