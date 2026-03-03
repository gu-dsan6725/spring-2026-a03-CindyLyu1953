"""Shared LLM client using LiteLLM with Groq."""

import time
from pathlib import Path

from dotenv import load_dotenv
import litellm
from litellm import RateLimitError

# Load .env from project root
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# Default model - Groq (llama-3.1-70b was decommissioned, use 3.3)
DEFAULT_MODEL = "groq/llama-3.3-70b-versatile"


def complete(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float = 0.2,
    max_retries: int = 3,
) -> str:
    """
    Call LLM with messages and return the assistant reply.
    Retries on rate limit with exponential backoff.

    Args:
        messages: List of {"role": "user"|"system"|"assistant", "content": "..."}
        model: Override model (default: Groq llama-3.3-70b)
        temperature: Sampling temperature (default 0.2 for more deterministic)
        max_retries: Retries on rate limit

    Returns:
        The assistant's reply text.
    """
    model = model or DEFAULT_MODEL
    last_err = None
    for attempt in range(max_retries):
        try:
            response = litellm.completion(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except RateLimitError as e:
            last_err = e
            if attempt < max_retries - 1:
                wait = 15 * (2**attempt)  # 15s, 30s, 60s
                time.sleep(wait)
            else:
                raise
    raise last_err


def complete_with_context(
    question: str,
    context: str,
    system_prompt: str | None = None,
    model: str | None = None,
) -> str:
    """
    Convenience: answer a question given retrieved context.

    Args:
        question: User's question
        context: Retrieved context to ground the answer
        system_prompt: Optional system instructions
        model: Optional model override

    Returns:
        The model's answer.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    user_content = f"{question}\n\nContext:\n{context}"
    messages.append({"role": "user", "content": user_content})
    return complete(messages, model=model)
