"""Single-model OpenRouter helper for explanation and Q&A."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "z-ai/glm-4.5-air:free"
REQUEST_TIMEOUT_SECONDS = 20
MAX_RETRIES = 3
EXPLAIN_MAX_TOKENS = 140
QA_MAX_TOKENS = 140


def _reload_env() -> None:
    """Reload project environment files."""
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    load_dotenv(PROJECT_ROOT / "app" / ".env", override=True)


def _get_openrouter_api_key() -> str:
    """Fetch API key for the current process context."""
    _reload_env()
    key = (
        (os.getenv("OPENROUTER_API_KEY") or "").strip()
        or (os.getenv("\ufeffOPENROUTER_API_KEY") or "").strip()
    )
    if key.startswith('"') and key.endswith('"'):
        key = key[1:-1].strip()
    if key.startswith("'") and key.endswith("'"):
        key = key[1:-1].strip()
    return key


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert arbitrary input to float with a default."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clean_features(features_dict: dict | None) -> dict[str, float]:
    """Sanitize feature dictionary for prompt and local-fallback usage."""
    source = features_dict if isinstance(features_dict, dict) else {}
    return {
        "GrLivArea": _safe_float(source.get("GrLivArea"), 0.0),
        "BedroomAbvGr": _safe_float(source.get("BedroomAbvGr"), 0.0),
        "FullBath": _safe_float(source.get("FullBath"), 0.0),
        "YearBuilt": _safe_float(source.get("YearBuilt"), 0.0),
        "GarageArea": _safe_float(source.get("GarageArea"), 0.0),
    }


def _serialize_features(features_dict: dict | None) -> str:
    """Serialize feature values to compact JSON for prompt stability."""
    clean = _clean_features(features_dict)
    return json.dumps(clean, ensure_ascii=True, sort_keys=True)


def _build_headers() -> dict[str, str]:
    """Build OpenRouter request headers."""
    api_key = _get_openrouter_api_key()
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is missing.")
    referer = (os.getenv("OPENROUTER_HTTP_REFERER") or "http://localhost").strip()
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": referer,
        "X-Title": "Valora AI",
    }


def _normalize_content(raw_content: Any) -> str:
    """Normalize message content into plain text."""
    if isinstance(raw_content, str):
        return raw_content.strip()
    if isinstance(raw_content, list):
        chunks: list[str] = []
        for item in raw_content:
            if isinstance(item, dict):
                chunks.append(str(item.get("text", "")).strip())
            else:
                chunks.append(str(item).strip())
        return " ".join(part for part in chunks if part).strip()
    return str(raw_content or "").strip()


def _extract_response_content(response: requests.Response) -> str:
    """Extract assistant text from OpenRouter response."""
    payload = response.json()
    choices = payload.get("choices", [])
    if not isinstance(choices, list) or not choices:
        raise ValueError("OpenRouter response is missing choices.")

    first_choice = choices[0]
    message = first_choice.get("message", {}) if isinstance(first_choice, dict) else {}
    if isinstance(message, dict) and "content" in message:
        content = message["content"]
    elif isinstance(first_choice, dict) and "text" in first_choice:
        content = first_choice["text"]
    else:
        raise ValueError("OpenRouter response is missing text content.")

    text = _normalize_content(content)
    if not text:
        raise ValueError("OpenRouter returned empty content.")
    return text


def _first_nonempty_line(text: str) -> str:
    """Return the first non-empty output line."""
    for line in (text or "").splitlines():
        candidate = line.strip()
        if candidate:
            return candidate
    return ""


def _normalize_word_count(text: str, min_words: int = 20, max_words: int = 30) -> str:
    """Force explanation text into a bounded word range for UI consistency."""
    words = [part for part in (text or "").strip().split() if part]
    if len(words) > max_words:
        trimmed = " ".join(words[:max_words]).rstrip(",;:-")
        if not trimmed.endswith("."):
            trimmed += "."
        return trimmed
    return " ".join(words)


def _local_explanation(features_dict: dict | None, predicted_price: float) -> str:
    """Guaranteed local explanation when the API is unavailable."""
    features = _clean_features(features_dict)
    return (
        f"This estimate is ${predicted_price:,.0f}, driven by {features['GrLivArea']:,.0f} sq ft living space, "
        f"{features['BedroomAbvGr']:.0f} bedrooms, {features['FullBath']:.0f} bathrooms, year built {features['YearBuilt']:.0f}, and "
        f"{features['GarageArea']:,.0f} sq ft garage size."
    )


def _local_answer(question: str, features_dict: dict | None, predicted_price: float) -> str:
    """Guaranteed local Q&A answer when the API is unavailable."""
    q = (question or "").strip().lower()
    features = _clean_features(features_dict)

    if "what if" in q and ("bedroom" in q or "bedrooms" in q):
        return (
            f"If bedrooms increase from {features['BedroomAbvGr']:.0f}, the estimate usually rises; "
            "adjust BedroomAbvGr and click Predict Price for the exact updated value."
        )
    if "what if" in q and ("bath" in q or "bathroom" in q):
        return (
            f"If bathrooms increase from {features['FullBath']:.0f}, the estimate usually rises; "
            "adjust FullBath and run prediction again for the exact change."
        )
    if "what if" in q and "garage" in q:
        return (
            f"If garage area increases from {features['GarageArea']:,.0f} sq ft, the estimate usually rises; "
            "adjust GarageArea and re-run prediction for the exact impact."
        )
    if "what if" in q and ("year" in q or "built" in q or "newer" in q or "older" in q):
        return (
            f"If year built changes from {features['YearBuilt']:.0f}, newer years often increase estimates; "
            "change YearBuilt and predict again for the exact value."
        )

    return (
        f"Current estimate is ${predicted_price:,.0f}, influenced mostly by size ({features['GrLivArea']:,.0f} sq ft), "
        f"layout ({features['BedroomAbvGr']:.0f} bed/{features['FullBath']:.0f} bath), age ({features['YearBuilt']:.0f}), "
        f"and garage area ({features['GarageArea']:,.0f} sq ft)."
    )


def _call_openrouter(prompt: str, max_tokens: int) -> str:
    """Call OpenRouter with bounded retries using one fixed model."""
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a real-estate assistant. Reply in clear English, concise, "
                    "and user-facing only. Avoid meta commentary."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }

    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                OPENROUTER_CHAT_URL,
                headers=_build_headers(),
                json=payload,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            return _extract_response_content(response)
        except Exception as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(0.6 * attempt)

    if last_error is None:
        raise RuntimeError("OpenRouter request failed with unknown error.")
    raise last_error


def explain_prediction(features_dict: dict, predicted_price: float) -> str:
    """Generate a plain-language explanation for a model prediction."""
    safe_features_json = _serialize_features(features_dict)
    safe_price = _safe_float(predicted_price, 0.0)

    prompt = (
        "Explain this house price prediction in exactly one sentence with 20 to 30 words, plain English. "
        "No bullets, no markdown, no hidden reasoning.\n\n"
        f"Features JSON: {safe_features_json}\n"
        f"Predicted Sale Price: ${safe_price:,.2f}"
    )

    try:
        text = _call_openrouter(prompt, EXPLAIN_MAX_TOKENS)
        first_line = _first_nonempty_line(text)
        candidate = first_line or _local_explanation(features_dict, safe_price)
        normalized = _normalize_word_count(candidate, min_words=20, max_words=30)
        if len(normalized.split()) < 20:
            return _local_explanation(features_dict, safe_price)
        return normalized
    except Exception:
        return _local_explanation(features_dict, safe_price)


def answer_user_question(question: str, features_dict: dict, predicted_price: float) -> str:
    """Answer a user question using current prediction context."""
    question_text = (question or "").strip()
    if not question_text:
        return "Please ask a specific question about this prediction."

    safe_features_json = _serialize_features(features_dict)
    safe_price = _safe_float(predicted_price, 0.0)
    prompt = (
        "You are helping with a house price prediction app. "
        "Answer in plain English in one concise sentence. "
        "No bullets, no markdown, no hidden reasoning.\n\n"
        f"Question: {question_text}\n"
        f"Features JSON: {safe_features_json}\n"
        f"Predicted Sale Price: ${safe_price:,.2f}"
    )

    try:
        text = _call_openrouter(prompt, QA_MAX_TOKENS)
        first_line = _first_nonempty_line(text)
        return first_line or _local_answer(question_text, features_dict, safe_price)
    except Exception:
        return _local_answer(question_text, features_dict, safe_price)
