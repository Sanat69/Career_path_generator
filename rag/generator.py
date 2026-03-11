from __future__ import annotations
import json
import time
import re
from groq import Groq
from config import get_settings
from prompts.roadmap import ROADMAP_SYSTEM_PROMPT, build_roadmap_prompt
from prompts.audit import AUDIT_SYSTEM_PROMPT, build_audit_prompt

settings = get_settings()

# ──────────────────────────────────────────────
# Groq client (initialized once)
# ──────────────────────────────────────────────

_groq_client: Groq | None = None


def get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=settings.groq_api_key)
        print("Groq client initialized.")
    return _groq_client


def is_groq_available() -> bool:
    """Quick check if Groq API is reachable."""
    try:
        client = get_groq_client()
        # Minimal test call
        client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=5,
        )
        return True
    except Exception as e:
        print(f"Groq availability check failed: {e}")
        return False


# ──────────────────────────────────────────────
# JSON parsing helpers
# ──────────────────────────────────────────────

def _clean_json_response(text: str) -> str:
    """
    Strip markdown fences, leading/trailing whitespace,
    and any non-JSON preamble from LLM output.
    """
    text = text.strip()

    # Remove ```json ... ``` wrappers
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()

    # If there's text before the first {, strip it
    first_brace = text.find('{')
    if first_brace > 0:
        text = text[first_brace:]

    # If there's text after the last }, strip it
    last_brace = text.rfind('}')
    if last_brace >= 0 and last_brace < len(text) - 1:
        text = text[:last_brace + 1]

    return text


def _parse_json_safe(text: str) -> dict | None:
    """Attempt to parse JSON from LLM output, with cleanup."""
    cleaned = _clean_json_response(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"JSON parse failed: {e}")
        print(f"Raw output (first 500 chars): {text[:500]}")
        return None


# ──────────────────────────────────────────────
# Groq call with exponential backoff retry
# ──────────────────────────────────────────────

def _call_groq(
    system_prompt: str,
    user_message: str,
    max_retries: int = 3,
    temperature: float = 0.3,
) -> dict | None:
    """
    Call Groq LLaMA 3 with retry logic.
    Returns parsed JSON dict or None on failure.
    """
    client = get_groq_client()

    for attempt in range(max_retries):
        try:
            print(f"Groq call attempt {attempt + 1}/{max_retries}...")
            response = client.chat.completions.create(
                model=settings.groq_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=temperature,
                max_tokens=4096,
                response_format={"type": "json_object"},
            )

            raw_text = response.choices[0].message.content
            parsed = _parse_json_safe(raw_text)

            if parsed is not None:
                print(f"Groq call succeeded on attempt {attempt + 1}.")
                return parsed

            # JSON parse failed — retry with lower temperature
            print(f"Attempt {attempt + 1}: JSON parse failed, retrying...")
            temperature = max(0.1, temperature - 0.1)

        except Exception as e:
            error_str = str(e).lower()

            if "rate_limit" in error_str or "429" in error_str:
                wait_time = (2 ** attempt) * 5  # 5, 10, 20 seconds
                print(f"Rate limited. Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            elif "503" in error_str or "service unavailable" in error_str:
                wait_time = (2 ** attempt) * 3
                print(f"Service unavailable. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"Groq error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)

    print("All Groq retry attempts exhausted.")
    return None


# ──────────────────────────────────────────────
# Public API: Generate roadmap
# ──────────────────────────────────────────────

def generate_roadmap(
    profile_dict: dict,
    retrieved_docs: list[dict],
) -> dict | None:
    """
    Generate a career transition roadmap using Groq LLaMA 3.
    Returns parsed JSON roadmap or None on failure.
    """
    user_message = build_roadmap_prompt(profile_dict, retrieved_docs)
    result = _call_groq(ROADMAP_SYSTEM_PROMPT, user_message)

    if result:
        # Validate minimum structure
        required_keys = ["roadmap_nodes", "roadmap_edges", "current_role",
                         "target_role", "success_probability", "explanation"]
        missing = [k for k in required_keys if k not in result]
        if missing:
            print(f"WARNING: Roadmap missing keys: {missing}")
            # Still return what we have — the response model will handle defaults

    return result


# ──────────────────────────────────────────────
# Public API: Generate ethical audit
# ──────────────────────────────────────────────

def generate_audit(
    profile_dict: dict,
    roadmap_json: dict,
) -> list[dict]:
    """
    Generate PASSIONIT/PRUTL ethical audit scores.
    Returns list of audit score dicts.
    """
    user_message = build_audit_prompt(profile_dict, roadmap_json)
    result = _call_groq(AUDIT_SYSTEM_PROMPT, user_message, temperature=0.2)

    if result and "audit_scores" in result:
        scores = result["audit_scores"]
        print(f"Ethical audit generated: {len(scores)} dimensions evaluated.")
        return scores

    # Fallback: return empty audit if LLM fails
    print("WARNING: Ethical audit generation failed. Returning empty scores.")
    return []
