from __future__ import annotations
import json
import hashlib
import redis
from config import get_settings

settings = get_settings()

# ──────────────────────────────────────────────
# Redis connection
# ──────────────────────────────────────────────

_redis_client: redis.Redis | None = None


def get_redis() -> redis.Redis | None:
    """
    Connect to Redis. Returns None if REDIS_URL is not set
    or connection fails (graceful degradation).
    """
    global _redis_client

    if not settings.redis_url:
        return None

    if _redis_client is None:
        try:
            _redis_client = redis.from_url(
                settings.redis_url,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
            )
            _redis_client.ping()
            print("Redis connected.")
        except Exception as e:
            print(f"Redis connection failed: {e}. Continuing without cache.")
            _redis_client = None

    return _redis_client


def is_connected() -> bool:
    """Check if Redis is reachable."""
    try:
        client = get_redis()
        if client:
            client.ping()
            return True
    except Exception:
        pass
    return False


# ──────────────────────────────────────────────
# Cache key generation
# ──────────────────────────────────────────────

def _make_cache_key(profile_dict: dict) -> str:
    """
    Generate a deterministic cache key from profile data.
    Only includes fields that affect roadmap output
    (ignores timestamps, user_id, etc.)
    """
    relevant_fields = {
        "current_role": profile_dict.get("current_role"),
        "current_industry": profile_dict.get("current_industry"),
        "years_of_experience": profile_dict.get("years_of_experience"),
        "technical_skills": sorted(profile_dict.get("technical_skills", [])),
        "soft_skills": sorted(profile_dict.get("soft_skills", [])),
        "certifications": sorted(profile_dict.get("certifications", [])),
        "interest_domains": sorted(profile_dict.get("interest_domains", [])),
        "career_goal": profile_dict.get("career_goal"),
        "life_stage": profile_dict.get("life_stage"),
        "burnout_level": profile_dict.get("burnout_level"),
        "stress_tolerance": profile_dict.get("stress_tolerance"),
        "has_dependents": profile_dict.get("has_dependents"),
        "recent_life_event": profile_dict.get("recent_life_event"),
        "work_life_priority": profile_dict.get("work_life_priority"),
        "age": profile_dict.get("age"),
    }

    key_string = json.dumps(relevant_fields, sort_keys=True)
    key_hash = hashlib.sha256(key_string.encode()).hexdigest()[:16]
    return f"rag:roadmap:{key_hash}"


# ──────────────────────────────────────────────
# Get / Set cached responses
# ──────────────────────────────────────────────

def get_cached_response(profile_dict: dict) -> dict | None:
    """
    Look up a cached RAG response for the given profile.
    Returns the parsed JSON dict or None.
    """
    client = get_redis()
    if not client:
        return None

    try:
        key = _make_cache_key(profile_dict)
        cached = client.get(key)
        if cached:
            print(f"Cache HIT: {key}")
            return json.loads(cached)
        print(f"Cache MISS: {key}")
    except Exception as e:
        print(f"Cache read error: {e}")

    return None


def set_cached_response(profile_dict: dict, response_dict: dict) -> bool:
    """
    Store a RAG response in Redis with TTL.
    Returns True on success.
    """
    client = get_redis()
    if not client:
        return False

    try:
        key = _make_cache_key(profile_dict)
        client.set(
            key,
            json.dumps(response_dict),
            ex=settings.redis_ttl_seconds,
        )
        print(f"Cache SET: {key} (TTL: {settings.redis_ttl_seconds}s)")
        return True
    except Exception as e:
        print(f"Cache write error: {e}")
        return False
