from __future__ import annotations

import asyncio
import csv
import time
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path
from threading import Lock
from typing import Any, Dict

from fastapi import FastAPI

from services.card_generator import generate_profile_card
from services.classifier import classify_profile
from services.github_scraper import GitHubScraperError, fetch_github_profile
from services.nlp_pipeline import DEFAULT_BASELINE_STATS, extract_features


ANALYTICS_PATH = Path("analytics.csv")
ANALYTICS_LOCK = Lock()


class ProfileAnalysisError(Exception):
    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def _normalized_username(username: str) -> str:
    cleaned = username.strip().lstrip("@")
    if not cleaned:
        raise ProfileAnalysisError("Please enter a GitHub username.", 400)
    return cleaned


def _cache_entry_valid(entry: Dict[str, Any], ttl_seconds: int) -> bool:
    cached_at = entry.get("cached_at")
    if not isinstance(cached_at, datetime):
        return False
    return datetime.now(timezone.utc) - cached_at < timedelta(seconds=ttl_seconds)


def _append_analytics(username: str, archetype: str) -> None:
    with ANALYTICS_LOCK:
        file_exists = ANALYTICS_PATH.exists()
        with ANALYTICS_PATH.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if not file_exists:
                writer.writerow(["timestamp_utc", "username", "archetype"])
            writer.writerow([datetime.now(timezone.utc).isoformat(), username, archetype])


def _card_output_path(username: str) -> str:
    return f"/tmp/{username}_card.png"


async def _compute_profile(app: FastAPI, normalized_username: str) -> Dict[str, Any]:
    started_at = time.perf_counter()
    loop = asyncio.get_running_loop()
    executor = app.state.executor
    model_bundle = app.state.model_bundle
    baseline_stats = model_bundle.get("baseline_stats", DEFAULT_BASELINE_STATS) if model_bundle else DEFAULT_BASELINE_STATS

    try:
        raw_data = await loop.run_in_executor(executor, partial(fetch_github_profile, normalized_username))
    except GitHubScraperError as exc:
        raise ProfileAnalysisError(exc.message, exc.status_code) from exc
    except Exception as exc:
        raise ProfileAnalysisError("Unexpected error while reaching GitHub. Please try again shortly.", 503) from exc

    try:
        feature_data = await loop.run_in_executor(
            executor,
            partial(extract_features, raw_data, baseline_stats, True, False),
        )
        classification = await loop.run_in_executor(
            executor,
            partial(classify_profile, feature_data, model_bundle),
        )
    except ProfileAnalysisError:
        raise
    except Exception as exc:
        raise ProfileAnalysisError(
            "We fetched the GitHub profile, but one of the analysis steps failed. Please try again with another user or refresh in a moment.",
            500,
        ) from exc

    duration_ms = int((time.perf_counter() - started_at) * 1000)
    result = {
        "username": raw_data["user"]["login"],
        "raw_data": raw_data,
        "features": feature_data,
        "classification": classification,
        "card_path": _card_output_path(raw_data["user"]["login"]),
        "cached_at": datetime.now(timezone.utc).isoformat(),
        "meta": {
            "cached": False,
            "analysis_ms": duration_ms,
            "card_ready": Path(_card_output_path(raw_data["user"]["login"])).exists(),
        },
    }

    cache = app.state.profile_cache
    cache[normalized_username.lower()] = {
        "payload": result,
        "cached_at": datetime.now(timezone.utc),
    }
    _append_analytics(normalized_username, classification["archetype"])
    return result


async def ensure_profile_card(app: FastAPI, username: str) -> Dict[str, Any]:
    result = await analyze_profile(app, username)
    card_path = Path(result["card_path"])
    if card_path.exists():
        result["meta"]["card_ready"] = True
        return result

    loop = asyncio.get_running_loop()
    executor = app.state.executor
    card_path_string = await loop.run_in_executor(
        executor,
        partial(generate_profile_card, result["raw_data"], result["features"], result["classification"]),
    )
    result["card_path"] = card_path_string
    result["meta"]["card_ready"] = True

    cache_entry = app.state.profile_cache.get(result["username"].lower())
    if cache_entry:
        cache_entry["payload"] = result
    return result


async def analyze_profile(app: FastAPI, username: str, force_refresh: bool = False) -> Dict[str, Any]:
    normalized_username = _normalized_username(username)
    cache = app.state.profile_cache
    ttl_seconds = app.state.cache_ttl_seconds

    cached = cache.get(normalized_username.lower())
    if cached and not force_refresh and _cache_entry_valid(cached, ttl_seconds):
        result = cached["payload"]
        result["meta"]["cached"] = True
        result["meta"]["card_ready"] = Path(result["card_path"]).exists()
        _append_analytics(normalized_username, result["classification"]["archetype"])
        return result

    inflight = app.state.inflight_analyses
    existing_task = inflight.get(normalized_username.lower())
    if existing_task and not force_refresh:
        return await existing_task

    task = asyncio.create_task(_compute_profile(app, normalized_username))
    inflight[normalized_username.lower()] = task
    try:
        return await task
    finally:
        if inflight.get(normalized_username.lower()) is task:
            inflight.pop(normalized_username.lower(), None)
