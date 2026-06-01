"""User-ad preference score construction and validation summaries."""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from typing import List

from .embedding_utils import cosine


WEIGHTS = {
    "cosine": 0.45,
    "segment_match": 0.30,
    "affordability_fit": 0.20,
}
NOISE_SCALE = 0.05


def compute_user_ad_scores(
    users: list[dict],
    advertisers: list[dict],
    ad_embeddings: List[List[float]],
    seed: int = 211,
) -> dict:
    rows = []
    raw_values = []

    for user in users:
        for advertiser, ad_embedding in zip(advertisers, ad_embeddings):
            cosine_score = (cosine(user["embedding"], ad_embedding) + 1.0) / 2.0
            segment_match_bonus = _segment_match(user["segment"], advertiser["target_audience"])
            affordability_fit = _affordability_fit(user, advertiser["target_audience"])
            noise = _stable_noise(seed, user["user_id"], advertiser["advertiser_id"])
            raw_score = (
                WEIGHTS["cosine"] * cosine_score
                + WEIGHTS["segment_match"] * segment_match_bonus
                + WEIGHTS["affordability_fit"] * affordability_fit
                + noise
            )
            row = {
                "user_id": user["user_id"],
                "segment": user["segment"],
                "advertiser_id": advertiser["advertiser_id"],
                "advertiser_name": advertiser["name"],
                "advertiser_target": advertiser["target_audience"],
                "cosine_score": round(cosine_score, 6),
                "segment_match_bonus": round(segment_match_bonus, 6),
                "memory_affordability_component": round(affordability_fit, 6),
                "noise": round(noise, 6),
                "raw_score": raw_score,
            }
            rows.append(row)
            raw_values.append(raw_score)

    min_raw = min(raw_values)
    max_raw = max(raw_values)
    span = max_raw - min_raw if max_raw > min_raw else 1.0
    for row in rows:
        row["s_iu"] = round((row.pop("raw_score") - min_raw) / span, 6)

    return {
        "score_formula": (
            "s_iu = 0.45*cosine(h_u,e_i) + 0.30*match(k(u),target_i) "
            "+ 0.20*affordability_fit(u,i) + epsilon_iu; min-max normalized."
        ),
        "weights": WEIGHTS,
        "noise_scale": NOISE_SCALE,
        "scores": rows,
    }


def summarize_scores(users: list[dict], advertisers: list[dict], score_payload: dict) -> dict:
    segments = defaultdict(int)
    archetypes = defaultdict(int)
    for user in users:
        segments[user["segment"]] += 1
        archetypes[user["generation_archetype"]] += 1

    memories = [user["memory_text"] for user in users]
    duplicate_memory_count = len(memories) - len(set(memories))

    by_segment_target = defaultdict(list)
    by_segment_ad = defaultdict(list)
    for row in score_payload["scores"]:
        by_segment_target[(row["segment"], row["advertiser_target"])].append(row["s_iu"])
        by_segment_ad[(row["segment"], row["advertiser_name"])].append(row["s_iu"])

    target_means = {
        f"{segment}__{target}": round(_mean(values), 6)
        for (segment, target), values in sorted(by_segment_target.items())
    }

    within_segment_variation = {}
    for segment in ["budget", "luxury"]:
        values_by_user = defaultdict(list)
        for row in score_payload["scores"]:
            if row["segment"] == segment:
                values_by_user[row["user_id"]].append(row["s_iu"])
        user_means = [_mean(values) for values in values_by_user.values()]
        within_segment_variation[segment] = round(_std(user_means), 6)

    return {
        "segment_counts": dict(segments),
        "generation_archetype_counts": dict(archetypes),
        "duplicate_memory_count": duplicate_memory_count,
        "target_mean_s_iu": target_means,
        "within_segment_user_mean_s_iu_std": within_segment_variation,
        "checks": {
            "has_100_users": len(users) == 100,
            "memories_not_duplicates": duplicate_memory_count == 0,
            "has_budget_and_luxury_segments": set(segments) == {"budget", "luxury"},
            "budget_scores_budget_ads_higher_than_luxury_ads": (
                target_means.get("budget__budget", 0.0)
                > target_means.get("budget__luxury", 1.0)
            ),
            "luxury_scores_luxury_ads_higher_than_budget_ads": (
                target_means.get("luxury__luxury", 0.0)
                > target_means.get("luxury__budget", 1.0)
            ),
            "irrelevant_ads_low_on_average": (
                _mean(target_means[k] for k in target_means if k.endswith("__neither"))
                < _mean(target_means.values())
            ),
            "budget_users_have_score_variation": within_segment_variation.get("budget", 0.0) > 0,
            "luxury_users_have_score_variation": within_segment_variation.get("luxury", 0.0) > 0,
        },
    }


def _segment_match(user_segment: str, target: str) -> float:
    if target == "both":
        return 0.6
    if target == "neither":
        return 0.0
    return 1.0 if user_segment == target else 0.0


def _affordability_fit(user: dict, target: str) -> float:
    if target == "luxury":
        return float(user["luxury_score"])
    if target == "budget":
        return float(user["budget_score"])
    if target == "both":
        return 0.5 + 0.5 * max(float(user["budget_score"]), float(user["luxury_score"]))
    return 0.1


def _stable_noise(seed: int, user_id: str, advertiser_id: str) -> float:
    key = f"{seed}:{user_id}:{advertiser_id}".encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    value = int.from_bytes(digest, "big") / float(2**64 - 1)
    centered = value - 0.5
    return centered * 2.0 * NOISE_SCALE


def _mean(values) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def _std(values) -> float:
    vals = list(values)
    if len(vals) < 2:
        return 0.0
    mean = _mean(vals)
    return (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5

