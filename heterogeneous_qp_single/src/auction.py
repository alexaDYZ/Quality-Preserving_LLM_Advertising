"""Auction-only simulation for the heterogeneous QP single mechanism."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .embedding_utils import Embedder, cosine


EPS = 1e-12
ORGANIC_ID = "organic"
ORGANIC_NAME = "organic"


@dataclass(frozen=True)
class AuctionConfig:
    rho_q: float = 1.0
    rho_s: float = 1.0
    mu: float = 1.0
    lambda_tilde: float = 1.0
    click_scale: float = 1.0


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_scenario_data(data_dir: Path) -> dict:
    payload = {
        "users": load_json(data_dir / "users_classified.json"),
        "advertisers": load_json(data_dir / "advertisers.json"),
        "bids_by_segment": load_json(data_dir / "bids_by_segment.json"),
        "user_ad_scores": load_json(data_dir / "user_ad_scores.json"),
        "organic_reference": load_json(data_dir / "organic_reference.json"),
    }
    summary_path = data_dir / "validation_summary.json"
    payload["validation_summary"] = load_json(summary_path) if summary_path.exists() else {}
    return payload


def load_no_ad_outputs(path: Path) -> list[str]:
    payload = load_json(path)
    outputs = payload.get("outputs", [])
    if not outputs:
        raise ValueError(f"No organic/no-ad outputs found in {path}")
    return outputs


def build_score_lookup(score_payload: dict) -> tuple[dict[tuple[str, str], float], tuple[float, float]]:
    score_lookup = {}
    raw_scores = []
    weights = score_payload["weights"]
    for row in score_payload["scores"]:
        score_lookup[(row["user_id"], row["advertiser_id"])] = float(row["s_iu"])
        raw_score = (
            weights["cosine"] * float(row["cosine_score"])
            + weights["segment_match"] * float(row["segment_match_bonus"])
            + weights["affordability_fit"] * float(row["memory_affordability_component"])
            + float(row["noise"])
        )
        raw_scores.append(raw_score)
    if not raw_scores:
        raise ValueError("No advertiser preference scores found.")
    return score_lookup, (min(raw_scores), max(raw_scores))


def infer_embedding_backend(data: dict, requested_backend: str) -> str:
    if requested_backend != "data":
        return requested_backend
    backend = data.get("validation_summary", {}).get("embedding_backend")
    if backend in {"hashing", "sentence_transformers"}:
        return backend
    return "auto"


def advertiser_relevance(
    advertisers: list[dict],
    query: str,
    embedder: Embedder,
) -> dict[str, float]:
    missing_relevance = [
        advertiser
        for advertiser in advertisers
        if "han_dai_q_i_v1" not in advertiser
    ]
    relevance = {
        advertiser["advertiser_id"]: float(advertiser["han_dai_q_i_v1"])
        for advertiser in advertisers
        if "han_dai_q_i_v1" in advertiser
    }
    if missing_relevance:
        query_embedding = embedder.encode([query])[0]
        ad_embeddings = embedder.encode([a["profile_text"] for a in missing_relevance])
        for advertiser, ad_embedding in zip(missing_relevance, ad_embeddings):
            relevance[advertiser["advertiser_id"]] = (cosine(query_embedding, ad_embedding) + 1.0) / 2.0
    return relevance


def organic_primitives(
    user: dict,
    organic_text: str,
    query: str,
    embedder: Embedder,
    score_payload: dict,
    raw_score_bounds: tuple[float, float],
) -> dict:
    query_embedding, organic_embedding = embedder.encode([query, organic_text])
    q0 = (cosine(query_embedding, organic_embedding) + 1.0) / 2.0

    user_embedding = [float(v) for v in user["embedding"]]
    if len(user_embedding) != len(organic_embedding):
        raise ValueError(
            "User and organic embeddings have different dimensions. Regenerate data "
            "or run this script with an embedding backend matching validation_summary.json."
        )

    cosine_score = (cosine(user_embedding, organic_embedding) + 1.0) / 2.0
    weights = score_payload["weights"]
    affordability_fit = 0.5 + 0.5 * max(float(user["budget_score"]), float(user["luxury_score"]))
    raw_score = (
        weights["cosine"] * cosine_score
        + weights["segment_match"] * 0.6
        + weights["affordability_fit"] * affordability_fit
    )
    min_raw, max_raw = raw_score_bounds
    span = max(max_raw - min_raw, EPS)
    s0 = min(1.0, max(0.0, (raw_score - min_raw) / span))
    return {
        "q0": q0,
        "s0u": s0,
        "organic_cosine_score": cosine_score,
        "organic_raw_score": raw_score,
    }


def mnl_probabilities(utilities: dict[str, float]) -> dict[str, float]:
    exp_values = {source_id: math.exp(value) for source_id, value in utilities.items()}
    denominator = 1.0 + sum(exp_values.values())
    probabilities = {source_id: exp_value / denominator for source_id, exp_value in exp_values.items()}
    probabilities["outside"] = 1.0 / denominator
    return probabilities


def kl_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    value = 0.0
    for key, p_value in p.items():
        if p_value <= 0:
            continue
        q_value = max(q.get(key, 0.0), EPS)
        value += p_value * math.log(p_value / q_value)
    return value


def normalize_over(source_ids: Iterable[str], values: dict[str, float]) -> dict[str, float]:
    ids = list(source_ids)
    total = sum(max(values[source_id], 0.0) for source_id in ids)
    if total <= 0:
        equal = 1.0 / len(ids)
        return {source_id: equal for source_id in ids}
    return {source_id: max(values[source_id], 0.0) / total for source_id in ids}


def stable_softmax(log_weights: dict[str, float]) -> dict[str, float]:
    max_log_weight = max(log_weights.values())
    weights = {
        source_id: math.exp(log_weight - max_log_weight)
        for source_id, log_weight in log_weights.items()
    }
    total = sum(weights.values())
    return {source_id: weight / total for source_id, weight in weights.items()}


def run_single_round(
    *,
    user: dict,
    advertisers: list[dict],
    bids_by_segment: dict,
    score_lookup: dict[tuple[str, str], float],
    ad_relevance: dict[str, float],
    organic: dict,
    used_advertiser_ids: set[str],
    with_replacement: bool,
    config: AuctionConfig,
) -> dict:
    segment = user["segment"]
    active_advertisers = [
        advertiser
        for advertiser in advertisers
        if with_replacement or advertiser["advertiser_id"] not in used_advertiser_ids
    ]

    utilities = {}
    source_scores = {}
    source_relevance = {}
    bids = {}
    reserves = {}
    theoretical_reserves = {}
    ctr = {}

    for advertiser in active_advertisers:
        advertiser_id = advertiser["advertiser_id"]
        bid = float(bids_by_segment[advertiser_id][segment])
        s_iu = score_lookup[(user["user_id"], advertiser_id)]
        q_i = ad_relevance[advertiser_id]
        utility = config.rho_q * q_i + config.rho_s * s_iu
        utilities[advertiser_id] = utility
        source_scores[advertiser_id] = s_iu
        source_relevance[advertiser_id] = q_i
        bids[advertiser_id] = bid

    utilities[ORGANIC_ID] = config.rho_q * organic["q0"] + config.rho_s * organic["s0u"]
    source_scores[ORGANIC_ID] = organic["s0u"]
    source_relevance[ORGANIC_ID] = organic["q0"]
    bids[ORGANIC_ID] = 0.0

    pi_mnl = mnl_probabilities(utilities)
    organic_utility = utilities[ORGANIC_ID]

    eligible_ad_ids = []
    for advertiser in active_advertisers:
        advertiser_id = advertiser["advertiser_id"]
        pi_i = max(pi_mnl[advertiser_id], EPS)
        theoretical_reserve = -config.mu * (utilities[advertiser_id] - organic_utility) / (
            config.click_scale * pi_i
        )
        reserve = max(0.0, theoretical_reserve)
        theoretical_reserves[advertiser_id] = theoretical_reserve
        reserves[advertiser_id] = reserve
        ctr[advertiser_id] = config.click_scale * pi_mnl[advertiser_id]
        if bids[advertiser_id] >= reserve:
            eligible_ad_ids.append(advertiser_id)

    ctr[ORGANIC_ID] = 0.0
    candidate_ids = eligible_ad_ids + [ORGANIC_ID]
    pi_alloc = normalize_over(candidate_ids, pi_mnl)

    if not eligible_ad_ids:
        allocation = {ORGANIC_ID: 1.0}
        payments = {}
        xi_at_reserve = {}
    else:
        allocation = _allocation_for_bids(
            candidate_ids=candidate_ids,
            bids=bids,
            utilities=utilities,
            pi_mnl=pi_mnl,
            pi_alloc=pi_alloc,
            config=config,
        )
        payments = {}
        xi_at_reserve = {}
        for advertiser_id in eligible_ad_ids:
            counterfactual_bids = dict(bids)
            counterfactual_bids[advertiser_id] = reserves[advertiser_id]
            counterfactual_allocation = _allocation_for_bids(
                candidate_ids=candidate_ids,
                bids=counterfactual_bids,
                utilities=utilities,
                pi_mnl=pi_mnl,
                pi_alloc=pi_alloc,
                config=config,
            )
            xi_b = max(allocation[advertiser_id], EPS)
            xi_r = max(counterfactual_allocation[advertiser_id], EPS)
            xi_at_reserve[advertiser_id] = xi_r
            ctr_i = max(config.click_scale * pi_mnl[advertiser_id], EPS)
            payments[advertiser_id] = (
                bids[advertiser_id] * (xi_b - 1.0)
                + reserves[advertiser_id]
                + (config.lambda_tilde / ctr_i) * math.log(xi_b / xi_r)
            )

    allocation_full = {advertiser["advertiser_id"]: 0.0 for advertiser in advertisers}
    allocation_full[ORGANIC_ID] = 0.0
    allocation_full.update(allocation)

    payments_full = {advertiser["advertiser_id"]: 0.0 for advertiser in advertisers}
    payments_full.update(payments)

    revenue_terms = {
        advertiser_id: allocation_full[advertiser_id] * payment
        for advertiser_id, payment in payments.items()
    }
    expected_revenue = sum(revenue_terms.values())
    expected_ads = sum(allocation_full[advertiser["advertiser_id"]] for advertiser in advertisers)
    expected_relevance = sum(
        allocation[source_id] * source_relevance[source_id]
        for source_id in candidate_ids
    )
    kl = kl_divergence(allocation, pi_alloc)
    gross_social_welfare = sum(
        allocation[source_id] * _welfare_score(source_id, bids, utilities, pi_mnl, config)
        for source_id in candidate_ids
    )
    social_welfare = gross_social_welfare - config.lambda_tilde * kl

    return {
        "user_id": user["user_id"],
        "segment": segment,
        "active_advertiser_ids": [a["advertiser_id"] for a in active_advertisers],
        "eligible_advertiser_ids": eligible_ad_ids,
        "candidate_ids": candidate_ids,
        "organic_text": organic["text"],
        "organic_index": organic["index"],
        "utilities": utilities,
        "pi_mnl": pi_mnl,
        "pi_alloc": pi_alloc,
        "ctr": ctr,
        "bids": bids,
        "theoretical_reserves": theoretical_reserves,
        "reserves": reserves,
        "allocation": allocation_full,
        "payments_per_click": payments_full,
        "xi_at_reserve": xi_at_reserve,
        "expected_revenue": expected_revenue,
        "revenue_terms": revenue_terms,
        "expected_ads": expected_ads,
        "expected_relevance": expected_relevance,
        "kl_alloc_vs_pi": kl,
        "gross_social_welfare": gross_social_welfare,
        "social_welfare": social_welfare,
    }


def _allocation_for_bids(
    *,
    candidate_ids: list[str],
    bids: dict[str, float],
    utilities: dict[str, float],
    pi_mnl: dict[str, float],
    pi_alloc: dict[str, float],
    config: AuctionConfig,
) -> dict[str, float]:
    log_weights = {}
    for source_id in candidate_ids:
        welfare_score = _welfare_score(source_id, bids, utilities, pi_mnl, config)
        log_weights[source_id] = math.log(max(pi_alloc[source_id], EPS)) + (
            welfare_score / config.lambda_tilde
        )
    return stable_softmax(log_weights)


def _welfare_score(
    source_id: str,
    bids: dict[str, float],
    utilities: dict[str, float],
    pi_mnl: dict[str, float],
    config: AuctionConfig,
) -> float:
    if source_id == ORGANIC_ID:
        return config.mu * utilities[source_id]
    return config.click_scale * pi_mnl[source_id] * bids[source_id] + config.mu * utilities[source_id]


def sample_winner(allocation: dict[str, float], rng: random.Random) -> str:
    source_ids = list(allocation)
    weights = [allocation[source_id] for source_id in source_ids]
    return rng.choices(source_ids, weights=weights, k=1)[0]


def run_user_response(
    *,
    user: dict,
    advertisers: list[dict],
    bids_by_segment: dict,
    score_lookup: dict[tuple[str, str], float],
    ad_relevance: dict[str, float],
    organic_outputs: list[str],
    query: str,
    embedder: Embedder,
    score_payload: dict,
    raw_score_bounds: tuple[float, float],
    n_rounds: int,
    with_replacement: bool,
    rng: random.Random,
    config: AuctionConfig,
) -> dict:
    used_advertiser_ids: set[str] = set()
    rounds = []
    for round_idx in range(1, n_rounds + 1):
        organic_index = rng.randrange(len(organic_outputs))
        organic_text = organic_outputs[organic_index]
        organic = organic_primitives(
            user=user,
            organic_text=organic_text,
            query=query,
            embedder=embedder,
            score_payload=score_payload,
            raw_score_bounds=raw_score_bounds,
        )
        organic["text"] = organic_text
        organic["index"] = organic_index

        round_log = run_single_round(
            user=user,
            advertisers=advertisers,
            bids_by_segment=bids_by_segment,
            score_lookup=score_lookup,
            ad_relevance=ad_relevance,
            organic=organic,
            used_advertiser_ids=used_advertiser_ids,
            with_replacement=with_replacement,
            config=config,
        )
        sampled_winner = sample_winner(
            {
                source_id: allocation
                for source_id, allocation in round_log["allocation"].items()
                if allocation > 0
            },
            rng,
        )
        round_log["round"] = round_idx
        round_log["sampled_winner"] = sampled_winner
        round_log["sampled_ad"] = sampled_winner != ORGANIC_ID
        if sampled_winner != ORGANIC_ID and not with_replacement:
            used_advertiser_ids.add(sampled_winner)
        rounds.append(round_log)

    return {
        "user_id": user["user_id"],
        "segment": user["segment"],
        "with_replacement": with_replacement,
        "rounds": rounds,
    }


def summarize_trials(trials: list[dict], advertisers: list[dict], config: AuctionConfig) -> dict:
    rows = [round_log for trial in trials for round_log in trial["rounds"]]
    n_responses = len(trials)
    n_rounds_total = len(rows)
    total_revenue = sum(row["expected_revenue"] for row in rows)
    total_ads = sum(row["expected_ads"] for row in rows)
    sampled_ads = sum(1 for row in rows if row["sampled_ad"])

    by_segment = {}
    for segment in sorted({trial["segment"] for trial in trials}):
        segment_rows = [row for row in rows if row["segment"] == segment]
        segment_revenue = sum(row["expected_revenue"] for row in segment_rows)
        segment_ads = sum(row["expected_ads"] for row in segment_rows)
        by_segment[segment] = _summary_from_rows(
            segment_rows,
            segment_revenue,
            segment_ads,
            len({row["user_id"] for row in segment_rows}),
        )

    by_advertiser = {}
    for advertiser in advertisers:
        advertiser_id = advertiser["advertiser_id"]
        advertiser_revenue = sum(
            row["revenue_terms"].get(advertiser_id, 0.0)
            for row in rows
        )
        advertiser_ads = sum(
            row["allocation"].get(advertiser_id, 0.0)
            for row in rows
        )
        by_advertiser[advertiser["name"]] = {
            "advertiser_id": advertiser_id,
            "expected_revenue": advertiser_revenue,
            "expected_ads": advertiser_ads,
            "revenue_per_ad": advertiser_revenue / advertiser_ads if advertiser_ads > 0 else 0.0,
            "mean_allocation_per_round": advertiser_ads / n_rounds_total if n_rounds_total else 0.0,
        }

    return {
        "config": {
            "rho_q": config.rho_q,
            "rho_s": config.rho_s,
            "mu": config.mu,
            "lambda_tilde": config.lambda_tilde,
            "click_scale": config.click_scale,
            "ctr_assumption": "ctr_iu = click_scale * pi_iu; default click_scale=1.",
            "reserve_policy": "Effective reserve is max(0, theoretical reserve).",
            "revenue_metric": "Expected revenue uses sum_i allocation_i * per_click_payment_i.",
            "organic_sampling": "One saved no-ad output is sampled per user response round.",
        },
        "n_responses": n_responses,
        "n_rounds_total": n_rounds_total,
        "expected_revenue": total_revenue,
        "expected_ads": total_ads,
        "sampled_ads": sampled_ads,
        "revenue_per_ad": total_revenue / total_ads if total_ads > 0 else 0.0,
        "expected_ads_per_response": total_ads / n_responses if n_responses else 0.0,
        "sampled_ads_per_response": sampled_ads / n_responses if n_responses else 0.0,
        "social_welfare_per_response": sum(row["social_welfare"] for row in rows) / n_responses,
        "gross_social_welfare_per_response": (
            sum(row["gross_social_welfare"] for row in rows) / n_responses
        ),
        "mean_relevance_per_round": sum(row["expected_relevance"] for row in rows) / n_rounds_total,
        "mean_kl_per_round": sum(row["kl_alloc_vs_pi"] for row in rows) / n_rounds_total,
        "mean_expected_revenue_per_round": total_revenue / n_rounds_total,
        "by_segment": by_segment,
        "by_advertiser": by_advertiser,
    }


def _summary_from_rows(
    rows: list[dict],
    total_revenue: float,
    total_ads: float,
    n_responses: int,
) -> dict:
    n_rounds = len(rows)
    sampled_ads = sum(1 for row in rows if row["sampled_ad"])
    return {
        "n_responses": n_responses,
        "n_rounds": n_rounds,
        "expected_revenue": total_revenue,
        "expected_ads": total_ads,
        "sampled_ads": sampled_ads,
        "revenue_per_ad": total_revenue / total_ads if total_ads > 0 else 0.0,
        "expected_ads_per_response": total_ads / n_responses if n_responses else 0.0,
        "sampled_ads_per_response": sampled_ads / n_responses if n_responses else 0.0,
        "social_welfare_per_response": sum(row["social_welfare"] for row in rows) / n_responses,
        "gross_social_welfare_per_response": (
            sum(row["gross_social_welfare"] for row in rows) / n_responses
        ),
        "mean_relevance_per_round": sum(row["expected_relevance"] for row in rows) / n_rounds,
        "mean_kl_per_round": sum(row["kl_alloc_vs_pi"] for row in rows) / n_rounds,
        "mean_expected_revenue_per_round": total_revenue / n_rounds,
    }
