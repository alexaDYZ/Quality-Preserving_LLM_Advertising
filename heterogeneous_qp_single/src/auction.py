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
    platform_reserve_eta: float = 2.0
    platform_reserve_beta: float = 0.8
    revenue_metric: str = "eligible_payment_per_inserted_ad"


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
    welfare_reserves = {}
    platform_reserves = {}
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
    pi_not_outside = normalize_over(
        [source_id for source_id in pi_mnl if source_id != "outside"],
        {source_id: pi_mnl[source_id] for source_id in pi_mnl if source_id != "outside"},
    )
    organic_utility = utilities[ORGANIC_ID]

    eligible_ad_ids = []
    for advertiser in active_advertisers:
        advertiser_id = advertiser["advertiser_id"]
        pi_i = max(pi_mnl[advertiser_id], EPS)
        welfare_reserve = -config.mu * (utilities[advertiser_id] - organic_utility) / (
            config.click_scale * pi_i
        )
        platform_reserve = _platform_reserve(
            q0=source_relevance[ORGANIC_ID],
            qi=source_relevance[advertiser_id],
            config=config,
        )
        reserve = max(platform_reserve, welfare_reserve)
        welfare_reserves[advertiser_id] = welfare_reserve
        platform_reserves[advertiser_id] = platform_reserve
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

    if config.revenue_metric == "allocation_payment":
        revenue_terms = {
            advertiser_id: allocation_full[advertiser_id] * payment
            for advertiser_id, payment in payments.items()
        }
    elif config.revenue_metric == "allocation_ctr_payment":
        revenue_terms = {
            advertiser_id: allocation_full[advertiser_id] * ctr.get(advertiser_id, 0.0) * payment
            for advertiser_id, payment in payments.items()
        }
    elif config.revenue_metric == "sampled_normalized_pi_payment":
        revenue_terms = {
            advertiser_id: allocation_full[advertiser_id] * pi_not_outside[advertiser_id] * payment
            for advertiser_id, payment in payments.items()
        }
    elif config.revenue_metric == "eligible_payment_per_inserted_ad":
        revenue_terms = {
            advertiser_id: allocation_full[advertiser_id] * payment
            for advertiser_id, payment in payments.items()
        }
    else:
        raise ValueError(f"Unknown revenue metric: {config.revenue_metric!r}")
    expected_revenue = sum(revenue_terms.values())
    expected_ads = sum(allocation_full[advertiser["advertiser_id"]] for advertiser in advertisers)
    expected_relevance = sum(
        allocation[source_id] * source_relevance[source_id]
        for source_id in candidate_ids
    )
    kl = kl_divergence(allocation, pi_alloc)
    kl_alloc_vs_pi_not_outside = kl_divergence(allocation_full, pi_not_outside)
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
        "pi_not_outside": pi_not_outside,
        "pi_alloc": pi_alloc,
        "ctr": ctr,
        "bids": bids,
        "welfare_reserves": welfare_reserves,
        "platform_reserves": platform_reserves,
        "reserves": reserves,
        "allocation": allocation_full,
        "payments_per_click": payments_full,
        "xi_at_reserve": xi_at_reserve,
        "expected_revenue": expected_revenue,
        "revenue_terms": revenue_terms,
        "expected_ads": expected_ads,
        "expected_relevance": expected_relevance,
        "kl_alloc_vs_pi": kl,
        "kl_alloc_vs_pi_not_outside": kl_alloc_vs_pi_not_outside,
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


def _platform_reserve(q0: float, qi: float, config: AuctionConfig) -> float:
    return config.platform_reserve_eta * (q0 ** config.platform_reserve_beta) / max(qi, EPS)


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
        round_log["sampled_revenue"] = _sampled_revenue(round_log, config)
        round_log["sampled_revenue_terms"] = _sampled_revenue_terms(round_log, config)
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
    total_expected_revenue = sum(row["expected_revenue"] for row in rows)
    metric_revenue, revenue_per_ad = _reported_revenue(rows, config)
    total_ads = sum(row["expected_ads"] for row in rows)
    sampled_ads = sum(1 for row in rows if row["sampled_ad"])
    advertiser_metrics = _advertiser_metrics(rows, n_responses)
    ad_user_utility = _sampled_ad_user_utility(rows, n_responses)

    by_segment = {}
    for segment in sorted({trial["segment"] for trial in trials}):
        segment_rows = [row for row in rows if row["segment"] == segment]
        segment_ads = sum(row["expected_ads"] for row in segment_rows)
        by_segment[segment] = _summary_from_rows(
            segment_rows,
            segment_ads,
            len({row["user_id"] for row in segment_rows}),
            config,
        )

    by_advertiser = {}
    for advertiser in advertisers:
        advertiser_id = advertiser["advertiser_id"]
        advertiser_revenue = _advertiser_revenue(rows, advertiser_id, config)
        advertiser_ads = sum(
            row["allocation"].get(advertiser_id, 0.0)
            for row in rows
        )
        advertiser_sampled_ads = sum(
            1 for row in rows if row["sampled_winner"] == advertiser_id
        )
        advertiser_denominator = (
            advertiser_sampled_ads
            if config.revenue_metric in {
                "sampled_normalized_pi_payment",
                "eligible_payment_per_inserted_ad",
            }
            else advertiser_ads
        )
        by_advertiser[advertiser["name"]] = {
            "advertiser_id": advertiser_id,
            "reported_revenue": advertiser_revenue,
            "expected_revenue": sum(row["revenue_terms"].get(advertiser_id, 0.0) for row in rows),
            "expected_ads": advertiser_ads,
            "sampled_ads": advertiser_sampled_ads,
            **_advertiser_metrics(rows, n_responses, advertiser_id=advertiser_id),
            "revenue_per_ad": (
                advertiser_revenue / advertiser_denominator
                if advertiser_denominator > 0
                else 0.0
            ),
            "mean_allocation_per_round": advertiser_ads / n_rounds_total if n_rounds_total else 0.0,
        }

    return {
        "config": {
            "rho_q": config.rho_q,
            "rho_s": config.rho_s,
            "mu": config.mu,
            "lambda_tilde": config.lambda_tilde,
            "click_scale": config.click_scale,
            "platform_reserve_eta": config.platform_reserve_eta,
            "platform_reserve_beta": config.platform_reserve_beta,
            "revenue_metric_name": config.revenue_metric,
            "ctr_assumption": "ctr_iu = click_scale * pi_iu; default click_scale=1.",
            "reserve_policy": (
                "Effective reserve is max(platform reserve, welfare reserve), "
                "where platform reserve = eta * q0^beta / q_i."
            ),
            "revenue_metric": _revenue_metric_note(config.revenue_metric),
            "organic_sampling": "One saved no-ad output is sampled per user response round.",
        },
        "n_responses": n_responses,
        "n_rounds_total": n_rounds_total,
        "reported_revenue": metric_revenue,
        "expected_revenue": total_expected_revenue,
        "expected_ads": total_ads,
        "sampled_ads": sampled_ads,
        "revenue_per_ad": revenue_per_ad,
        "expected_ads_per_response": total_ads / n_responses if n_responses else 0.0,
        "sampled_ads_per_response": sampled_ads / n_responses if n_responses else 0.0,
        "ad_user_utility_per_response": ad_user_utility["per_response"],
        "ad_user_utility_per_inserted_ad": ad_user_utility["per_inserted_ad"],
        "social_welfare_per_response": sum(row["social_welfare"] for row in rows) / n_responses,
        "gross_social_welfare_per_response": (
            sum(row["gross_social_welfare"] for row in rows) / n_responses
        ),
        "mean_relevance_per_round": sum(row["expected_relevance"] for row in rows) / n_rounds_total,
        "mean_kl_per_round": sum(row["kl_alloc_vs_pi"] for row in rows) / n_rounds_total,
        "mean_kl_alloc_vs_pi_not_outside_per_round": (
            sum(row["kl_alloc_vs_pi_not_outside"] for row in rows) / n_rounds_total
        ),
        "advertiser_per_click_roi": advertiser_metrics["per_click_roi"],
        "advertiser_surplus_per_response": advertiser_metrics["surplus_per_response"],
        "allocated_ctr": advertiser_metrics["allocated_ctr"],
        "ctr_lift": advertiser_metrics["ctr_lift"],
        "mean_reported_revenue_per_round": metric_revenue / n_rounds_total,
        "mean_expected_revenue_per_round": total_expected_revenue / n_rounds_total,
        "by_segment": by_segment,
        "by_advertiser": by_advertiser,
    }


def _summary_from_rows(
    rows: list[dict],
    total_ads: float,
    n_responses: int,
    config: AuctionConfig,
) -> dict:
    n_rounds = len(rows)
    sampled_ads = sum(1 for row in rows if row["sampled_ad"])
    total_expected_revenue = sum(row["expected_revenue"] for row in rows)
    metric_revenue, revenue_per_ad = _reported_revenue(rows, config)
    advertiser_metrics = _advertiser_metrics(rows, n_responses)
    ad_user_utility = _sampled_ad_user_utility(rows, n_responses)
    return {
        "n_responses": n_responses,
        "n_rounds": n_rounds,
        "reported_revenue": metric_revenue,
        "expected_revenue": total_expected_revenue,
        "expected_ads": total_ads,
        "sampled_ads": sampled_ads,
        "revenue_per_ad": revenue_per_ad,
        "expected_ads_per_response": total_ads / n_responses if n_responses else 0.0,
        "sampled_ads_per_response": sampled_ads / n_responses if n_responses else 0.0,
        "ad_user_utility_per_response": ad_user_utility["per_response"],
        "ad_user_utility_per_inserted_ad": ad_user_utility["per_inserted_ad"],
        "social_welfare_per_response": sum(row["social_welfare"] for row in rows) / n_responses,
        "gross_social_welfare_per_response": (
            sum(row["gross_social_welfare"] for row in rows) / n_responses
        ),
        "mean_relevance_per_round": sum(row["expected_relevance"] for row in rows) / n_rounds,
        "mean_kl_per_round": sum(row["kl_alloc_vs_pi"] for row in rows) / n_rounds,
        "mean_kl_alloc_vs_pi_not_outside_per_round": (
            sum(row["kl_alloc_vs_pi_not_outside"] for row in rows) / n_rounds
        ),
        "advertiser_per_click_roi": advertiser_metrics["per_click_roi"],
        "advertiser_surplus_per_response": advertiser_metrics["surplus_per_response"],
        "allocated_ctr": advertiser_metrics["allocated_ctr"],
        "ctr_lift": advertiser_metrics["ctr_lift"],
        "mean_reported_revenue_per_round": metric_revenue / n_rounds,
        "mean_expected_revenue_per_round": total_expected_revenue / n_rounds,
    }


def _revenue_metric_note(revenue_metric: str) -> str:
    if revenue_metric == "allocation_payment":
        return "Expected revenue uses sum_i allocation_i * per_click_payment_i."
    if revenue_metric == "allocation_ctr_payment":
        return "Expected revenue uses sum_i allocation_i * ctr_i * per_click_payment_i."
    if revenue_metric == "sampled_normalized_pi_payment":
        return (
            "Revenue per ad averages normalized_pi_i * per_click_payment_i over sampled "
            "rounds with ad insertion, where normalized_pi_i = pi_i / (1 - pi_outside)."
        )
    if revenue_metric == "eligible_payment_per_inserted_ad":
        return (
            "Revenue per ad matches the Han-Dai log convention: sum nonzero "
            "payments over rounds with sampled ad insertion, divided by the number "
            "of sampled ad-insertion rounds."
        )
    return revenue_metric


def _sampled_revenue(round_log: dict, config: AuctionConfig) -> float:
    winner = round_log["sampled_winner"]
    if winner == ORGANIC_ID:
        return 0.0

    payment = round_log["payments_per_click"].get(winner, 0.0)
    if config.revenue_metric == "allocation_payment":
        return payment
    if config.revenue_metric == "allocation_ctr_payment":
        return round_log["ctr"].get(winner, 0.0) * payment
    if config.revenue_metric == "sampled_normalized_pi_payment":
        return round_log["pi_not_outside"].get(winner, 0.0) * payment
    if config.revenue_metric == "eligible_payment_per_inserted_ad":
        return sum(
            payment
            for payment in round_log["payments_per_click"].values()
            if payment
        )
    raise ValueError(f"Unknown revenue metric: {config.revenue_metric!r}")


def _sampled_revenue_terms(round_log: dict, config: AuctionConfig) -> dict[str, float]:
    winner = round_log["sampled_winner"]
    if winner == ORGANIC_ID:
        return {}
    if config.revenue_metric == "eligible_payment_per_inserted_ad":
        return {
            advertiser_id: payment
            for advertiser_id, payment in round_log["payments_per_click"].items()
            if payment
        }
    return {winner: round_log["sampled_revenue"]}


def _reported_revenue(rows: list[dict], config: AuctionConfig) -> tuple[float, float]:
    if config.revenue_metric in {
        "sampled_normalized_pi_payment",
        "eligible_payment_per_inserted_ad",
    }:
        total_revenue = sum(row.get("sampled_revenue", 0.0) for row in rows)
        sampled_ads = sum(1 for row in rows if row["sampled_ad"])
        return total_revenue, total_revenue / sampled_ads if sampled_ads > 0 else 0.0

    total_revenue = sum(row["expected_revenue"] for row in rows)
    total_ads = sum(row["expected_ads"] for row in rows)
    return total_revenue, total_revenue / total_ads if total_ads > 0 else 0.0


def _advertiser_revenue(rows: list[dict], advertiser_id: str, config: AuctionConfig) -> float:
    if config.revenue_metric in {
        "sampled_normalized_pi_payment",
        "eligible_payment_per_inserted_ad",
    }:
        return sum(
            row.get("sampled_revenue_terms", {}).get(advertiser_id, 0.0)
            for row in rows
        )
    return sum(row["revenue_terms"].get(advertiser_id, 0.0) for row in rows)


def _advertiser_metrics(
    rows: list[dict],
    n_responses: int,
    advertiser_id: str | None = None,
) -> dict[str, float]:
    expected_click_value = 0.0
    expected_click_spend = 0.0
    expected_click_surplus = 0.0
    expected_clicks = 0.0
    expected_impressions = 0.0
    random_ctr_denominator = 0.0

    for row in rows:
        advertiser_ids = (
            [advertiser_id]
            if advertiser_id is not None
            else row["active_advertiser_ids"]
        )
        active_ctrs = [
            row["ctr"].get(source_id, 0.0)
            for source_id in row["active_advertiser_ids"]
            if row["ctr"].get(source_id, 0.0) > 0
        ]
        random_ctr = sum(active_ctrs) / len(active_ctrs) if active_ctrs else 0.0

        row_expected_impressions = 0.0
        for source_id in advertiser_ids:
            payment = row["payments_per_click"].get(source_id, 0.0)
            if payment <= 0:
                continue
            allocation = row["allocation"].get(source_id, 0.0)
            ctr = row["ctr"].get(source_id, 0.0)
            value = row["bids"].get(source_id, 0.0)
            expected_click = allocation * ctr

            expected_click_value += expected_click * value
            expected_click_spend += expected_click * payment
            expected_click_surplus += expected_click * (value - payment)
            expected_clicks += expected_click
            expected_impressions += allocation
            row_expected_impressions += allocation

        random_ctr_denominator += row_expected_impressions * random_ctr

    return {
        "expected_click_value": expected_click_value,
        "expected_click_spend": expected_click_spend,
        "expected_clicks": expected_clicks,
        "per_click_roi": (
            expected_click_surplus / expected_click_spend
            if expected_click_spend > 0
            else 0.0
        ),
        "surplus_per_response": (
            expected_click_surplus / n_responses
            if n_responses > 0
            else 0.0
        ),
        "allocated_ctr": (
            expected_clicks / expected_impressions
            if expected_impressions > 0
            else 0.0
        ),
        "ctr_lift": (
            expected_clicks / random_ctr_denominator
            if random_ctr_denominator > 0
            else 0.0
        ),
    }


def _sampled_ad_user_utility(rows: list[dict], n_responses: int) -> dict[str, float]:
    total_utility = 0.0
    inserted_ads = 0
    for row in rows:
        winner = row["sampled_winner"]
        if winner == ORGANIC_ID:
            continue
        total_utility += row["utilities"].get(winner, 0.0)
        inserted_ads += 1
    return {
        "total": total_utility,
        "per_response": total_utility / n_responses if n_responses > 0 else 0.0,
        "per_inserted_ad": total_utility / inserted_ads if inserted_ads > 0 else 0.0,
    }
