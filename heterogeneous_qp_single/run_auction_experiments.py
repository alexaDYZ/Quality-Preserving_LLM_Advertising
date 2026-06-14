"""Run auction-only heterogeneous QP single simulations for Scenario 1."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from src.auction import (
    AuctionConfig,
    advertiser_relevance,
    build_score_lookup,
    infer_embedding_backend,
    load_no_ad_outputs,
    load_scenario_data,
    run_user_response,
    summarize_trials,
    write_json,
)
from src.embedding_utils import Embedder


HAN_DAI_SCENARIO_1_TABLE = {
    "with_replacement": {
        "Mechanism": "Han-Dai QP w/ repl.",
        "Revenue per Ad": 1.64,
        "Soc. Wel.": 5.90,
        "Relevance": 2.17,
        "KL Div.": 0.02,
        "KL x||pi": None,
        "Num. Ads": 2.15,
    },
    "without_replacement": {
        "Mechanism": "Han-Dai QP w/o repl.",
        "Revenue per Ad": 1.63,
        "Soc. Wel.": 5.42,
        "Relevance": 2.12,
        "KL Div.": 0.01,
        "KL x||pi": None,
        "Num. Ads": 1.67,
    },
}

HAN_DAI_SCENARIO_1_BIDS = {
    "SunWing Airlines": 3.0,
    "TropicStay": 3.0,
    "WanderBite": 2.0,
    "NovaSkin": 2.0,
    "GridPower Bank": 1.0,
}


def run_experiment(
    *,
    data_dir: Path,
    organic_outputs_path: Path,
    han_dai_log_dir: Path,
    output_dir: Path,
    n_rounds: int,
    replacement_mode: str,
    seed: int,
    embedding_backend: str,
    config: AuctionConfig,
) -> dict:
    data = load_scenario_data(data_dir)
    users = data["users"]
    advertisers = data["advertisers"]
    score_payload = data["user_ad_scores"]
    score_lookup, raw_score_bounds = build_score_lookup(score_payload)
    organic_outputs = load_no_ad_outputs(organic_outputs_path)

    backend = infer_embedding_backend(data, embedding_backend)
    model_name = data.get("validation_summary", {}).get(
        "embedding_model",
        "multi-qa-MiniLM-L6-cos-v1",
    )
    embedder = Embedder(model_name=model_name, backend=backend)

    query = data["organic_reference"].get("query", '"What can I visit on a trip to Hawaii?"')
    ad_relevance = advertiser_relevance(advertisers, query, embedder)
    write_advertiser_tables(
        output_dir,
        advertisers,
        data["bids_by_segment"],
        ad_relevance,
    )

    modes = (
        [True, False]
        if replacement_mode == "both"
        else [replacement_mode == "with"]
    )
    summaries = {}
    for with_replacement in modes:
        mode_name = "with_replacement" if with_replacement else "without_replacement"
        rng = random.Random(seed + (0 if with_replacement else 100_000))
        trials = [
            run_user_response(
                user=user,
                advertisers=advertisers,
                bids_by_segment=data["bids_by_segment"],
                score_lookup=score_lookup,
                ad_relevance=ad_relevance,
                organic_outputs=organic_outputs,
                query=query,
                embedder=embedder,
                score_payload=score_payload,
                raw_score_bounds=raw_score_bounds,
                n_rounds=n_rounds,
                with_replacement=with_replacement,
                rng=rng,
                config=config,
            )
            for user in users
        ]
        summary = summarize_trials(trials, advertisers, config)
        summary.update(
            {
                "data_dir": str(data_dir),
                "organic_outputs_path": str(organic_outputs_path),
                "query": query,
                "replacement_mode": mode_name,
                "n_rounds_per_response": n_rounds,
                "seed": seed,
                "embedding_backend_requested": embedding_backend,
                "embedding_backend_used": embedder.backend_used,
                "embedding_model": embedder.model_name,
                "ad_relevance_source": (
                    "han_dai_q_i_v1 when available; embedding cosine fallback otherwise"
                ),
            }
        )
        write_json(output_dir / f"scenario_1_{mode_name}_logs.json", trials)
        write_json(output_dir / f"scenario_1_{mode_name}_summary.json", summary)
        write_source_summary_table(output_dir, mode_name, trials, advertisers)
        write_advertiser_performance_table(output_dir, mode_name, summary)
        summaries[mode_name] = summary

    combined_summary = {
        "scenario": "scenario_1",
        "summaries": summaries,
        "han_dai_qp_single": load_han_dai_qp_single_summaries(
            han_dai_log_dir,
            replacement_mode,
            users=users,
            advertisers=advertisers,
            score_lookup=score_lookup,
            ad_relevance=ad_relevance,
            config=config,
            seed=seed,
        ),
    }
    write_json(output_dir / "scenario_1_combined_summary.json", combined_summary)
    write_results_tables(output_dir, combined_summary)
    write_advertiser_comparison_tables(output_dir, combined_summary)
    return combined_summary


def write_advertiser_tables(
    output_dir: Path,
    advertisers: list[dict],
    bids_by_segment: dict,
    ad_relevance: dict[str, float],
) -> None:
    rows = []
    for advertiser in advertisers:
        advertiser_id = advertiser["advertiser_id"]
        bids = bids_by_segment[advertiser_id]
        rows.append(
            {
                "Advertiser": advertiser["name"],
                "Target": advertiser["target_audience"],
                "Bid (budget)": float(bids["budget"]),
                "Bid (luxury)": float(bids["luxury"]),
                "q_i^(1)": ad_relevance[advertiser_id],
            }
        )
    _write_table_files(output_dir, "scenario_1_advertiser_table", rows)


def write_source_summary_table(
    output_dir: Path,
    mode_name: str,
    trials: list[dict],
    advertisers: list[dict],
) -> None:
    advertiser_ids = [advertiser["advertiser_id"] for advertiser in advertisers]
    id_to_name = {advertiser["advertiser_id"]: advertiser["name"] for advertiser in advertisers}
    source_ids = ["organic"] + advertiser_ids
    rows_by_round = [round_log for trial in trials for round_log in trial["rounds"]]
    n_rounds = len(rows_by_round)

    source_stats = {
        source_id: {
            "sampled_count": 0,
            "sampled_revenue": 0.0,
            "expected_allocation": 0.0,
            "expected_revenue": 0.0,
            "nonzero_payments": [],
        }
        for source_id in source_ids
    }
    for round_log in rows_by_round:
        source_stats[round_log["sampled_winner"]]["sampled_count"] += 1
        source_stats[round_log["sampled_winner"]]["sampled_revenue"] += round_log.get(
            "sampled_revenue",
            0.0,
        )
        for source_id in source_ids:
            source_stats[source_id]["expected_allocation"] += round_log["allocation"].get(source_id, 0.0)
            source_stats[source_id]["expected_revenue"] += round_log["revenue_terms"].get(source_id, 0.0)
            payment = round_log["payments_per_click"].get(source_id, 0.0)
            if payment:
                source_stats[source_id]["nonzero_payments"].append(payment)

    table_rows = []
    for source_id in source_ids:
        stats = source_stats[source_id]
        expected_allocation = stats["expected_allocation"]
        payments = stats["nonzero_payments"]
        table_rows.append(
            {
                "Source": id_to_name.get(source_id, "organic"),
                "Sampled Count": stats["sampled_count"],
                "Sampled Share": stats["sampled_count"] / n_rounds,
                "Sampled Revenue": stats["sampled_revenue"],
                "Revenue per Sampled Ad": (
                    stats["sampled_revenue"] / stats["sampled_count"]
                    if stats["sampled_count"] > 0
                    else 0.0
                ),
                "Expected Allocation": expected_allocation,
                "Allocation Share": expected_allocation / n_rounds,
                "Expected Revenue": stats["expected_revenue"],
                "Revenue per Expected Ad": (
                    stats["expected_revenue"] / expected_allocation
                    if expected_allocation > 0
                    else 0.0
                ),
                "Mean Nonzero Payment": (
                    sum(payments) / len(payments)
                    if payments
                    else 0.0
                ),
            }
        )
    _write_table_files(output_dir, f"scenario_1_{mode_name}_source_summary", table_rows)


def write_advertiser_performance_table(
    output_dir: Path,
    mode_name: str,
    summary: dict,
) -> None:
    table_rows = []
    for advertiser, metrics in summary["by_advertiser"].items():
        table_rows.append(
            {
                "Advertiser": advertiser,
                "Reported Revenue": metrics["reported_revenue"],
                "Sampled Ads": metrics["sampled_ads"],
                "Expected Clicks": metrics["expected_clicks"],
                "Expected Click Spend": metrics["expected_click_spend"],
                "Per-Click ROI": metrics["per_click_roi"],
                "Surplus per Response": metrics["surplus_per_response"],
                "Allocated CTR": metrics["allocated_ctr"],
                "CTR Lift": metrics["ctr_lift"],
            }
        )
    _write_table_files(output_dir, f"scenario_1_{mode_name}_advertiser_performance", table_rows)


def load_han_dai_qp_single_summaries(
    log_dir: Path,
    replacement_mode: str,
    users: list[dict],
    advertisers: list[dict],
    score_lookup: dict[tuple[str, str], float],
    ad_relevance: dict[str, float],
    config: AuctionConfig,
    seed: int,
) -> dict:
    modes = (
        ["with_replacement", "without_replacement"]
        if replacement_mode == "both"
        else ["with_replacement" if replacement_mode == "with" else "without_replacement"]
    )
    summaries = {}
    for mode_name in modes:
        log_path = log_dir / f"QP_single_{mode_name}_logs_scenario_1.json"
        if not log_path.exists():
            continue
        summaries[mode_name] = summarize_han_dai_qp_single_logs(
            log_path,
            mode_name,
            users=users,
            advertisers=advertisers,
            score_lookup=score_lookup,
            ad_relevance=ad_relevance,
            config=config,
            seed=seed,
        )
    return summaries


def summarize_han_dai_qp_single_logs(
    log_path: Path,
    mode_name: str,
    users: list[dict],
    advertisers: list[dict],
    score_lookup: dict[tuple[str, str], float],
    ad_relevance: dict[str, float],
    config: AuctionConfig,
    seed: int,
) -> dict:
    trials = json.loads(log_path.read_text(encoding="utf-8"))
    rows = [round_log for trial in trials for round_log in trial]
    inserted_rows = [row for row in rows if row.get("ad_injected")]
    n_responses = len(trials)
    sampled_ads = len(inserted_rows)
    user_assignments = _matched_users_for_han_dai_trials(users, mode_name, seed)
    utility_lookup = _advertiser_utility_lookup(
        users,
        advertisers,
        score_lookup,
        ad_relevance,
        config,
    )

    reported_revenue = sum(
        sum(float(payment) for payment in row.get("payments_dict", {}).values())
        for row in inserted_rows
    )
    advertiser_metrics = _han_dai_advertiser_metrics(rows, n_responses)
    ad_user_utility = _han_dai_ad_user_utility(
        trials,
        user_assignments,
        utility_lookup,
    )
    paper_row = HAN_DAI_SCENARIO_1_TABLE[mode_name]

    return {
        "source": "Han-Dai Scenario 1 paper table plus saved QP single logs",
        "log_path": str(log_path),
        "n_responses": n_responses,
        "n_rounds_total": len(rows),
        "sampled_ads": sampled_ads,
        "log_revenue_per_ad": reported_revenue / sampled_ads if sampled_ads else 0.0,
        "revenue_per_ad": paper_row["Revenue per Ad"],
        "social_welfare_per_response": paper_row["Soc. Wel."],
        "mean_relevance_per_round": paper_row["Relevance"],
        "mean_kl_per_round": paper_row["KL Div."],
        "mean_kl_alloc_vs_pi_not_outside_per_round": None,
        "expected_ads_per_response": paper_row["Num. Ads"],
        "ad_user_utility_per_response": ad_user_utility["per_response"],
        "ad_user_utility_per_inserted_ad": ad_user_utility["per_inserted_ad"],
        "advertiser_per_click_roi": advertiser_metrics["per_click_roi"],
        "advertiser_surplus_per_response": advertiser_metrics["surplus_per_response"],
        "allocated_ctr": advertiser_metrics["allocated_ctr"],
        "ctr_lift": advertiser_metrics["ctr_lift"],
        "by_advertiser": advertiser_metrics["by_advertiser"],
    }


def _han_dai_advertiser_metrics(rows: list[dict], n_responses: int) -> dict:
    aggregate = _empty_han_dai_metric_accumulator()
    by_advertiser = {
        advertiser: _empty_han_dai_metric_accumulator()
        for advertiser in HAN_DAI_SCENARIO_1_BIDS
    }

    for row in rows:
        payments = {
            advertiser: float(payment)
            for advertiser, payment in row.get("payments_dict", {}).items()
        }
        if not payments:
            continue

        paid_advertisers = list(payments)
        pseudo_ctrs = [
            float(row["q_tilde"].get(advertiser, 0.0))
            for advertiser in paid_advertisers
        ]
        random_ctr = sum(pseudo_ctrs) / len(pseudo_ctrs) if pseudo_ctrs else 0.0

        for advertiser, payment in payments.items():
            allocation = float(row["allocation"].get(advertiser, 0.0))
            pseudo_ctr = float(row["q_tilde"].get(advertiser, 0.0))
            value = HAN_DAI_SCENARIO_1_BIDS[advertiser]
            _update_han_dai_metric_accumulator(
                aggregate,
                allocation=allocation,
                pseudo_ctr=pseudo_ctr,
                value=value,
                payment=payment,
                random_ctr=random_ctr,
            )
            _update_han_dai_metric_accumulator(
                by_advertiser[advertiser],
                allocation=allocation,
                pseudo_ctr=pseudo_ctr,
                value=value,
                payment=payment,
                random_ctr=random_ctr,
            )

    aggregate_metrics = _finish_han_dai_metric_accumulator(aggregate, n_responses)
    aggregate_metrics["by_advertiser"] = {
        advertiser: _finish_han_dai_metric_accumulator(metrics, n_responses)
        for advertiser, metrics in by_advertiser.items()
    }
    return aggregate_metrics


def _matched_users_for_han_dai_trials(
    users: list[dict],
    mode_name: str,
    seed: int,
) -> list[dict]:
    matched_users = list(users)
    mode_offset = 0 if mode_name == "with_replacement" else 100_000
    rng = random.Random(seed + mode_offset)
    rng.shuffle(matched_users)
    return matched_users


def _advertiser_utility_lookup(
    users: list[dict],
    advertisers: list[dict],
    score_lookup: dict[tuple[str, str], float],
    ad_relevance: dict[str, float],
    config: AuctionConfig,
) -> dict[tuple[str, str], float]:
    lookup = {}
    name_to_id = {advertiser["name"]: advertiser["advertiser_id"] for advertiser in advertisers}
    for user in users:
        user_id = user["user_id"]
        for advertiser_name, advertiser_id in name_to_id.items():
            lookup[(user_id, advertiser_name)] = (
                config.rho_q * ad_relevance[advertiser_id]
                + config.rho_s * score_lookup[(user_id, advertiser_id)]
            )
    return lookup


def _han_dai_ad_user_utility(
    trials: list[list[dict]],
    user_assignments: list[dict],
    utility_lookup: dict[tuple[str, str], float],
) -> dict[str, float]:
    total_utility = 0.0
    inserted_ads = 0
    for trial, user in zip(trials, user_assignments):
        user_id = user["user_id"]
        for round_log in trial:
            if not round_log.get("ad_injected"):
                continue
            total_utility += utility_lookup.get((user_id, round_log["winner"]), 0.0)
            inserted_ads += 1
    return {
        "total": total_utility,
        "per_response": total_utility / len(trials) if trials else 0.0,
        "per_inserted_ad": total_utility / inserted_ads if inserted_ads > 0 else 0.0,
    }


def _empty_han_dai_metric_accumulator() -> dict[str, float]:
    return {
        "expected_click_value": 0.0,
        "expected_click_spend": 0.0,
        "expected_click_surplus": 0.0,
        "expected_clicks": 0.0,
        "expected_impressions": 0.0,
        "random_ctr_denominator": 0.0,
    }


def _update_han_dai_metric_accumulator(
    metrics: dict[str, float],
    *,
    allocation: float,
    pseudo_ctr: float,
    value: float,
    payment: float,
    random_ctr: float,
) -> None:
    expected_click = allocation * pseudo_ctr
    metrics["expected_click_value"] += expected_click * value
    metrics["expected_click_spend"] += expected_click * payment
    metrics["expected_click_surplus"] += expected_click * (value - payment)
    metrics["expected_clicks"] += expected_click
    metrics["expected_impressions"] += allocation
    metrics["random_ctr_denominator"] += allocation * random_ctr


def _finish_han_dai_metric_accumulator(metrics: dict[str, float], n_responses: int) -> dict:
    expected_click_spend = metrics["expected_click_spend"]
    expected_impressions = metrics["expected_impressions"]
    random_ctr_denominator = metrics["random_ctr_denominator"]
    return {
        "expected_click_value": metrics["expected_click_value"],
        "expected_click_spend": expected_click_spend,
        "expected_clicks": metrics["expected_clicks"],
        "per_click_roi": (
            metrics["expected_click_surplus"] / expected_click_spend
            if expected_click_spend > 0
            else 0.0
        ),
        "surplus_per_response": (
            metrics["expected_click_surplus"] / n_responses
            if n_responses > 0
            else 0.0
        ),
        "allocated_ctr": (
            metrics["expected_clicks"] / expected_impressions
            if expected_impressions > 0
            else 0.0
        ),
        "ctr_lift": (
            metrics["expected_clicks"] / random_ctr_denominator
            if random_ctr_denominator > 0
            else 0.0
        ),
    }


def write_results_tables(output_dir: Path, combined_summary: dict) -> None:
    rows = []
    mechanism_names = {
        "with_replacement": "Hetero QP w/ repl.",
        "without_replacement": "Hetero QP w/o repl.",
    }
    han_dai_names = {
        "with_replacement": "Han-Dai QP w/ repl.",
        "without_replacement": "Han-Dai QP w/o repl.",
    }
    for key, summary in combined_summary.get("han_dai_qp_single", {}).items():
        rows.append(
            {
                "Mechanism": han_dai_names.get(key, key),
                "Revenue per Ad": summary["revenue_per_ad"],
                "Soc. Wel.": summary["social_welfare_per_response"],
                "Relevance": summary["mean_relevance_per_round"],
                "KL Div.": summary["mean_kl_per_round"],
                "KL x||pi": summary["mean_kl_alloc_vs_pi_not_outside_per_round"],
                "Num. Ads": summary["expected_ads_per_response"],
                "Ad User Util./Resp.": summary["ad_user_utility_per_response"],
                "Ad User Util./Ad": summary["ad_user_utility_per_inserted_ad"],
                "Per-Click ROI": summary["advertiser_per_click_roi"],
                "Adv. Surplus/Resp.": summary["advertiser_surplus_per_response"],
                "Allocated CTR": summary["allocated_ctr"],
                "CTR Lift": summary["ctr_lift"],
            }
        )
    for key, summary in combined_summary["summaries"].items():
        rows.append(
            {
                "Mechanism": mechanism_names.get(key, key),
                "Revenue per Ad": summary["revenue_per_ad"],
                "Soc. Wel.": summary["social_welfare_per_response"],
                "Relevance": summary["mean_relevance_per_round"],
                "KL Div.": summary["mean_kl_per_round"],
                "KL x||pi": summary["mean_kl_alloc_vs_pi_not_outside_per_round"],
                "Num. Ads": summary["expected_ads_per_response"],
                "Ad User Util./Resp.": summary["ad_user_utility_per_response"],
                "Ad User Util./Ad": summary["ad_user_utility_per_inserted_ad"],
                "Per-Click ROI": summary["advertiser_per_click_roi"],
                "Adv. Surplus/Resp.": summary["advertiser_surplus_per_response"],
                "Allocated CTR": summary["allocated_ctr"],
                "CTR Lift": summary["ctr_lift"],
            }
        )

    _write_table_files(output_dir, "scenario_1_results_table", rows)


def write_advertiser_comparison_tables(output_dir: Path, combined_summary: dict) -> None:
    rows = []
    mode_names = {
        "with_replacement": "w/ repl.",
        "without_replacement": "w/o repl.",
    }
    for mode_name, summary in combined_summary.get("han_dai_qp_single", {}).items():
        for advertiser, metrics in summary["by_advertiser"].items():
            rows.append(
                _advertiser_comparison_row(
                    mechanism=f"Han-Dai QP {mode_names.get(mode_name, mode_name)}",
                    advertiser=advertiser,
                    metrics=metrics,
                )
            )
    for mode_name, summary in combined_summary["summaries"].items():
        for advertiser, metrics in summary["by_advertiser"].items():
            rows.append(
                _advertiser_comparison_row(
                    mechanism=f"Hetero QP {mode_names.get(mode_name, mode_name)}",
                    advertiser=advertiser,
                    metrics=metrics,
                )
            )
    _write_table_files(output_dir, "scenario_1_advertiser_performance_comparison", rows)


def _advertiser_comparison_row(
    *,
    mechanism: str,
    advertiser: str,
    metrics: dict,
) -> dict:
    return {
        "Mechanism": mechanism,
        "Advertiser": advertiser,
        "Expected Clicks": metrics["expected_clicks"],
        "Expected Click Spend": metrics["expected_click_spend"],
        "Per-Click ROI": metrics["per_click_roi"],
        "Surplus per Response": metrics["surplus_per_response"],
        "Allocated CTR": metrics["allocated_ctr"],
        "CTR Lift": metrics["ctr_lift"],
    }


def _write_table_files(output_dir: Path, stem: str, rows: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{stem}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    markdown_path = output_dir / f"{stem}.md"
    markdown_path.write_text(_markdown_table(rows), encoding="utf-8")

    latex_path = output_dir / f"{stem}.tex"
    latex_path.write_text(_latex_table(rows), encoding="utf-8")


def _markdown_table(rows: list[dict]) -> str:
    headers = list(rows[0])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row[header]) for header in headers) + " |")
    return "\n".join(lines) + "\n"


def _latex_table(rows: list[dict]) -> str:
    headers = list(rows[0])
    lines = [
        "\\begin{tabular}{" + "l" + "r" * (len(headers) - 1) + "}",
        "\\hline",
        " & ".join(headers) + r" \\",
        "\\hline",
    ]
    for row in rows:
        lines.append(" & ".join(_format_cell(row[header]) for header in headers) + r" \\")
    lines.extend(["\\hline", "\\end{tabular}", ""])
    return "\n".join(lines)


def _format_cell(value) -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    repo_dir = project_dir.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_dir / "generated_data" / "scenario_1",
        help="Scenario 1 generated data directory.",
    )
    parser.add_argument(
        "--organic-outputs",
        type=Path,
        default=repo_dir / "No-Ad Response" / "generated_no_ad_outputs_scenario_1.json",
        help="Saved Han-Dai no-ad/organic outputs; one output is sampled per round.",
    )
    parser.add_argument(
        "--han-dai-log-dir",
        type=Path,
        default=repo_dir / "QP single",
        help="Directory containing saved Han-Dai QP single logs for Scenario 1 comparison.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "generated_data" / "scenario_1" / "auction_results",
    )
    parser.add_argument("--n-rounds", type=int, default=3)
    parser.add_argument(
        "--replacement-mode",
        choices=["with", "without", "both"],
        default="both",
    )
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument(
        "--embedding-backend",
        choices=["data", "auto", "sentence_transformers", "hashing"],
        default="data",
        help=(
            "Use the backend recorded in validation_summary.json by default. "
            "Override only when regenerating or smoke-testing intentionally."
        ),
    )
    parser.add_argument("--rho-q", type=float, default=1.0)
    parser.add_argument("--rho-s", type=float, default=1.0)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--lambda-tilde", type=float, default=1.0)
    parser.add_argument(
        "--platform-reserve-eta",
        type=float,
        default=2.0,
        help="eta in platform reserve eta * q0^beta / q_i.",
    )
    parser.add_argument(
        "--platform-reserve-beta",
        type=float,
        default=0.8,
        help="beta in platform reserve eta * q0^beta / q_i.",
    )
    parser.add_argument(
        "--click-scale",
        type=float,
        default=1.0,
        help="C_u normalization in ctr_iu = C_u * pi_iu. Default assumes C_u=1.",
    )
    parser.add_argument(
        "--revenue-metric",
        choices=[
            "eligible_payment_per_inserted_ad",
            "sampled_normalized_pi_payment",
            "allocation_payment",
            "allocation_ctr_payment",
        ],
        default="eligible_payment_per_inserted_ad",
        help=(
            "eligible_payment_per_inserted_ad matches the Han-Dai log-implied "
            "revenue table; sampled_normalized_pi_payment, allocation_payment, "
            "and allocation_ctr_payment are diagnostic alternatives."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AuctionConfig(
        rho_q=args.rho_q,
        rho_s=args.rho_s,
        mu=args.mu,
        lambda_tilde=args.lambda_tilde,
        click_scale=args.click_scale,
        platform_reserve_eta=args.platform_reserve_eta,
        platform_reserve_beta=args.platform_reserve_beta,
        revenue_metric=args.revenue_metric,
    )
    summary = run_experiment(
        data_dir=args.data_dir,
        organic_outputs_path=args.organic_outputs,
        han_dai_log_dir=args.han_dai_log_dir,
        output_dir=args.output_dir,
        n_rounds=args.n_rounds,
        replacement_mode=args.replacement_mode,
        seed=args.seed,
        embedding_backend=args.embedding_backend,
        config=config,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
