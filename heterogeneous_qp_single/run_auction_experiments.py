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


def run_experiment(
    *,
    data_dir: Path,
    organic_outputs_path: Path,
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
        summaries[mode_name] = summary

    combined_summary = {
        "scenario": "scenario_1",
        "summaries": summaries,
    }
    write_json(output_dir / "scenario_1_combined_summary.json", combined_summary)
    write_results_tables(output_dir, combined_summary)
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
            "expected_allocation": 0.0,
            "expected_revenue": 0.0,
            "nonzero_payments": [],
        }
        for source_id in source_ids
    }
    for round_log in rows_by_round:
        source_stats[round_log["sampled_winner"]]["sampled_count"] += 1
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


def write_results_tables(output_dir: Path, combined_summary: dict) -> None:
    rows = []
    mechanism_names = {
        "with_replacement": "Hetero QP w/ repl.",
        "without_replacement": "Hetero QP w/o repl.",
    }
    for key, summary in combined_summary["summaries"].items():
        rows.append(
            {
                "Mechanism": mechanism_names.get(key, key),
                "Revenue per Ad": summary["revenue_per_ad"],
                "Soc. Wel.": summary["social_welfare_per_response"],
                "Relevance": summary["mean_relevance_per_round"],
                "KL Div.": summary["mean_kl_per_round"],
                "Num. Ads": summary["expected_ads_per_response"],
            }
        )

    _write_table_files(output_dir, "scenario_1_results_table", rows)


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
        "\\begin{tabular}{lrrrrr}",
        "\\hline",
        " & ".join(headers) + r" \\",
        "\\hline",
    ]
    for row in rows:
        lines.append(" & ".join(_format_cell(row[header]) for header in headers) + r" \\")
    lines.extend(["\\hline", "\\end{tabular}", ""])
    return "\n".join(lines)


def _format_cell(value) -> str:
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
        "--click-scale",
        type=float,
        default=1.0,
        help="C_u normalization in ctr_iu = C_u * pi_iu. Default assumes C_u=1.",
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
    )
    summary = run_experiment(
        data_dir=args.data_dir,
        organic_outputs_path=args.organic_outputs,
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
