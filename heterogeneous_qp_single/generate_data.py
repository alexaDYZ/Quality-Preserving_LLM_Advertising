"""Generate synthetic data for the heterogeneous QP single extension."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.embedding_utils import Embedder
from src.preference_scores import compute_user_ad_scores, summarize_scores
from src.synthetic_advertisers import get_advertisers, get_bids_by_segment
from src.synthetic_users import classify_users, generate_user_memories


DEFAULT_QUERY = '"What can I visit on a trip to Hawaii?"'


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_dataset(
    output_dir: Path,
    n_users: int = 100,
    seed: int = 211,
    embedding_backend: str = "auto",
    model_name: str = "multi-qa-MiniLM-L6-cos-v1",
) -> dict:
    raw_users = generate_user_memories(n_users=n_users, seed=seed)
    advertisers = get_advertisers()
    bids_by_segment = get_bids_by_segment(advertisers)

    embedder = Embedder(model_name=model_name, backend=embedding_backend)

    user_texts = [u["memory_text"] for u in raw_users]
    ad_texts = [a["profile_text"] for a in advertisers]
    user_embeddings = embedder.encode(user_texts)
    ad_embeddings = embedder.encode(ad_texts)

    classified_users = classify_users(raw_users, user_embeddings)

    score_payload = compute_user_ad_scores(
        users=classified_users,
        advertisers=advertisers,
        ad_embeddings=ad_embeddings,
        seed=seed,
    )

    organic_reference = {
        "query": DEFAULT_QUERY,
        "q0_policy_v1": "shared organic content; not user-personalized",
        "notes": (
            "For comparability with Dai/Han QP single, v1 keeps q0 tied to the "
            "query/current-generation context rather than to each user memory. "
            "If personalized organic content is studied later, regenerate and "
            "save organic_doc and q0 per user or per segment."
        ),
    }

    validation_summary = summarize_scores(classified_users, advertisers, score_payload)
    validation_summary["embedding_backend"] = embedder.backend_used
    validation_summary["embedding_model"] = embedder.model_name
    validation_summary["seed"] = seed
    validation_summary["n_users"] = n_users

    write_json(output_dir / "users_raw_memories.json", raw_users)
    write_json(output_dir / "users_classified.json", classified_users)
    write_json(output_dir / "advertisers.json", advertisers)
    write_json(output_dir / "bids_by_segment.json", bids_by_segment)
    write_json(output_dir / "user_ad_scores.json", score_payload)
    write_json(output_dir / "organic_reference.json", organic_reference)
    write_json(output_dir / "validation_summary.json", validation_summary)

    return validation_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-users", type=int, default=100)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "generated_data",
    )
    parser.add_argument(
        "--embedding-backend",
        choices=["auto", "sentence_transformers", "hashing"],
        default="auto",
        help="Use sentence_transformers when available, otherwise deterministic hashing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_dataset(
        output_dir=args.output_dir,
        n_users=args.n_users,
        seed=args.seed,
        embedding_backend=args.embedding_backend,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

