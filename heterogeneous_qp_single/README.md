# Heterogeneous QP Single Data Generation

Standalone data-generation code for a consumer-heterogeneity extension of Dai/Han's QP single-allocation setup.

This folder does not modify the original notebooks or logs. It generates synthetic LLM memories, classifies users into budget/luxury segments, embeds user memories and advertiser profiles, and computes user-specific advertiser preference scores.

## Run

```bash
python3 heterogeneous_qp_single/generate_data.py
```

Outputs are written to:

```text
heterogeneous_qp_single/generated_data/
```

The generator first tries to use `sentence-transformers` with `multi-qa-MiniLM-L6-cos-v1`, matching the Dai/Han relevance model. If that dependency is unavailable, it falls back to a deterministic hashing embedder so the full pipeline can run without downloads.

## Files

- `users_raw_memories.json`: synthetic LLM memory profiles.
- `users_classified.json`: memory embeddings and inferred `k(u)`.
- `advertisers.json`: advertiser profiles and ad text.
- `bids_by_segment.json`: budget/luxury bid matrix.
- `user_ad_scores.json`: full `s_iu` matrix with score components.
- `validation_summary.json`: checks for duplicates, segment split, score variation, and targeting patterns.

