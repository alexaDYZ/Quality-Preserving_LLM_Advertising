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

## Scenario 1 Auction Simulation

After generating Scenario 1 data, run the auction-only simulation with:

```bash
cd heterogeneous_qp_single
python3 -B run_auction_experiments.py \
  --data-dir generated_data/scenario_1 \
  --output-dir generated_data/scenario_1/auction_results \
  --replacement-mode both
```

This writes:

- `scenario_1_with_replacement_logs.json`
- `scenario_1_with_replacement_summary.json`
- `scenario_1_without_replacement_logs.json`
- `scenario_1_without_replacement_summary.json`
- `scenario_1_combined_summary.json`

The simulation assumes truthful bidding, `v_ik = b_ik`, and `ctr_iu = pi_iu`
with `C_u = 1`. The effective reserve is `max(0, theoretical_reserve)`;
both values are logged. To match the Han-Dai log-implied revenue metric,
expected revenue is computed as `sum_i allocation_i * per_click_payment_i`,
without an additional CTR multiplier. For each user response round, one saved no-ad output is sampled from
`No-Ad Response/generated_no_ad_outputs_scenario_1.json` as the organic source.
