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
with `C_u = 1`. The effective reserve is
`max(platform_reserve, welfare_reserve)`, where the default platform reserve is
`2 * q0^0.8 / q_i`, matching Han-Dai Scenario 1. Both reserve components are
logged. By default, `Revenue per Ad` matches the Han-Dai log-implied table
metric: it sums nonzero payments over sampled rounds with ad insertion and
divides by the number of sampled ad-insertion rounds. Diagnostic runs can still
use `--revenue-metric sampled_normalized_pi_payment`,
`--revenue-metric allocation_payment`, or `--revenue-metric allocation_ctr_payment`.
For each user response round, one saved no-ad output is sampled from
`No-Ad Response/generated_no_ad_outputs_scenario_1.json` as the organic source.

Advertiser-facing metrics are reported separately from platform revenue. The
per-click ROI is `sum x_i * ctr_i * (v_i - p_i) / sum x_i * ctr_i * p_i`, with
truthful values `v_i = b_i` in Han-Dai and `v_ik = b_ik` in the heterogeneous
setting. Expected advertiser surplus per response is
`sum x_i * ctr_i * (v_i - p_i) / n_responses`. Allocated CTR is the
allocation-weighted click probability, and CTR lift compares it with a random
active-ad baseline for the same users and rounds.

The ad user-utility metrics report realized advertiser-content utility over
sampled ad insertions: `sum V_iu / n_responses` and
`sum V_iu / inserted_ads`. For Han-Dai QP single comparisons, saved Han-Dai
trials are deterministically shuffle-matched to the same 100 heterogeneous
users, so differences come from realized ad-user pairs rather than from changing
the user population.
