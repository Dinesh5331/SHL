"""
eval — Evaluation harness for the SHL Assessment Recommender Agent.

Usage
-----
    python -m eval.run_eval                         # local server
    python -m eval.run_eval --base-url <URL>        # deployed instance
    python -m eval.run_eval --base-url <URL> --k 5 # Recall@5

Probes
------
short_scenarios   - 4 specific single-turn queries → must return recommendations
long_multiturn    - clarify → recommend → refine × 2 chain
compare           - compare turn returns text only, no shortlist mutation
broad_offtopic    - 10 genuinely off-topic prompts → must return 0 recommendations
injection_variants - 4 injection attempts → must return 0 recommendations

Metrics
-------
Recall@K = |relevant ∩ returned[:K]| / |relevant|
Mean Recall@K = (1/N) × Σ Recall@K_i

Fill EXPECTED_ITEMS in run_eval.py with manually-verified catalog item names
to get a real Recall@K score.
"""
