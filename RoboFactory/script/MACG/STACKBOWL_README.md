# Bowl stacking evaluation

This is an additional dual-arm experiment in the convenience suite. For the full environment setup, use the [installation guide](../../../docs/installation.md). For model-server requirements, seeds, metrics, and phase behavior, use the [evaluation guide](../../../docs/evaluation.md).

From the repository root:

```bash
bash scripts/evaluate.sh --task stack_bowls \
  --server-url http://127.0.0.1:20011 --dry-run
bash scripts/evaluate.sh --task stack_bowls \
  --server-url http://127.0.0.1:20011
```

This runs four clean conditions (in-domain, reorder, sync, sync + reorder), each over seeds 1000–1099. It is included in the nine-family convenience suite; it is not part of the paper's eight-family ACG-Bench protocol.

The evaluator is `two_arms_stack_bowl/eval_stackbowl.py`; condition files live in that same directory. Use `two_arms_stack_bowl_12` with the base / sync condition and `two_arms_stack_bowl_21` with reordered conditions, as specified in `eval_scripts_rule_base/stack_bowls.sh`.

The launcher refuses existing video output directories. Archive results before another run. Optional VLM phase-decider experiments live in `eval_scripts_vlm/` and require separate credentials and setup; they are outside the rule-based workflow.
