# Contributing

Start with the [code map](docs/architecture.md) and the [evaluation protocol](docs/evaluation.md). Issues and pull requests are welcome for simulator bugs, clearer setup instructions, new task configurations, and reproducibility improvements.

When reporting a rollout issue, include the source revision, task / condition, seed, simulator / Python / GPU versions, command, checkpoint configuration and step, and the relevant traceback. Remove credentials and private server addresses from logs. A full-task failure without a process error is an evaluation result; describe the physical behavior when reporting it as a suspected evaluator bug.

## Local validation

The fast checks do not require model weights, downloaded assets, or a GPU:

```bash
python scripts/check_release.py
python -m unittest discover -s tests -v
PYTHONPATH=RoboFactory python -m unittest discover \
  -s RoboFactory/script/MACG/eval_common/tests -v
PYTHONPATH=RoboFactory/script/MACG/eval_scripts_rule_base_peram \
  python -m unittest discover \
  -s RoboFactory/script/MACG/eval_scripts_rule_base_peram -p 'test_*.py' -v
```

The proxy tests require NumPy and requests. CI runs these checks with Python 3.9 and 3.11. They verify launcher behavior, progress semantics, and per-arm input / action routing; they do not validate GPU simulation or a checkpoint's task performance.

For a runtime change, also run an affected single episode in the simulator and record the seed and configuration. For a scoring change, explain the physical predicate, order / synchronization semantics, and the effect on published comparisons. Include a focused regression test when the change corrects an observable failure.

## Pull requests

Keep changes focused. Preserve task names, seed budgets, prompt text, milestone predicates, and sync tolerances unless a protocol change is intentional and documented. Retain upstream notices and add provenance for copied components. Keep downloaded assets, checkpoints, result files, credentials, and machine-specific paths out of commits.

Update the relevant documentation when changing an entry point, requirement, configuration, output schema, or evaluation condition. Separate new experiments from the documented clean suite until their assumptions and reporting protocol are specified.

Contributions are accepted under the repository's MIT license, subject to the existing licenses of third-party components. See [third-party notices](THIRD_PARTY_NOTICES.md).
