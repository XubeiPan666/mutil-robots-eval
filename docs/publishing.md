# Publishing a source release

[Documentation](README.md) · [Contributing](../CONTRIBUTING.md) · [Security](../SECURITY.md)

The public repository is [future-robots-eval](https://github.com/zhangzaibin/future-robots-eval). Keep a development repository with sensitive history private. Copying a `.git` directory, pushing an existing branch, or changing that repository's visibility can expose earlier versions of removed files.

## Validate and export

From the development checkout, run the [fast checks and unit tests](../CONTRIBUTING.md#local-validation), then export the current source tree to a new directory outside the checkout:

```bash
python scripts/check_release.py
python scripts/export_release.py /tmp/future-robots-eval-release
```

The exporter uses Git's tracked and non-ignored source list, skips deleted files, rejects symlinks, runs syntax / documentation / credential-pattern checks, and writes a SHA-256 `PUBLIC_RELEASE_MANIFEST.json`. It copies the current working-tree files, including newly authored documentation, and excludes `.git`, ignored credentials, downloaded assets, checkpoints, videos, and result directories. Review the file list before publishing; pattern checks do not detect every kind of sensitive content.

Create a fresh Git repository inside that exported directory and publish only its new initial commit. Do not add the development repository as a remote, fetch its branches, or push its refs. Preserve all license and attribution files. If a credential was ever committed, revoke it with its provider even when releasing only a clean snapshot.

## Release scope

The included source supports rule-based dual-arm simulator evaluation and a local per-arm proxy. External simulation assets and model checkpoints must be obtained separately. The convenience suite has nine families / 26 configured conditions; the paper defines eight families / 23 conditions. A paper-aligned result must identify its selected conditions and checkpoint, rather than relabeling the full convenience suite as the paper split.

CI covers source checks, launch behavior, progress metrics, and proxy routing. GPU simulation, asset availability, and complete policy rollouts require the documented Linux / GPU environment and are separate validation steps. Record the source revision, installed dependency versions, assets, checkpoint, task conditions, seeds, and complete command for each reported experiment.
