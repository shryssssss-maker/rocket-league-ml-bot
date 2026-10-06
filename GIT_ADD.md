# Git staging guide

The root `.gitadd` file is the project's staging allowlist. It includes the project documentation and the training source, historical text, compact evaluation evidence, and reports under `training/`. It excludes local environments, model weights, and the large diagnostic captures listed in `.gitignore`.

The October research additions keep V0–V16 source, specifications, manifests, integrity hashes and review reports eligible for Git. Raw JSONL trajectories, tensor/array binaries, per-callback evidence, repeated summary publications, preflight output and model checkpoints are ignored. They remain on disk; ignoring them neither deletes them nor removes already-tracked files from Git. Frozen manifests may reference these local artifacts: a source checkout alone does not include the complete dataset or trained weights.

Both `python-example/` and `python-example-original/` are independent repositories and are excluded from root staging. The source-only RocketSim investigation checkout is excluded too. This changes only root repository bookkeeping; neither bot checkout nor its history is modified. `.gitadd` anchors the training path at the repository root; `.gitignore` filters generated output within it. Never use `git add -f` to bypass these exclusions without reviewing the exact files.

From the repository root, preview the exact paths first:

```powershell
git add --dry-run --all --pathspec-from-file=.gitadd
```

If the preview contains only the files you intend to version, stage them with:

```powershell
git add --all --pathspec-from-file=.gitadd
```

Then review the staged change summary before committing:

```powershell
git status --short
git diff --cached --stat
```

`.gitadd` is a plain Git pathspec list, not a built-in auto-staging feature. `python-example/` is intentionally omitted: it contains its own `.git` repository, so adding that directory from the root would stage only a repository pointer and leave the local bot changes out of the main project. Its own `.gitignore` now excludes local environment files and generated diagnostic logs. Note that `.env` is already tracked in that nested repository, so an ignore rule alone will not remove it from that repository's history or index.

Choose how `python-example` should belong to the root project before staging it there: preserve it as a deliberate submodule/subtree, or archive its history and flatten its source into the root repository. Until that decision is made, inspect its independent changes with:

```powershell
git -C python-example status --short
```
