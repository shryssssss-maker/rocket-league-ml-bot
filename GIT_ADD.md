# Git staging guide

The root `.gitadd` file is the project's staging allowlist. It includes the project documentation and the training source, historical text, compact evaluation evidence, and reports under `training/`. It excludes local environments, model weights, and the large diagnostic captures listed in `.gitignore`.

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
