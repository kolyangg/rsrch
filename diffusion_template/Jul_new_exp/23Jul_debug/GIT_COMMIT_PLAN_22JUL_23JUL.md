# Smart commit plan for `22Jul_debug` and `23Jul_debug`

## Keep in Git

- experiment and architecture source code;
- launch, validation, audit, reporting, and scheduling scripts;
- notebooks used to define or reproduce the work;
- architecture/spec JSON files and small data split manifests;
- Markdown plans, handoffs, architecture cards, experiment logs, and findings;
- compact CSV/JSON metric summaries exported outside ignored run folders.

## Keep out of Git

- complete `experiments`, `experiments_4k`, dry-run, smoke-test, and scheduler
  runtime directories;
- checkpoints and weights (`.pth`, `.pt`);
- generated validation images and contact sheets;
- image-heavy PDFs and visual-report directories;
- transient logs, NumPy dumps, and caches.

The root `/home/niko/rsrch/.gitignore` contains paths scoped specifically to
the two July lab folders. It does not globally ignore research PDFs, images,
or checkpoints elsewhere in the repository.

## Result preservation before commit

Run directories are the authoritative local/Comet artifacts but are ignored
by Git. Before the final commit:

1. finish the 4k queue;
2. consolidate each run's configuration, Comet key, pairing audit verdict,
   best checkpoint, full metric trajectory, and visual verdict into
   `EXPERIMENT_LOG_4K.md` and compact top-level CSV/JSON tables;
3. ensure every architecture has a reproducible registry entry/card;
4. use `git check-ignore -v` to verify large files are excluded;
5. inspect `git status` and stage the two debug folders plus the root
   `.gitignore` intentionally—do not use a repository-wide blind add.

Suggested final staging shape:

```bash
git add .gitignore \
  diffusion_template/Jul_new_exp/22Jul_debug \
  diffusion_template/Jul_new_exp/23Jul_debug
git status --short
git diff --cached --stat
```

No commit should be created until the 4k results have been consolidated and
the staged file/size audit is clean.
