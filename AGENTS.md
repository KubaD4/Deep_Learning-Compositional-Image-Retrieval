# Project Agent Instructions

This repository is a Deep Learning course project workspace for compositional image retrieval on CelebA.

## Start Here

Every future agent/chat working in this folder must first read:

1. `AGENTS.md`
2. `llm-wiki/index.md`
3. `PROJECT_STATE_AND_SOLUTION.md` for current status, selected proposal, and remaining work
4. `PROJECT_FOLDER_AND_TRAINING_SCHEMA.md` for target folder structure and batch/epoch data flow
5. `PROJECT_GUIDE.md` for detailed implementation, experiment, or onboarding instructions
6. Any other wiki pages linked from the index that match the task

For active website work, also read `llm-wiki/website-goal.md` and
`llm-wiki/monitoring-and-logs.md`. These are the handoff contract for the current
product goal, verified status, outstanding decisions, and operational logs.

The wiki is the persistent working memory for the project. Always keep it current whenever new files, results, decisions, experiments, or explanations are produced.

`llm-wiki/` is local-only by policy and has an ignore rule for new files. Some
legacy wiki files are still tracked by git; the ignore rule does not untrack
them. Do not stage, force-add, commit, or publish wiki changes unless the user
explicitly requests a policy change. Do not rewrite git history to resolve this
legacy state without a separate request.

## LLM Wiki Contract

This project follows Andrej Karpathy's LLM Wiki pattern:

- Raw sources are the original project files in the repository root and `celeba/`. Do not rewrite or reorganize them unless the user explicitly asks.
- `llm-wiki/` is the generated local knowledge layer. Agents own and maintain these Markdown pages, but it must remain outside GitHub.
- `AGENTS.md` is the operating schema. Update it if the workflow changes.
- `llm-wiki/index.md` is content-oriented. Update it whenever adding or changing wiki pages.
- `llm-wiki/log.md` is chronological. Append an entry for each ingest, major query, lint pass, setup change, or experiment summary.

When answering project questions, prefer the wiki first, then source Markdown in `llm-wiki/source-markdown/`, then original files.

## MarkItDown Workflow

Microsoft MarkItDown is installed in the project-local virtual environment:

```bash
source .venv/bin/activate
markitdown --version
```

The installed version is `markitdown 0.1.6`, with `[all]` optional dependencies recorded in `requirements.txt`.

Use MarkItDown when a source file is not convenient for direct agent reading, especially PDFs, notebooks, Word/PowerPoint/Excel files, HTML, or other document formats. Generated Markdown belongs in `llm-wiki/source-markdown/`.

Recommended command:

```bash
.venv/bin/markitdown "Some Source.pdf" -o "llm-wiki/source-markdown/Some Source.md"
```

After conversion:

- Link the generated Markdown from `llm-wiki/source-inventory.md`.
- Link relevant generated Markdown from any topical wiki page that depends on it.
- Append a conversion or ingest entry to `llm-wiki/log.md`.
- Treat generated Markdown as a readable derivative, not the authoritative raw source.

## Project-Specific Rules

- The assignment canonical source is `Project assignment - V1.2.pdf`. A byte-identical `Project assignment - V1.2-2.pdf` was inspected earlier and later removed; its historical Markdown derivative remains in the local wiki.
- Use the PyTorch `CelebA` dataset object for evaluation indices. Do not map JSON source keys directly to filenames.
- The expected model backbone for the final report is HuggingFace `openai/clip-vit-base-patch32`.
- Report Recall@K and Precision@K at K = 1, 5, and 10. Recall@K is the primary metric.
- The final deliverable is a single self-contained Google Colab notebook with code and report-style Markdown.
- There must be exactly one canonical final-best package: `final_best_system/`. When a better model is found, update this folder in place. Do not create `final_best_system_v2`, `final_best_system_v6_candidate`, or similar duplicate final folders unless the user explicitly asks for a temporary backup.

## Maintenance Checklist

When new work happens:

- Update or add the narrowest relevant wiki page.
- Update `llm-wiki/index.md`.
- Append to `llm-wiki/log.md`.
- When findings, results, or design decisions change, update the wiki and log in the same turn before handing work back.
- If a paper, blog post, or external reference materially informs a decision, save its link in the relevant wiki page with a one-line note explaining why it matters.
- Keep the wiki local-only. Do not suggest pushing it to GitHub unless the user explicitly changes that policy.
- If a new source file appears, add it to `llm-wiki/source-inventory.md`.
- If a source was converted with MarkItDown, link both the raw source and generated Markdown.

## Goal, Monitoring, And Run Logs

- Record the current requested outcome, acceptance checks, status, and next steps
  in the narrowest goal page, currently `llm-wiki/website-goal.md`.
- Separate facts verified by a command/test from assumptions or proposed work.
  Every live-status statement must say when it was checked and how to recheck it.
- After meaningful progress, update the goal page and append a dated entry to
  `llm-wiki/log.md` in the same turn. Keep prior log entries intact.
- Runtime logs are machine-generated evidence, not wiki prose. The website's
  operational log is `website/logs/app.jsonl`; it is local-only and append-only.
- When monitoring, record only meaningful transitions, failures, results, or
  required user decisions in the wiki. Do not turn unchanged polling into noise.
- Do not create a background monitoring schedule merely because this schema
  mentions monitoring. Use the app's automation mechanism only when the user
  actually requests recurring checks, reminders, or continued monitoring.
- Preserve raw sources, trained weights, and datasets. Do not duplicate or
  replace the canonical `final_best_system/` as part of website work.
