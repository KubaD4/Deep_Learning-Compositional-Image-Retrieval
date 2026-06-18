# MarkItDown Workflow

## Installed Tool

Microsoft MarkItDown is installed project-locally:

```bash
.venv/bin/markitdown --version
```

Current output:

```text
markitdown 0.1.6
```

The reproducible dependency is recorded in `requirements.txt`:

```text
markitdown[all]==0.1.6
```

## When To Use It

Use MarkItDown when a source file needs to become LLM-readable Markdown:

- PDFs
- PowerPoint slides
- Word documents
- Excel workbooks
- HTML pages
- notebooks or other document-like formats supported by the tool

For this project, it was used to convert:

- all course/project PDFs,
- `Project Skeleton.ipynb`.

## Where Output Goes

Generated Markdown should go in:

```text
llm-wiki/source-markdown/
```

Name the file after the source:

```bash
.venv/bin/markitdown "Project assignment - V1.2.pdf" \
  -o "llm-wiki/source-markdown/Project assignment - V1.2.md"
```

## How To Link It Into The Wiki

After converting a file:

1. Add or update its row in [Source Inventory](source-inventory.md).
2. Add a link from [Index](index.md) if it is a major source.
3. Link from topical pages that depend on it.
4. Append a log entry in [Log](log.md).

Example source inventory row:

```markdown
| `New Source.pdf` | [source-markdown/New Source.md](source-markdown/New%20Source.md) | What this source contributes. |
```

## Quality Notes

MarkItDown is a conversion aid, not a source of truth. PDF layout extraction can produce awkward tables, merged words, or noisy page breaks. If a claim matters, verify against:

- the original PDF,
- the notebook,
- the dataset file,
- or the assignment text.

Use converted Markdown for navigation and search, then cite or reason from the original source when precision matters.

