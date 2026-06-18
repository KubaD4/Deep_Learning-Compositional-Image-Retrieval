# Tools

This directory is reserved for local helper tools used during project preparation.

## MarkItDown

The local workstation currently uses Microsoft's MarkItDown to convert PDFs and notebooks into Markdown for `llm-wiki/`.

The full MarkItDown checkout and its virtual environment are intentionally not tracked in this repository because they are third-party/cache state. To recreate it locally:

```bash
cd tools
git clone https://github.com/microsoft/markitdown.git
cd markitdown
python3 -m venv .venv
source .venv/bin/activate
pip install "markitdown[all]"
```

Converted assignment/source text that agents need is tracked under:

```text
llm-wiki/source-markdown/
.llm-wiki-work/pdf_text/
```
