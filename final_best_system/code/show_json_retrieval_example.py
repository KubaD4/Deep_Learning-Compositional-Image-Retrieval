#!/usr/bin/env python3
"""Visualize one official JSON retrieval case for the final system.

The script does not recompute CLIP inference. It reads the already saved
retrievals.jsonl produced by the final system and renders two blocks:

1. source image + top-k predicted images;
2. a sample of official-valid JSON targets for that same source/query.

Predicted images use four border colors:

- blue: source/input image;
- light green: official-valid target from the JSON;
- yellow: satisfies the requested query attributes, but is not JSON-valid;
- red: does not satisfy at least one requested query attribute.

The bottom block is always official-valid.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METHOD_DIR = (
    ROOT
    / "final_best_system"
    / "results"
    / "final_best_model_plus_generic_delta_beta_1p50"
)
DEFAULT_OUTPUT_DIR = ROOT / "final_best_system" / "results" / "qualitative_examples"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render source image plus top-k retrievals for one official JSON query."
    )
    parser.add_argument("--query-id", type=int, default=12, help="Evaluation query id.")
    parser.add_argument(
        "--source-index",
        type=int,
        default=None,
        help="Test-set source index. If omitted, pick a source with a hit in top-k if possible.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="How many retrieved images to show.")
    parser.add_argument(
        "--valid-k",
        type=int,
        default=None,
        help="How many official-valid JSON targets to show below. Defaults to top-k.",
    )
    parser.add_argument(
        "--method-dir",
        type=Path,
        default=DEFAULT_METHOD_DIR,
        help="Directory containing retrievals.jsonl for the method to visualize.",
    )
    parser.add_argument(
        "--evaluation-json",
        type=Path,
        default=ROOT / "celeba_evaluation.json",
        help="Official evaluation JSON.",
    )
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=ROOT
        / "cluster"
        / "data"
        / "celeba"
        / "embeddings"
        / "openai_clip_vit_b32"
        / "test_image_embeddings.pt",
        help="Test image embedding cache containing filenames.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=ROOT / "celeba" / "img_align_celeba",
        help="CelebA aligned images directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where to save the rendered PNG and metadata JSON.",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Save the PNG without opening it with the macOS open command.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def load_filenames(cache_path: Path, images_dir: Path) -> list[str]:
    if cache_path.exists():
        try:
            import torch

            cache = torch.load(cache_path, map_location="cpu")
            filenames = cache.get("filenames")
            if filenames:
                return list(filenames)
        except Exception as exc:  # pragma: no cover - fallback path is for portability.
            print(f"Warning: could not load embedding cache filenames ({exc}); using split file.")

    split_file_candidates = [
        ROOT / "celeba" / "list_eval_partition.txt",
        ROOT / "cluster" / "data" / "celeba" / "annotations" / "list_eval_partition.txt",
        ROOT / "data" / "celeba" / "annotations" / "list_eval_partition.txt",
    ]
    split_file = next((p for p in split_file_candidates if p.exists()), None)
    if split_file is None:
        raise FileNotFoundError(
            "Could not find test filenames. Expected embedding cache or list_eval_partition.txt."
        )

    filenames: list[str] = []
    with split_file.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2 and parts[1] == "2":
                filenames.append(parts[0])
    if not filenames:
        raise RuntimeError(f"No test filenames found in {split_file}.")
    if not (images_dir / filenames[0]).exists():
        raise FileNotFoundError(f"Images not found under {images_dir}.")
    return filenames


def parse_query(query: str) -> list[tuple[int, str]]:
    conditions: list[tuple[int, str]] = []
    for raw_condition in query.split(","):
        token = raw_condition.strip()
        if not token:
            continue
        if token[0] not in "+-":
            raise ValueError(f"Invalid query condition: {raw_condition!r}")
        conditions.append((1 if token[0] == "+" else -1, token[1:].strip()))
    return conditions


def load_gallery_attributes(gallery_filenames: list[str]) -> tuple[list[str], dict[str, int], list[list[int]]]:
    """Load CelebA attributes and align them to test/gallery indices."""
    attr_candidates = [
        ROOT / "celeba" / "list_attr_celeba.txt",
        ROOT / "cluster" / "data" / "celeba" / "annotations" / "list_attr_celeba.txt",
        ROOT / "data" / "celeba" / "annotations" / "list_attr_celeba.txt",
    ]
    attr_path = next((p for p in attr_candidates if p.exists()), None)
    if attr_path is None:
        raise FileNotFoundError(
            "Could not find list_attr_celeba.txt; needed to color query-satisfied predictions."
        )

    file_to_attrs: dict[str, list[int]] = {}
    with attr_path.open(encoding="utf-8") as handle:
        _ = int(handle.readline().strip())
        attributes = [name for name in handle.readline().split() if name]
        for line in handle:
            parts = line.split()
            if not parts:
                continue
            file_to_attrs[parts[0]] = [1 if int(value) == 1 else -1 for value in parts[1:]]

    missing = [filename for filename in gallery_filenames if filename not in file_to_attrs]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} gallery filenames in {attr_path}")

    gallery_attrs = [file_to_attrs[filename] for filename in gallery_filenames]
    attribute_to_index = {name: index for index, name in enumerate(attributes)}
    return attributes, attribute_to_index, gallery_attrs


def failed_query_conditions(
    image_index: int,
    conditions: list[tuple[int, str]],
    attribute_to_index: dict[str, int],
    gallery_attrs: list[list[int]],
) -> list[str]:
    failed: list[str] = []
    attrs = gallery_attrs[image_index]
    for sign, attribute in conditions:
        attr_index = attribute_to_index[attribute]
        if attrs[attr_index] != sign:
            failed.append(f"{'+' if sign > 0 else '-'}{attribute}")
    return failed


def classify_prediction(
    image_index: int,
    valid_targets: set[int],
    conditions: list[tuple[int, str]],
    attribute_to_index: dict[str, int],
    gallery_attrs: list[list[int]],
) -> dict[str, Any]:
    failed = failed_query_conditions(image_index, conditions, attribute_to_index, gallery_attrs)
    official_valid = image_index in valid_targets
    query_ok = not failed
    if official_valid:
        status = "official_json_valid"
        color = (92, 196, 105)  # light green
        short_label = "JSON VALID"
    elif query_ok:
        status = "query_ok_not_json"
        color = (235, 180, 45)  # yellow/amber
        short_label = "query OK, not JSON"
    else:
        status = "query_fail"
        color = (205, 60, 55)  # red
        short_label = "query FAIL"
    return {
        "status": status,
        "official_valid": official_valid,
        "query_ok": query_ok,
        "failed_query_conditions": failed,
        "color": color,
        "short_label": short_label,
    }


def load_retrieval_rows(retrievals_path: Path, query_id: int) -> list[dict[str, Any]]:
    if not retrievals_path.exists():
        raise FileNotFoundError(f"Missing retrievals file: {retrievals_path}")

    rows: list[dict[str, Any]] = []
    with retrievals_path.open() as f:
        for line in f:
            row = json.loads(line)
            if int(row["query_id"]) == query_id:
                rows.append(row)
    if not rows:
        raise ValueError(f"No retrieval rows found for query_id={query_id} in {retrievals_path}")
    return rows


def choose_row(
    rows: list[dict[str, Any]],
    valid_by_source: dict[int, set[int]],
    source_index: int | None,
    top_k: int,
) -> dict[str, Any]:
    if source_index is not None:
        for row in rows:
            if int(row["source_index"]) == source_index:
                return row
        raise ValueError(f"source_index={source_index} not found for this query.")

    # Prefer a visually more informative example: at least one valid target in top-k.
    for row in rows:
        source = int(row["source_index"])
        top = [int(x) for x in row.get("top10", [])[:top_k]]
        if set(top).intersection(valid_by_source.get(source, set())):
            return row
    return rows[0]


def make_tile(
    image_path: Path,
    label: str,
    border: tuple[int, int, int],
    size: int,
    label_h: int,
) -> "Image.Image":
    from PIL import Image, ImageDraw

    image = Image.open(image_path).convert("RGB")
    image.thumbnail((size - 12, size - 12))

    tile = Image.new("RGB", (size, size + label_h), "white")
    draw = ImageDraw.Draw(tile)
    draw.rectangle([0, 0, size - 1, size + label_h - 1], outline=border, width=6)

    x = (size - image.width) // 2
    y = 8 + (size - 16 - image.height) // 2
    tile.paste(image, (x, y))

    draw.multiline_text((8, size + 6), label, fill=(20, 20, 20), spacing=2)
    return tile


def render_grid(
    output_path: Path,
    query_id: int,
    query: str,
    source_index: int,
    top_indices: list[int],
    valid_targets: set[int],
    shown_valid_targets: list[int],
    predicted_status: dict[int, dict[str, Any]],
    filenames: list[str],
    images_dir: Path,
    method_name: str,
) -> None:
    from PIL import Image, ImageDraw

    tile_size = 168
    label_h = 64
    gap = 14
    cols = 6
    title_h = 104
    section_h = 34
    predicted_count = 1 + len(top_indices)
    predicted_rows = (predicted_count + cols - 1) // cols
    valid_rows = (len(shown_valid_targets) + cols - 1) // cols
    rows = predicted_rows + valid_rows

    canvas_w = cols * tile_size + (cols + 1) * gap
    canvas_h = (
        title_h
        + section_h
        + predicted_rows * (tile_size + label_h)
        + valid_rows * (tile_size + label_h)
        + (rows + 3) * gap
    )
    canvas = Image.new("RGB", (canvas_w, canvas_h), (250, 250, 248))
    draw = ImageDraw.Draw(canvas)

    hits = set(top_indices).intersection(valid_targets)
    title = (
        f"query_id={query_id} | {query}\n"
        f"method={method_name} | source_index={source_index} | "
        f"hits@{len(top_indices)}={len(hits)} | "
        "blue=input | light green=JSON valid | yellow=query OK not JSON | red=query fail"
    )
    draw.multiline_text((gap, 18), title, fill=(15, 15, 15), spacing=5)

    predicted_items: list[tuple[int, str, tuple[int, int, int]]] = [
        (source_index, f"SOURCE\nidx={source_index}\n{filenames[source_index]}", (40, 105, 220))
    ]
    for rank, idx in enumerate(top_indices, start=1):
        status = predicted_status[idx]
        color = status["color"]
        failed = ", ".join(status["failed_query_conditions"][:2])
        if len(status["failed_query_conditions"]) > 2:
            failed += ", ..."
        status_line = status["short_label"] if not failed else f"{status['short_label']}: {failed}"
        label = f"rank {rank} {status_line}\nidx={idx}\n{filenames[idx]}"
        predicted_items.append((idx, label, color))

    valid_items = [
        (
            idx,
            f"JSON valid {rank}\nidx={idx}\n{filenames[idx]}",
            (92, 196, 105),
        )
        for rank, idx in enumerate(shown_valid_targets, start=1)
    ]

    y0 = title_h + gap
    draw.text((gap, y0), "Predicted by the final system", fill=(15, 15, 15))
    y0 += section_h

    for pos, (idx, label, color) in enumerate(predicted_items):
        r = pos // cols
        c = pos % cols
        x = gap + c * (tile_size + gap)
        y = y0 + r * (tile_size + label_h + gap)
        image_path = images_dir / filenames[idx]
        tile = make_tile(image_path, label, color, tile_size, label_h)
        canvas.paste(tile, (x, y))

    y1 = y0 + predicted_rows * (tile_size + label_h + gap) + gap
    draw.text(
        (gap, y1),
        f"Official-valid JSON targets for this same source/query (showing {len(valid_items)} of {len(valid_targets)})",
        fill=(15, 15, 15),
    )
    y1 += section_h

    for pos, (idx, label, color) in enumerate(valid_items):
        r = pos // cols
        c = pos % cols
        x = gap + c * (tile_size + gap)
        y = y1 + r * (tile_size + label_h + gap)
        image_path = images_dir / filenames[idx]
        tile = make_tile(image_path, label, color, tile_size, label_h)
        canvas.paste(tile, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> int:
    args = parse_args()

    try:
        from PIL import Image  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Missing Pillow. Install it in the project venv with: "
            ".venv/bin/python -m pip install pillow"
        ) from exc

    evaluation = load_json(args.evaluation_json)
    if not (0 <= args.query_id < len(evaluation)):
        raise ValueError(f"query_id must be in [0, {len(evaluation) - 1}]")

    query_entry = evaluation[args.query_id]
    query = query_entry["query"]
    conditions = parse_query(query)
    valid_by_source = {
        int(source): {int(x) for x in targets}
        for source, targets in query_entry["ground_truth"].items()
    }

    filenames = load_filenames(args.embedding_cache, args.images_dir)
    _, attribute_to_index, gallery_attrs = load_gallery_attributes(filenames)
    rows = load_retrieval_rows(args.method_dir / "retrievals.jsonl", args.query_id)
    row = choose_row(rows, valid_by_source, args.source_index, args.top_k)

    source_index = int(row["source_index"])
    top_indices = [int(x) for x in row["top10"][: args.top_k]]
    valid_targets = valid_by_source.get(source_index, set())
    method_name = row.get("method", args.method_dir.name)
    predicted_status = {
        idx: classify_prediction(idx, valid_targets, conditions, attribute_to_index, gallery_attrs)
        for idx in top_indices
    }
    valid_k = args.valid_k or args.top_k
    shown_valid_targets = [
        idx for idx in top_indices if idx in valid_targets
    ]
    shown_valid_targets.extend(
        idx for idx in sorted(valid_targets) if idx not in set(shown_valid_targets)
    )
    shown_valid_targets = shown_valid_targets[:valid_k]

    output_png = (
        args.output_dir
        / f"query_{args.query_id:02d}_source_{source_index}_top{len(top_indices)}.png"
    )
    render_grid(
        output_png,
        args.query_id,
        query,
        source_index,
        top_indices,
        valid_targets,
        shown_valid_targets,
        predicted_status,
        filenames,
        args.images_dir,
        method_name,
    )

    metadata = {
        "query_id": args.query_id,
        "query": query,
        "source_index": source_index,
        "source_filename": filenames[source_index],
        "top_indices": top_indices,
        "top_filenames": [filenames[i] for i in top_indices],
        "top_status": [
            {
                "index": idx,
                "filename": filenames[idx],
                **{
                    key: value
                    for key, value in predicted_status[idx].items()
                    if key not in {"color"}
                },
            }
            for idx in top_indices
        ],
        "valid_targets_in_top_k": sorted(set(top_indices).intersection(valid_targets)),
        "shown_valid_targets": shown_valid_targets,
        "shown_valid_filenames": [filenames[i] for i in shown_valid_targets],
        "num_valid_targets_for_source": len(valid_targets),
        "method_dir": str(args.method_dir),
        "output_png": str(output_png),
    }
    output_json = output_png.with_suffix(".json")
    output_json.write_text(json.dumps(metadata, indent=2))

    print(f"Saved: {output_png}")
    print(f"Saved: {output_json}")
    print(
        f"Query {args.query_id} ({query}) source={source_index}: "
        f"{len(metadata['valid_targets_in_top_k'])}/{len(top_indices)} valid in top-{len(top_indices)}"
    )

    if not args.no_open:
        subprocess.run(["open", str(output_png)], check=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
