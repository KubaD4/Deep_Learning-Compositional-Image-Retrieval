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
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT / "final_best_system"
CODE_ROOT = ROOT / "final_best_system" / "code"
DEFAULT_METHOD_DIR = (
    ROOT
    / "final_best_system"
    / "results"
    / "probe_reranker_v2_calibrated"
    / "A_cal_query_hardh2_accuracy"
)
DEFAULT_OUTPUT_DIR = ROOT / "final_best_system" / "results" / "qualitative_examples"
DEFAULT_FREE_OUTPUT_DIR = ROOT / "final_best_system" / "results" / "free_query_examples"
DEFAULT_CHECKPOINT = ROOT / "final_best_system" / "weights" / "best_val_official_like_at10.pt"
DEFAULT_PROMPT_CACHE = (
    ROOT
    / "final_best_system"
    / "data"
    / "celeba"
    / "embeddings"
    / "openai_clip_vit_b32"
    / "signed_attribute_prompt_embeddings_v2_photo_templates.pt"
)
DEFAULT_TEST_EMBEDDINGS = (
    PACKAGE_ROOT
    / "data"
    / "celeba"
    / "embeddings"
    / "openai_clip_vit_b32"
    / "test_image_embeddings.pt"
)
DEFAULT_EVALUATION_JSON = PACKAGE_ROOT / "data" / "celeba_evaluation.json"
DEFAULT_PROBE_RESULTS = (
    ROOT
    / "final_best_system"
    / "results"
    / "probe_reranker_v2_calibrated"
)
DEFAULT_PROBE_CHECKPOINT = DEFAULT_PROBE_RESULTS / "probe" / "best_probe.pt"


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
        default=DEFAULT_EVALUATION_JSON,
        help="Official evaluation JSON.",
    )
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=DEFAULT_TEST_EMBEDDINGS,
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
    parser.add_argument(
        "--free-query",
        type=str,
        default=None,
        help=(
            "Free-form signed query, e.g. '+young -lipstick'. "
            "When set, the script computes retrieval instead of reading official JSON rows."
        ),
    )
    parser.add_argument(
        "--free-source-index",
        type=int,
        default=None,
        help="Gallery/test source index for --free-query mode.",
    )
    parser.add_argument(
        "--free-image",
        type=Path,
        default=None,
        help=(
            "Image for --free-query mode. Can be a path, a CelebA filename, or a numeric gallery index."
        ),
    )
    parser.add_argument(
        "--free-output-dir",
        type=Path,
        default=DEFAULT_FREE_OUTPUT_DIR,
        help="Where to save free-query visualizations.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Learned gate checkpoint used in --free-query mode.",
    )
    parser.add_argument(
        "--prompt-cache",
        type=Path,
        default=DEFAULT_PROMPT_CACHE,
        help="Signed attribute prompt embedding cache used in --free-query mode.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.25,
        help="Vector-delta correction weight for --free-query mode.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="Device for --free-query mode.",
    )
    parser.add_argument(
        "--top-pool",
        type=int,
        default=500,
        help="Candidate pool before probe filtering in --free-query mode.",
    )
    parser.add_argument(
        "--no-probe-filter",
        action="store_true",
        help="Disable the final probe query/Hamming filter in --free-query mode.",
    )
    parser.add_argument(
        "--probe-results",
        type=Path,
        default=DEFAULT_PROBE_RESULTS,
        help="Probe result folder containing calibrated thresholds and test probabilities.",
    )
    parser.add_argument(
        "--probe-checkpoint",
        type=Path,
        default=DEFAULT_PROBE_CHECKPOINT,
        help="Probe checkpoint, needed for external free images.",
    )
    parser.add_argument(
        "--threshold-objective",
        default="accuracy",
        help="Threshold objective for calibrated probe filtering, usually 'accuracy'.",
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
        PACKAGE_ROOT / "data" / "celeba" / "annotations" / "list_eval_partition.txt",
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


def normalize_attribute_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def resolve_attribute_name(raw_attribute: str, attribute_to_index: dict[str, int]) -> str:
    alias = {
        "glasses": "Eyeglasses",
        "eyeglass": "Eyeglasses",
        "eyeglasses": "Eyeglasses",
        "specs": "Eyeglasses",
        "smile": "Smiling",
        "smiling": "Smiling",
        "young": "Young",
        "old": "Young",
        "older": "Young",
        "lipstick": "Wearing_Lipstick",
        "wearinglipstick": "Wearing_Lipstick",
        "makeup": "Heavy_Makeup",
        "heavymakeup": "Heavy_Makeup",
        "hat": "Wearing_Hat",
        "wearinghat": "Wearing_Hat",
        "blond": "Blond_Hair",
        "blonde": "Blond_Hair",
        "blondhair": "Blond_Hair",
        "blackhair": "Black_Hair",
        "brownhair": "Brown_Hair",
        "grayhair": "Gray_Hair",
        "greyhair": "Gray_Hair",
        "beard": "No_Beard",
        "nobeard": "No_Beard",
        "mustache": "Mustache",
        "moustache": "Mustache",
        "male": "Male",
        "female": "Male",
        "chubby": "Chubby",
    }
    key = normalize_attribute_key(raw_attribute)
    if key in alias:
        return alias[key]
    normalized = {normalize_attribute_key(name): name for name in attribute_to_index}
    if key in normalized:
        return normalized[key]
    suggestions = [
        name
        for name in attribute_to_index
        if key in normalize_attribute_key(name) or normalize_attribute_key(name) in key
    ][:8]
    extra = f" Did you mean one of: {', '.join(suggestions)}?" if suggestions else ""
    raise ValueError(f"Unknown CelebA attribute {raw_attribute!r}.{extra}")


def parse_free_query(query: str, attribute_to_index: dict[str, int]) -> list[dict[str, Any]]:
    """Parse '+young -lipstick' and keep unknown attributes as open CLIP text."""
    compact = query.replace(",", " ")
    matches = re.findall(r"([+-])\s*([^+-]+?)(?=\s+[+-]|$)", compact)
    if not matches:
        raise ValueError(
            "Free query must contain signed attributes, e.g. '+young -lipstick'."
        )
    conditions = []
    for sign_text, raw_attr in matches:
        raw_attr = raw_attr.strip()
        sign = 1 if sign_text == "+" else -1
        known = True
        try:
            attr = resolve_attribute_name(raw_attr, attribute_to_index)
        except ValueError:
            known = False
            attr = raw_attr.replace("_", " ").strip()
        else:
            # Friendly convention: '-beard' means request No_Beard=+1.
            if normalize_attribute_key(raw_attr) == "beard" and sign < 0:
                sign = 1
                attr = "No_Beard"
            # Friendly convention: '+female' means request Male=-1.
            if normalize_attribute_key(raw_attr) == "female" and sign > 0:
                sign = -1
                attr = "Male"
        conditions.append(
            {
                "sign": sign,
                "attribute": attr,
                "raw": raw_attr,
                "known": known,
            }
        )
    return conditions


def known_free_conditions(free_conditions: list[dict[str, Any]]) -> list[tuple[int, str]]:
    return [
        (int(condition["sign"]), str(condition["attribute"]))
        for condition in free_conditions
        if bool(condition["known"])
    ]


def format_free_condition(condition: dict[str, Any]) -> str:
    sign = "+" if int(condition["sign"]) > 0 else "-"
    label = str(condition["attribute"])
    suffix = "" if bool(condition["known"]) else " (open CLIP)"
    return f"{sign}{label}{suffix}"


def load_gallery_attributes(gallery_filenames: list[str]) -> tuple[list[str], dict[str, int], list[list[int]]]:
    """Load CelebA attributes and align them to test/gallery indices."""
    attr_candidates = [
        PACKAGE_ROOT / "data" / "celeba" / "annotations" / "list_attr_celeba.txt",
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


def choose_device(requested: str):
    import torch

    if requested != "auto":
        device = torch.device(requested)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but torch.backends.mps.is_available() is False")
    return device


def load_embedding_cache(cache_path: Path) -> dict[str, Any]:
    import torch

    if not cache_path.exists():
        raise FileNotFoundError(f"Missing embedding cache: {cache_path}")
    try:
        return torch.load(cache_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(cache_path, map_location="cpu")


def resolve_prompt_cache(path: Path) -> Path:
    candidates = [
        path,
        ROOT
        / "final_best_system"
        / "data"
        / "celeba"
        / "embeddings"
        / "openai_clip_vit_b32"
        / path.name,
        ROOT / "final_best_system" / "embeddings" / path.name,
        ROOT
        / "cluster"
        / "data"
        / "celeba"
        / "embeddings"
        / "openai_clip_vit_b32"
        / path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing prompt cache. Checked: {', '.join(map(str, candidates))}")


def setup_model_imports() -> None:
    os.environ.setdefault("DL_PROJECT_ROOT", str(PACKAGE_ROOT))
    for path in [
        CODE_ROOT / "scripts",
        CODE_ROOT / "orchestrator",
        CODE_ROOT / "probe_filtering",
    ]:
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)


def resolve_free_source(
    args: argparse.Namespace,
    gallery_cache: dict[str, Any],
    filenames: list[str],
):
    import torch
    import torch.nn.functional as F

    gallery_embeddings = F.normalize(gallery_cache["embeddings"].float(), dim=-1)
    if args.free_source_index is not None and args.free_image is not None:
        raise ValueError("Use either --free-source-index or --free-image, not both.")

    if args.free_source_index is not None:
        source_index = int(args.free_source_index)
        if not 0 <= source_index < len(filenames):
            raise ValueError(f"--free-source-index must be in [0, {len(filenames) - 1}]")
        return {
            "embedding": gallery_embeddings[source_index : source_index + 1],
            "source_index": source_index,
            "source_label": f"idx={source_index}\n{filenames[source_index]}",
            "source_image_path": args.images_dir / filenames[source_index],
            "external": False,
        }

    if args.free_image is None:
        raise ValueError("--free-query requires --free-source-index or --free-image.")

    token = str(args.free_image)
    if token.isdigit():
        source_index = int(token)
        if not 0 <= source_index < len(filenames):
            raise ValueError(f"Numeric --free-image must be in [0, {len(filenames) - 1}]")
        return {
            "embedding": gallery_embeddings[source_index : source_index + 1],
            "source_index": source_index,
            "source_label": f"idx={source_index}\n{filenames[source_index]}",
            "source_image_path": args.images_dir / filenames[source_index],
            "external": False,
        }

    image_path = args.free_image
    if not image_path.exists():
        by_filename = {filename: idx for idx, filename in enumerate(filenames)}
        if image_path.name in by_filename:
            source_index = by_filename[image_path.name]
            return {
                "embedding": gallery_embeddings[source_index : source_index + 1],
                "source_index": source_index,
                "source_label": f"idx={source_index}\n{filenames[source_index]}",
                "source_image_path": args.images_dir / filenames[source_index],
                "external": False,
            }
        candidate = args.images_dir / image_path.name
        if candidate.exists():
            image_path = candidate
        else:
            raise FileNotFoundError(f"Could not find --free-image {args.free_image}")

    source_embedding = compute_clip_image_embedding(image_path, args.device)
    return {
        "embedding": source_embedding.cpu(),
        "source_index": None,
        "source_label": f"external\n{image_path.name}",
        "source_image_path": image_path,
        "external": True,
    }


def compute_clip_image_embedding(image_path: Path, requested_device: str):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    device = choose_device(requested_device)
    model_id = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device)
    model.eval()
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=[image], return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        embedding = model.get_image_features(**inputs)
    return F.normalize(embedding.float(), dim=-1).cpu()


def compute_clip_text_embeddings(prompts: list[str], device):
    import torch
    import torch.nn.functional as F
    from transformers import CLIPModel, CLIPTokenizer

    model_id = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device)
    model.eval()
    inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        embeddings = model.get_text_features(**inputs)
    if hasattr(embeddings, "pooler_output"):
        embeddings = embeddings.pooler_output
    return F.normalize(embeddings.float(), dim=-1)


def open_attribute_prompt_pair(label: str) -> tuple[str, str]:
    readable = label.replace("_", " ").strip().lower()
    return (
        f"a portrait photo of a face with {readable}",
        f"a portrait photo of a face without {readable}",
    )


def build_free_condition_embeddings(
    free_conditions: list[dict[str, Any]],
    prompt_cache: dict[str, Any],
    attribute_to_index: dict[str, int],
    device,
):
    import torch
    import torch.nn.functional as F

    directions = F.normalize(prompt_cache["directions"].float(), dim=-1).to(device)
    condition_vectors = []
    open_positions: list[int] = []
    open_prompts: list[str] = []

    for position, condition in enumerate(free_conditions):
        if condition["known"]:
            attr_index = attribute_to_index[str(condition["attribute"])]
            direction = directions[attr_index] * int(condition["sign"])
            condition_vectors.append(direction)
        else:
            positive, negative = open_attribute_prompt_pair(str(condition["attribute"]))
            open_positions.append(position)
            open_prompts.extend([positive, negative])
            condition_vectors.append(torch.empty_like(directions[0]))

    if open_prompts:
        text_embeddings = compute_clip_text_embeddings(open_prompts, device)
        for pair_index, position in enumerate(open_positions):
            positive = text_embeddings[2 * pair_index]
            negative = text_embeddings[2 * pair_index + 1]
            direction = F.normalize(positive - negative, dim=-1)
            sign = int(free_conditions[position]["sign"])
            condition_vectors[position] = sign * direction

    conditions = torch.stack(condition_vectors, dim=0).unsqueeze(0)
    mask = torch.ones((1, len(condition_vectors)), dtype=torch.bool, device=device)
    return F.normalize(conditions.float(), dim=-1), mask


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


def render_free_grid(
    output_path: Path,
    query: str,
    source_label: str,
    source_image_path: Path,
    top_indices: list[int],
    predicted_status: dict[int, dict[str, Any]],
    filenames: list[str],
    images_dir: Path,
    method_name: str,
    top_pool: int,
    used_probe_filter: bool,
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

    canvas_w = cols * tile_size + (cols + 1) * gap
    canvas_h = (
        title_h
        + section_h
        + predicted_rows * (tile_size + label_h)
        + (predicted_rows + 3) * gap
    )
    canvas = Image.new("RGB", (canvas_w, canvas_h), (250, 250, 248))
    draw = ImageDraw.Draw(canvas)

    title = (
        f"free query | {query}\n"
        f"method={method_name} | top_pool={top_pool} | "
        f"probe_filter={'on' if used_probe_filter else 'off'} | "
        "blue=input | green=query OK | yellow=open attrs unchecked | red=query fail"
    )
    draw.multiline_text((gap, 18), title, fill=(15, 15, 15), spacing=5)

    y0 = title_h + gap
    draw.text((gap, y0), "Predicted by the final system", fill=(15, 15, 15))
    y0 += section_h

    source_tile = make_tile(
        source_image_path,
        f"SOURCE\n{source_label}",
        (40, 105, 220),
        tile_size,
        label_h,
    )
    canvas.paste(source_tile, (gap, y0))

    for pos, idx in enumerate(top_indices, start=1):
        status = predicted_status[idx]
        failed = ", ".join(status["failed_query_conditions"][:2])
        if len(status["failed_query_conditions"]) > 2:
            failed += ", ..."
        if status["failed_query_conditions"]:
            status_line = f"query FAIL: {failed}"
            color = (205, 60, 55)
        elif status.get("open_query_conditions"):
            status_line = "known OK, open unchecked"
            color = (235, 180, 45)
        else:
            status_line = "query OK"
            color = (92, 196, 105)
        label = f"rank {pos} {status_line}\nidx={idx}\n{filenames[idx]}"
        r = pos // cols
        c = pos % cols
        x = gap + c * (tile_size + gap)
        y = y0 + r * (tile_size + label_h + gap)
        tile = make_tile(images_dir / filenames[idx], label, color, tile_size, label_h)
        canvas.paste(tile, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def load_probe_filter_state(args: argparse.Namespace, device, gallery_cache: dict[str, Any]):
    """Load the calibrated probe state used by the current final reranker."""
    import torch
    import torch.nn.functional as F

    setup_model_imports()
    import probe_reranker_v1 as probe_v1  # noqa: WPS433

    thresholds_cache = load_embedding_cache(args.probe_results / "probe" / "calibrated_thresholds.pt")
    thresholds = thresholds_cache["thresholds"][args.threshold_objective].float().to(device)

    probs_path = args.probe_results / "probe" / "test_probe_probs.pt"
    if not probs_path.exists():
        raise FileNotFoundError(
            f"Missing precomputed test probe probabilities: {probs_path}"
        )
    probs_cache = load_embedding_cache(probs_path)
    gallery_probs = probs_cache["probs"].float().to(device)

    probe = None
    if args.probe_checkpoint.exists():
        probe, _, _ = probe_v1.load_probe(args.probe_checkpoint, device)

    # Keep gallery embeddings on the same device because source similarity/Hamming
    # utilities expect local tensors in some paths.
    gallery = F.normalize(gallery_cache["embeddings"].float(), dim=-1).to(device)
    return {
        "thresholds": thresholds,
        "gallery_probs": gallery_probs,
        "probe": probe,
        "gallery": gallery,
    }


def apply_probe_filter(
    top_indices,
    top_scores,
    source_embedding,
    source_index: int | None,
    conditions: list[tuple[int, str]],
    attribute_to_index: dict[str, int],
    probe_state: dict[str, Any],
    top_k: int,
):
    import torch

    setup_model_imports()
    import probe_reranker_v1 as probe_v1  # noqa: WPS433
    import probe_reranker_v2_calibrated as probe_v2  # noqa: WPS433

    device = top_scores.device
    attr_indices, signs = probe_v1.query_tensors(conditions, attribute_to_index, device)
    attr_indices = attr_indices.squeeze(0)
    signs = signs.squeeze(0)

    if source_index is not None:
        source_probs = probe_state["gallery_probs"][source_index : source_index + 1]
    else:
        probe = probe_state["probe"]
        if probe is None:
            raise FileNotFoundError(
                f"Missing probe checkpoint for external image: {DEFAULT_PROBE_CHECKPOINT}"
            )
        with torch.inference_mode():
            source_probs = torch.sigmoid(probe(source_embedding.to(device))).float()

    candidate_probs = probe_state["gallery_probs"][top_indices.unsqueeze(0)]
    query_ok = probe_v2.calibrated_query_ok(
        candidate_probs,
        attr_indices,
        signs,
        probe_state["thresholds"],
    )[0]
    hard_ham, _, _ = probe_v2.calibrated_hamming(
        source_probs,
        candidate_probs,
        attr_indices,
        probe_state["thresholds"],
    )
    keep = query_ok & (hard_ham[0] <= 2.0)
    score = top_scores.clone().masked_fill(~keep, -torch.inf)

    selected: list[int] = []
    if bool(torch.isfinite(score).any()):
        local = score.topk(min(top_k, int(torch.isfinite(score).sum().item()))).indices
        selected = top_indices[local].tolist()

    selected_set = set(selected)
    for idx in top_indices.tolist():
        if idx not in selected_set:
            selected.append(idx)
            selected_set.add(idx)
        if len(selected) >= top_k:
            break
    return selected[:top_k], {
        "kept_in_pool": int(keep.sum().item()),
        "avg_pred_hamming_kept": float(hard_ham[0, keep].float().mean().item()) if bool(keep.any()) else None,
    }


def run_free_query(args: argparse.Namespace) -> int:
    import torch
    import torch.nn.functional as F

    setup_model_imports()
    import evaluate_sum_model_blends as blends  # noqa: WPS433
    from learned_gate_core import load_model_checkpoint, load_prompt_embedding_cache  # noqa: WPS433

    try:
        from PIL import Image  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Missing Pillow. Install it with: .venv/bin/python -m pip install pillow"
        ) from exc

    device = choose_device(args.device)
    gallery_cache = load_embedding_cache(args.embedding_cache)
    filenames = list(gallery_cache["filenames"])
    gallery = F.normalize(gallery_cache["embeddings"].float(), dim=-1).to(device)
    _, attribute_to_index, gallery_attrs = load_gallery_attributes(filenames)
    free_conditions = parse_free_query(args.free_query, attribute_to_index)
    known_conditions = known_free_conditions(free_conditions)
    open_conditions = [
        format_free_condition(condition)
        for condition in free_conditions
        if not bool(condition["known"])
    ]
    canonical_query = ", ".join(format_free_condition(condition) for condition in free_conditions)

    source_info = resolve_free_source(args, gallery_cache, filenames)
    source = F.normalize(source_info["embedding"].float(), dim=-1).to(device)
    source_index = source_info["source_index"]

    model, checkpoint_data = load_model_checkpoint(args.checkpoint, device)
    model.eval()
    config = checkpoint_data["config"]
    prompt_path = resolve_prompt_cache(args.prompt_cache)
    prompt_cache = load_prompt_embedding_cache(prompt_path)

    with torch.inference_mode():
        condition_vectors, condition_mask = build_free_condition_embeddings(
            free_conditions,
            prompt_cache,
            attribute_to_index,
            device,
        )
        q_model, _, _ = model(source, condition_vectors, condition_mask)
        edit = condition_vectors.squeeze(0).sum(dim=0, keepdim=True)
        q_sum = F.normalize(source + edit, dim=-1)
        q_final = blends.vector_delta(q_model, q_sum, source, float(args.beta))

        scores = (q_final @ gallery.T).squeeze(0)
        if source_index is not None:
            scores[source_index] = -torch.inf
        pool_size = min(max(args.top_k, args.top_pool), len(scores))
        top = scores.topk(pool_size)
        top_indices = top.indices
        top_scores = top.values

        probe_debug: dict[str, Any] = {}
        used_probe_filter = (not args.no_probe_filter) and bool(known_conditions)
        if used_probe_filter:
            try:
                probe_state = load_probe_filter_state(args, device, gallery_cache)
                selected, probe_debug = apply_probe_filter(
                    top_indices,
                    top_scores,
                    source,
                    source_index,
                    known_conditions,
                    attribute_to_index,
                    probe_state,
                    args.top_k,
                )
            except Exception as exc:
                print(f"Warning: probe filter unavailable ({exc}); falling back to q_final top-k.")
                selected = top_indices[: args.top_k].tolist()
                used_probe_filter = False
                probe_debug = {"probe_filter_error": str(exc)}
        else:
            selected = top_indices[: args.top_k].tolist()
            if not known_conditions and not args.no_probe_filter:
                probe_debug = {
                    "probe_filter_skipped": "all query conditions are open CLIP attributes, not CelebA probe attributes"
                }

    predicted_status = {}
    for idx in selected:
        failed = failed_query_conditions(idx, known_conditions, attribute_to_index, gallery_attrs)
        predicted_status[idx] = {
            "query_ok": not failed and not open_conditions,
            "failed_query_conditions": failed,
            "open_query_conditions": open_conditions,
        }

    source_token = (
        f"idx{source_index}"
        if source_index is not None
        else re.sub(r"[^a-zA-Z0-9]+", "_", source_info["source_image_path"].stem).strip("_")
    )
    query_token = re.sub(r"[^a-zA-Z0-9+-]+", "_", canonical_query).strip("_")
    output_png = args.free_output_dir / f"free_{source_token}_{query_token}_top{len(selected)}.png"
    method_name = f"q_hybrid_beta_{args.beta:g}" + ("_probe_filter" if used_probe_filter else "")
    render_free_grid(
        output_png,
        canonical_query,
        source_info["source_label"],
        source_info["source_image_path"],
        selected,
        predicted_status,
        filenames,
        args.images_dir,
        method_name,
        args.top_pool,
        used_probe_filter,
    )

    metadata = {
        "mode": "free_query",
        "query_raw": args.free_query,
        "query_canonical": canonical_query,
        "conditions": free_conditions,
        "known_conditions": known_conditions,
        "open_conditions": open_conditions,
        "source_index": source_index,
        "source_label": source_info["source_label"],
        "source_image_path": str(source_info["source_image_path"]),
        "top_indices": selected,
        "top_filenames": [filenames[i] for i in selected],
        "top_status": [
            {
                "index": idx,
                "filename": filenames[idx],
                **predicted_status[idx],
            }
            for idx in selected
        ],
        "checkpoint": str(args.checkpoint),
        "prompt_cache": str(prompt_path),
        "beta": args.beta,
        "top_pool": args.top_pool,
        "used_probe_filter": used_probe_filter,
        "probe_debug": probe_debug,
        "output_png": str(output_png),
    }
    output_json = output_png.with_suffix(".json")
    output_json.write_text(json.dumps(metadata, indent=2))

    print(f"Saved: {output_png}")
    print(f"Saved: {output_json}")
    ok_count = sum(1 for idx in selected if predicted_status[idx]["query_ok"])
    if open_conditions:
        ok_phrase = (
            f"{len(selected) - sum(1 for idx in selected if predicted_status[idx]['failed_query_conditions'])}"
            f"/{len(selected)} satisfy known CelebA parts; open conditions are visual-only"
        )
    else:
        ok_phrase = f"{ok_count}/{len(selected)} query-satisfied"
    print(
        f"Free query ({canonical_query}) source={source_info['source_label'].replace(chr(10), ' ')}: "
        f"{ok_phrase} in top-{len(selected)}"
    )
    if not args.no_open:
        subprocess.run(["open", str(output_png)], check=False)
    return 0


def main() -> int:
    args = parse_args()
    if args.free_query:
        return run_free_query(args)

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
