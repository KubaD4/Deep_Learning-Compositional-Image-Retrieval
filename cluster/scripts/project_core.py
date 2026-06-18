"""Shared project utilities written for this assignment.

The evaluation equations mirror the instructor-provided starter notebook.
No third-party project repository code is used.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.datasets import CelebA


ROOT = Path(os.environ.get("DL_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
DATA_ROOT = ROOT / "data"
CELEBA_DIR = DATA_ROOT / "celeba"
EVALUATION_PATH = DATA_ROOT / "celeba_evaluation.json"
MODEL_ID = "openai/clip-vit-base-patch32"
MODEL_SLUG = "openai_clip_vit_b32"
EMBEDDING_DIR = CELEBA_DIR / "embeddings" / MODEL_SLUG
ARTIFACTS_DIR = ROOT / "artifacts"
RESULTS_DIR = ARTIFACTS_DIR / "results" / "baselines"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"
TOP_KS = (1, 5, 10)

METHOD_FOLDERS = {
    "direct_sum": "01_direct_sum",
    "direct_sequential": "02_direct_sequential",
    "contrastive_sum": "03_contrastive_sum",
    "contrastive_sequential": "04_contrastive_sequential",
    "adaptive_tangent_sequential": "05_adaptive_tangent_sequential",
}


def annotation_path(filename: str) -> Path:
    """Return a CelebA annotation path across the two layouts used locally."""
    candidates = [
        CELEBA_DIR / filename,
        CELEBA_DIR / "annotations" / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Missing CelebA annotation {filename}; checked "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def read_attribute_table() -> tuple[list[str], list[str], torch.Tensor]:
    """Load CelebA attributes as filenames plus a {-1,+1} int8 tensor."""
    path = annotation_path("list_attr_celeba.txt")
    with path.open(encoding="utf-8") as handle:
        _ = int(handle.readline().strip())
        names = [name for name in handle.readline().split() if name]
        filenames = []
        rows = []
        for line in handle:
            parts = line.split()
            if not parts:
                continue
            filenames.append(parts[0])
            rows.append([1 if int(value) == 1 else -1 for value in parts[1:]])
    if len(names) != 40:
        raise RuntimeError(f"Expected 40 attributes, found {len(names)}")
    return names, filenames, torch.tensor(rows, dtype=torch.int8)


def read_identity_map() -> dict[str, int]:
    path = annotation_path("identity_CelebA.txt")
    identities = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            filename, identity = line.split()
            identities[filename] = int(identity)
    return identities


def read_partition_map() -> dict[str, int]:
    path = annotation_path("list_eval_partition.txt")
    partitions = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            filename, partition = line.split()
            partitions[filename] = int(partition)
    return partitions


def choose_device(requested: str = "auto") -> torch.device:
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
    return device


def load_celeba(split: str) -> CelebA:
    if not (CELEBA_DIR / "img_align_celeba").is_dir():
        raise FileNotFoundError(
            f"Missing extracted images at {CELEBA_DIR / 'img_align_celeba'}. "
            "Run scripts/setup_data.py first."
        )
    return CelebA(
        root=DATA_ROOT,
        split=split,
        target_type="attr",
        download=False,
    )


def attribute_names(dataset: CelebA) -> list[str]:
    # Some torchvision versions preserve an empty trailing header token.
    names = [name for name in dataset.attr_names if name]
    if len(names) != 40:
        raise RuntimeError(f"Expected 40 CelebA attributes, found {len(names)}")
    return names


def prompt_pair(attribute: str) -> tuple[str, str]:
    special = {
        "Attractive": ("an attractive face", "an unattractive face"),
        "Bald": ("a bald person", "a person with hair"),
        "Blurry": ("a blurry face photo", "a sharp face photo"),
        "Chubby": ("a chubby face", "a slim face"),
        "Male": ("a male face", "a female face"),
        "Mouth_Slightly_Open": (
            "a face with an open mouth",
            "a face with a closed mouth",
        ),
        "No_Beard": (
            "a clean-shaven face without a beard",
            "a face with a beard",
        ),
        "Smiling": ("a smiling face", "a face that is not smiling"),
        "Young": ("a young face", "an older face"),
    }
    if attribute in special:
        return special[attribute]
    readable = attribute.replace("_", " ").lower()
    return f"a face with {readable}", f"a face without {readable}"


def parse_query(query: str) -> list[tuple[int, str]]:
    conditions = []
    for raw_condition in query.split(","):
        token = raw_condition.strip()
        if not token or token[0] not in "+-":
            raise ValueError(f"Invalid signed condition: {raw_condition!r}")
        conditions.append((1 if token[0] == "+" else -1, token[1:].strip()))
    return conditions


def unwrap_features(output):
    """Support Transformers versions returning either a tensor or model output."""
    return output.pooler_output if hasattr(output, "pooler_output") else output


def load_torch(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def atomic_torch_save(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_json_dump(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=path.name, suffix=".tmp", delete=False
    ) as handle:
        json.dump(value, handle, indent=2)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def load_evaluation() -> list[dict]:
    with EVALUATION_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def compose_queries(
    source_embeddings: torch.Tensor,
    conditions: list[tuple[int, str]],
    method: str,
    attribute_to_index: dict[str, int],
    positive_embeddings: torch.Tensor,
    directions: torch.Tensor,
) -> torch.Tensor:
    if method not in METHOD_FOLDERS:
        raise ValueError(f"Unknown method: {method}")
    query = source_embeddings.float().clone()

    if method == "adaptive_tangent_sequential":
        for sign, attribute in conditions:
            direction = directions[attribute_to_index[attribute]].to(query.device)
            signed_direction = sign * direction

            # A source-aware step: edit strongly when the requested attribute is
            # absent and gently when CLIP already sees it in the source image.
            alignment = (query * signed_direction).sum(dim=-1, keepdim=True)
            strength = (1.0 - alignment).clamp(0.25, 1.75)

            # Move along the tangent plane of the normalized CLIP space. This
            # avoids spending edit magnitude on the radial component of query.
            tangent = signed_direction.unsqueeze(0) - alignment * query
            query = F.normalize(query + strength * tangent, dim=-1)
        return query

    bank = directions if method.startswith("contrastive") else positive_embeddings
    sequential = method.endswith("sequential")

    if sequential:
        for sign, attribute in conditions:
            edit = sign * bank[attribute_to_index[attribute]].to(query.device)
            query = F.normalize(query + edit, dim=-1)
        return query

    edit = torch.zeros_like(query)
    for sign, attribute in conditions:
        edit = edit + sign * bank[attribute_to_index[attribute]].to(query.device)
    return F.normalize(query + edit, dim=-1)


def retrieval_metrics(
    ranked_indices: list[int], valid_targets: set[int], k: int
) -> tuple[int, float]:
    """Instructor starter-notebook definitions of Recall@K and Precision@K."""
    hits = set(ranked_indices[:k]).intersection(valid_targets)
    return int(bool(hits)), len(hits) / k
