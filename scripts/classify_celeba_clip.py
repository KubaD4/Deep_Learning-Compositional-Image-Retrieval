#!/usr/bin/env python3
"""Inspect an attribute change as a direction in OpenAI CLIP space."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = PROJECT_ROOT / "celeba" / "img_align_celeba"
ANNOTATION_DIR = PROJECT_ROOT / "data" / "celeba" / "annotations"
ATTRIBUTE_FILE = ANNOTATION_DIR / "list_attr_celeba.txt"
IDENTITY_FILE = ANNOTATION_DIR / "identity_CelebA.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Choose two images of one CelebA person, compute their CLIP image "
            "difference, and classify that direction using text directions."
        )
    )
    parser.add_argument("person_id", type=int, help="CelebA person identity ID.")
    parser.add_argument(
        "--attribute",
        default="No_Beard",
        help="Attribute that must change from absent to present (default: No_Beard).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of signed attribute directions to display (default: 10).",
    )
    return parser.parse_args()


def load_metadata() -> tuple[list[str], dict[str, list[int]], dict[int, list[str]]]:
    with ATTRIBUTE_FILE.open(encoding="utf-8") as handle:
        handle.readline()
        attribute_names = handle.readline().split()
        attributes = {
            parts[0]: [int(value) for value in parts[1:]]
            for line in handle
            if (parts := line.split())
        }

    images_by_person: dict[int, list[str]] = defaultdict(list)
    with IDENTITY_FILE.open(encoding="utf-8") as handle:
        for line in handle:
            filename, person_id = line.split()
            images_by_person[int(person_id)].append(filename)

    return attribute_names, attributes, dict(images_by_person)


def normalize_attribute_name(requested: str, attribute_names: list[str]) -> str:
    normalized = requested.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "without_beard": "no_beard",
        "clean_shaven": "no_beard",
    }
    normalized = aliases.get(normalized, normalized)

    by_normalized = {name.lower(): name for name in attribute_names}
    if normalized not in by_normalized:
        choices = ", ".join(attribute_names)
        raise SystemExit(f"Unknown attribute '{requested}'. Available attributes:\n{choices}")
    return by_normalized[normalized]


def choose_pair(
    filenames: list[str], attributes: dict[str, list[int]], attribute_index: int
) -> tuple[str, str, list[int]]:
    source_candidates = [
        filename for filename in filenames if attributes[filename][attribute_index] == -1
    ]
    target_candidates = [
        filename for filename in filenames if attributes[filename][attribute_index] == 1
    ]

    if not source_candidates or not target_candidates:
        raise SystemExit(
            "This person has no image pair where the requested attribute changes "
            "from absent (-1) to present (+1)."
        )

    best_pair = None
    best_differences = None
    for source in source_candidates:
        for target in target_candidates:
            differences = [
                index
                for index, (source_value, target_value) in enumerate(
                    zip(attributes[source], attributes[target])
                )
                if source_value != target_value
            ]
            if best_differences is None or len(differences) < len(best_differences):
                best_pair = (source, target)
                best_differences = differences

    assert best_pair is not None and best_differences is not None
    return best_pair[0], best_pair[1], best_differences


def prompt_pair(attribute: str) -> tuple[str, str]:
    special = {
        "Attractive": ("an attractive face", "an unattractive face"),
        "Bald": ("a bald person", "a person with hair"),
        "Blurry": ("a blurry face photo", "a sharp face photo"),
        "Chubby": ("a chubby face", "a slim face"),
        "Male": ("a male face", "a female face"),
        "Mouth_Slightly_Open": ("a face with an open mouth", "a face with a closed mouth"),
        "No_Beard": ("a clean-shaven face without a beard", "a face with a beard"),
        "Smiling": ("a smiling face", "a face that is not smiling"),
        "Young": ("a young face", "an older face"),
    }
    if attribute in special:
        return special[attribute]

    label = attribute.replace("_", " ").lower()
    return f"a face with {label}", f"a face without {label}"


def main() -> None:
    args = parse_args()
    attribute_names, attributes, images_by_person = load_metadata()
    requested_attribute = normalize_attribute_name(args.attribute, attribute_names)

    if args.person_id not in images_by_person:
        raise SystemExit(f"Unknown CelebA person ID: {args.person_id}")

    requested_index = attribute_names.index(requested_attribute)
    source_name, target_name, changed_indices = choose_pair(
        images_by_person[args.person_id], attributes, requested_index
    )
    source_path = IMAGE_DIR / source_name
    target_path = IMAGE_DIR / target_name

    try:
        import clip
        import torch
        from PIL import Image
    except ImportError as exc:
        raise SystemExit(
            "Missing dependencies. Run:\n"
            "  python3 -m pip install torch torchvision\n"
            "  python3 -m pip install git+https://github.com/openai/CLIP.git"
        ) from exc

    model, preprocess = clip.load("ViT-B/32", device="cpu")
    model.eval()

    with Image.open(source_path) as source_image, Image.open(target_path) as target_image:
        image_input = torch.stack(
            [
                preprocess(source_image.convert("RGB")),
                preprocess(target_image.convert("RGB")),
            ]
        )

    prompt_pairs = [prompt_pair(attribute) for attribute in attribute_names]
    flat_prompts = [prompt for pair in prompt_pairs for prompt in pair]
    text_input = clip.tokenize(flat_prompts).to("cpu")

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    image_delta = image_features[1] - image_features[0]
    image_delta /= image_delta.norm()

    positive_text = text_features[0::2]
    negative_text = text_features[1::2]
    text_directions = positive_text - negative_text
    text_directions /= text_directions.norm(dim=1, keepdim=True)

    requested_positive = positive_text[requested_index]
    requested_negative = negative_text[requested_index]
    requested_direction = text_directions[requested_index]

    direction_score = (image_delta @ requested_direction).item()
    positive_prompt_score = (image_delta @ requested_positive).item()
    negative_prompt_score = (image_delta @ requested_negative).item()

    all_direction_scores = image_delta @ text_directions.T
    signed_rankings = []
    for index, score in enumerate(all_direction_scores.tolist()):
        label = f"+{attribute_names[index]}" if score >= 0 else f"-{attribute_names[index]}"
        signed_rankings.append((label, abs(score), score))
    signed_rankings.sort(key=lambda item: item[1], reverse=True)

    print(f"Person ID: {args.person_id}")
    print(f"Requested direction: +{requested_attribute} (absent -> present)")
    print(f"Source image: {source_name}")
    print(f"Target image: {target_name}")
    print(f"Images available for this person: {len(images_by_person[args.person_id])}")

    print("\nAll ground-truth attribute changes in the selected pair:")
    for index in changed_indices:
        sign = "+" if attributes[target_name][index] == 1 else "-"
        marker = "  <-- requested" if index == requested_index else ""
        print(f"  {sign}{attribute_names[index]}{marker}")

    positive_prompt, negative_prompt = prompt_pairs[requested_index]
    print("\nText interpretation used by CLIP:")
    print(f'  positive: "{positive_prompt}"')
    print(f'  negative: "{negative_prompt}"')
    print("  text direction = embedding(positive) - embedding(negative)")
    print("  image direction = embedding(target) - embedding(source)")

    print("\nRaw cosine similarities [-1, 1]:")
    print(f"  image delta <-> requested text direction: {direction_score:+.4f}")
    print(f"  image delta <-> positive prompt:          {positive_prompt_score:+.4f}")
    print(f"  image delta <-> negative prompt:          {negative_prompt_score:+.4f}")

    top_k = max(1, min(args.top_k, len(signed_rankings)))
    print(f"\nTop {top_k} CLIP interpretations of the image difference:")
    for label, _, signed_score in signed_rankings[:top_k]:
        requested_marker = "  <-- requested" if label == f"+{requested_attribute}" else ""
        print(f"  {label:<28} cosine={signed_score:+.4f}{requested_marker}")


if __name__ == "__main__":
    main()
