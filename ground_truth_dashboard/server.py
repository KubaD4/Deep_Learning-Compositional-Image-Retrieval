#!/usr/bin/env python3
"""Local dashboard for exploring the CelebA ground-truth JSON."""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


IMAGE_NAME_RE = re.compile(r"^\d{6}\.jpg$")


def parse_query(query: str) -> list[dict[str, object]]:
    modifiers = []
    for raw_part in query.split(","):
        part = raw_part.strip()
        if not part or part[0] not in "+-":
            raise ValueError(f"Unsupported query modifier: {raw_part!r}")
        modifiers.append(
            {
                "operator": part[0],
                "attribute": part[1:].strip(),
                "expected": 1 if part[0] == "+" else -1,
            }
        )
    return modifiers


@dataclass(frozen=True)
class DashboardPaths:
    evaluation_json: Path
    celeba_root: Path

    @property
    def image_root(self) -> Path:
        return self.celeba_root / "img_align_celeba"

    @property
    def partition_file(self) -> Path:
        return self.celeba_root / "list_eval_partition.txt"

    @property
    def attribute_file(self) -> Path:
        return self.celeba_root / "list_attr_celeba.txt"

    @property
    def identity_file(self) -> Path:
        return self.celeba_root / "identity_CelebA.txt"


class CelebAGroundTruth:
    def __init__(self, paths: DashboardPaths) -> None:
        self.paths = paths
        self.queries = json.loads(paths.evaluation_json.read_text(encoding="utf-8"))
        self.test_filenames = self._load_test_filenames()
        self.attribute_names, self.attributes = self._load_attributes()
        self.identities = self._load_identities()
        self._validate()

    def _load_test_filenames(self) -> list[str]:
        filenames = []
        with self.paths.partition_file.open(encoding="utf-8") as handle:
            for line in handle:
                filename, partition = line.split()
                if partition == "2":
                    filenames.append(filename)
        return filenames

    def _load_attributes(self) -> tuple[list[str], dict[str, list[int]]]:
        wanted = set(self.test_filenames)
        values: dict[str, list[int]] = {}
        with self.paths.attribute_file.open(encoding="utf-8") as handle:
            handle.readline()
            names = handle.readline().split()
            for line in handle:
                parts = line.split()
                if parts[0] in wanted:
                    values[parts[0]] = [int(value) for value in parts[1:]]
        return names, values

    def _load_identities(self) -> dict[str, int]:
        wanted = set(self.test_filenames)
        identities: dict[str, int] = {}
        if not self.paths.identity_file.exists():
            return identities
        with self.paths.identity_file.open(encoding="utf-8") as handle:
            for line in handle:
                filename, identity = line.split()
                if filename in wanted:
                    identities[filename] = int(identity)
        return identities

    def _validate(self) -> None:
        if len(self.test_filenames) != 19962:
            raise ValueError(
                f"Expected 19,962 CelebA test images, found {len(self.test_filenames):,}."
            )
        if len(self.attribute_names) != 40:
            raise ValueError(f"Expected 40 attributes, found {len(self.attribute_names)}.")
        missing = set(self.test_filenames) - self.attributes.keys()
        if missing:
            raise ValueError(f"Missing attributes for {len(missing)} test images.")

    def overview(self) -> dict[str, object]:
        return {
            "query_count": len(self.queries),
            "test_size": len(self.test_filenames),
            "attribute_names": self.attribute_names,
            "queries": [
                {
                    "index": index,
                    "query": item["query"],
                    "source_count": len(item["ground_truth"]),
                    "modifiers": parse_query(item["query"]),
                }
                for index, item in enumerate(self.queries)
            ],
        }

    def entry(
        self,
        query_index: int,
        source_position: int = 0,
        source_index: int | None = None,
    ) -> dict[str, object]:
        if not 0 <= query_index < len(self.queries):
            raise IndexError("Query index outside the available range.")

        query_item = self.queries[query_index]
        ground_truth = query_item["ground_truth"]
        source_keys = list(ground_truth)

        if source_index is not None:
            source_key = str(source_index)
            if source_key not in ground_truth:
                raise ValueError(
                    f"Dataset index {source_index} is not a source for query "
                    f"{query_item['query']}."
                )
            source_position = source_keys.index(source_key)
        else:
            source_position %= len(source_keys)
            source_key = source_keys[source_position]
            source_index = int(source_key)

        modifiers = parse_query(query_item["query"])
        target_indices = ground_truth[source_key]
        source = self.image_record(source_index, modifiers)
        targets = [
            self.image_record(index, modifiers, source=source, rank=rank)
            for rank, index in enumerate(target_indices, start=1)
        ]
        same_identity_ranks = [
            target["rank"]
            for target in targets
            if target["identity_id"] == source["identity_id"]
        ]

        return {
            "query_index": query_index,
            "query": query_item["query"],
            "modifiers": modifiers,
            "source_position": source_position,
            "source_count": len(source_keys),
            "source": source,
            "targets": targets,
            "available_target_count": len(ground_truth[source_key]),
            "same_identity_target_count": len(same_identity_ranks),
            "same_identity_target_ranks": same_identity_ranks,
            "note": "Targets are valid ground truth entries in JSON order, not a model ranking.",
        }

    def image_record(
        self,
        dataset_index: int,
        modifiers: list[dict[str, object]],
        source: dict[str, object] | None = None,
        rank: int | None = None,
    ) -> dict[str, object]:
        if not 0 <= dataset_index < len(self.test_filenames):
            raise IndexError(f"Dataset index {dataset_index} is outside the test split.")

        filename = self.test_filenames[dataset_index]
        values = self.attributes[filename]
        by_name = dict(zip(self.attribute_names, values))
        active = [name for name, value in by_name.items() if value == 1]
        query_checks = []
        query_attributes = {str(item["attribute"]) for item in modifiers}

        for modifier in modifiers:
            attribute = str(modifier["attribute"])
            actual = by_name[attribute]
            expected = int(modifier["expected"])
            query_checks.append(
                {
                    **modifier,
                    "actual": actual,
                    "satisfied": actual == expected,
                }
            )

        hamming_distance = None
        differing_attributes: list[str] = []
        if source is not None:
            source_values = source["attribute_values"]
            differing_attributes = [
                name
                for name in self.attribute_names
                if name not in query_attributes and source_values[name] != by_name[name]
            ]
            hamming_distance = len(differing_attributes)

        return {
            "dataset_index": dataset_index,
            "filename": filename,
            "image_url": f"/images/{filename}",
            "identity_id": self.identities.get(filename),
            "rank": rank,
            "active_attributes": active,
            "attribute_values": by_name,
            "query_checks": query_checks,
            "hamming_distance": hamming_distance,
            "differing_attributes": differing_attributes,
        }


class DashboardHandler(BaseHTTPRequestHandler):
    repository: CelebAGroundTruth
    static_root: Path

    def do_GET(self) -> None:  # noqa: N802 - inherited API name
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/overview":
                self._send_json(self.repository.overview())
                return
            if parsed.path == "/api/entry":
                self._handle_entry(parsed.query)
                return
            if parsed.path.startswith("/images/"):
                self._serve_image(unquote(parsed.path.removeprefix("/images/")))
                return
            self._serve_static(parsed.path)
        except (IndexError, KeyError, ValueError) as error:
            self._send_json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)
        except FileNotFoundError:
            self.send_error(HTTPStatus.NOT_FOUND)

    def _handle_entry(self, query_string: str) -> None:
        params = parse_qs(query_string)
        query_index = int(params.get("query", ["0"])[0])
        position = int(params.get("position", ["0"])[0])
        source_raw = params.get("source", [None])[0]
        source_index = int(source_raw) if source_raw is not None else None
        payload = self.repository.entry(query_index, position, source_index)
        self._send_json(payload)

    def _serve_image(self, filename: str) -> None:
        if not IMAGE_NAME_RE.fullmatch(filename):
            self.send_error(HTTPStatus.BAD_REQUEST)
            return
        self._send_file(self.repository.paths.image_root / filename, "image/jpeg")

    def _serve_static(self, request_path: str) -> None:
        relative = "index.html" if request_path in {"", "/"} else request_path.lstrip("/")
        candidate = (self.static_root / relative).resolve()
        if self.static_root.resolve() not in candidate.parents and candidate != self.static_root.resolve():
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        self._send_file(candidate, content_type)

    def _send_file(self, path: Path, content_type: str) -> None:
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format_string: str, *args: object) -> None:
        print(f"[dashboard] {format_string % args}")


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8876, type=int)
    parser.add_argument(
        "--json",
        type=Path,
        default=project_root / "celeba_evaluation.json",
        help="Path to celeba_evaluation.json",
    )
    parser.add_argument(
        "--celeba-root",
        type=Path,
        default=project_root / "celeba",
        help="Directory containing img_align_celeba and CelebA metadata files",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = DashboardPaths(args.json.resolve(), args.celeba_root.resolve())
    repository = CelebAGroundTruth(paths)
    DashboardHandler.repository = repository
    DashboardHandler.static_root = Path(__file__).resolve().parent / "static"
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Ground-truth dashboard: http://{args.host}:{args.port}")
    print(f"JSON: {paths.evaluation_json}")
    print(f"CelebA: {paths.celeba_root}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
