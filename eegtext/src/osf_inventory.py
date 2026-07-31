"""Inventory public OSF files without materializing the dataset."""

from dataclasses import asdict, dataclass
import csv
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen


OSF_API_ROOT = "https://api.osf.io/v2"


@dataclass(frozen=True)
class RemoteFile:
    node: str
    file_id: str
    name: str
    path: str
    size_bytes: object
    download_url: str


def _fetch_json(url):
    request = Request(url, headers={"User-Agent": "EEGText corpus inventory"})
    with urlopen(request, timeout=60) as response:
        return json.load(response)


def _next_url(payload):
    value = payload.get("links", {}).get("next")
    if isinstance(value, dict):
        return value.get("href")
    return value


def inventory_node(node, fetch_json=_fetch_json):
    """Recursively list files in one public OSF storage provider."""

    queue = [f"{OSF_API_ROOT}/nodes/{node}/files/osfstorage/"]
    visited = set()
    files = []
    while queue:
        url = queue.pop(0)
        while url:
            if url in visited:
                break
            visited.add(url)
            payload = fetch_json(url)
            for item in payload.get("data", []):
                attributes = item.get("attributes", {})
                kind = attributes.get("kind")
                if kind == "folder":
                    related = (
                        item.get("relationships", {})
                        .get("files", {})
                        .get("links", {})
                        .get("related", {})
                    )
                    queue.append(related.get("href") if isinstance(related, dict) else related)
                    continue
                if kind != "file":
                    continue
                links = item.get("links", {})
                files.append(
                    RemoteFile(
                        node=str(node),
                        file_id=str(item.get("id", "")),
                        name=str(attributes.get("name", "")),
                        path=str(attributes.get("materialized_path", attributes.get("path", ""))),
                        size_bytes=attributes.get("size"),
                        download_url=str(links.get("download", "")),
                    )
                )
            url = _next_url(payload)
    return sorted(files, key=lambda item: (item.path.casefold(), item.file_id))


def _atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def write_inventory(files, node, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / "files.csv.tmp"
    fields = list(RemoteFile.__dataclass_fields__)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(item) for item in files)
    os.replace(temporary, output_dir / "files.csv")
    payload = {
        "node": str(node),
        "file_count": len(files),
        "known_size_file_count": sum(item.size_bytes is not None for item in files),
        "total_known_size_bytes": int(
            sum(int(item.size_bytes) for item in files if item.size_bytes is not None)
        ),
        "files": [asdict(item) for item in files],
    }
    _atomic_text(output_dir / "inventory.json", json.dumps(payload, indent=2) + "\n")
    return {key: value for key, value in payload.items() if key != "files"}


def inventory_osf(node, output_dir):
    files = inventory_node(node)
    return write_inventory(files, node, output_dir)
