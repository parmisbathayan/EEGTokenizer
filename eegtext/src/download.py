"""Select and resumably download the locked ZuCo 1.0 task files."""

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
from urllib.request import Request, urlopen


ZUCO_1_NODE = "q3zws"
ZUCO_1_SUBJECTS = (
    "ZAB",
    "ZDM",
    "ZDN",
    "ZGW",
    "ZJM",
    "ZJN",
    "ZJS",
    "ZKB",
    "ZKH",
    "ZKW",
    "ZMG",
    "ZPH",
)
TASK_PATTERNS = {
    "NR": re.compile(
        r"^/task2 - NR/Matlab files/results(?P<subject>[A-Z]+)_NR\.mat$",
        re.IGNORECASE,
    ),
    "TSR": re.compile(
        r"^/task3 - TSR/Matlab files/results(?P<subject>[A-Z]+)_TSR\.mat$",
        re.IGNORECASE,
    ),
}


@dataclass(frozen=True)
class DownloadItem:
    task: str
    subject: str
    name: str
    remote_path: str
    size_bytes: int
    download_url: str
    destination: str


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_inventory(path):
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("node") != ZUCO_1_NODE:
        raise ValueError(
            f"expected ZuCo 1.0 OSF node {ZUCO_1_NODE}, got {payload.get('node')!r}"
        )
    if not isinstance(payload.get("files"), list):
        raise ValueError("inventory lacks a files list")
    return payload


def select_task_files(inventory, task, output_dir, expected_subjects=ZUCO_1_SUBJECTS):
    """Select exactly the official subject-level MATLAB files for one task."""

    task = task.upper()
    if task not in TASK_PATTERNS:
        raise ValueError(f"unsupported extra ZuCo task: {task}")
    pattern = TASK_PATTERNS[task]
    output_dir = Path(output_dir)
    selected = []
    for remote in inventory["files"]:
        remote_path = str(remote.get("path", ""))
        match = pattern.fullmatch(remote_path)
        if not match:
            continue
        size = remote.get("size_bytes")
        url = str(remote.get("download_url", ""))
        if size is None or int(size) <= 0:
            raise ValueError(f"missing official size for {remote_path}")
        if not url.startswith("https://"):
            raise ValueError(f"invalid download URL for {remote_path}")
        subject = match.group("subject").upper()
        name = str(remote.get("name", ""))
        selected.append(
            DownloadItem(
                task=task,
                subject=subject,
                name=name,
                remote_path=remote_path,
                size_bytes=int(size),
                download_url=url,
                destination=str((output_dir / name).resolve()),
            )
        )
    selected.sort(key=lambda item: item.subject)
    observed = [item.subject for item in selected]
    expected = sorted(str(subject).upper() for subject in expected_subjects)
    if observed != expected:
        raise ValueError(
            f"{task} subject files do not match the locked set; "
            f"expected {expected}, observed {observed}"
        )
    if len({item.name for item in selected}) != len(selected):
        raise ValueError(f"duplicate destination names selected for {task}")
    return selected


def task_plan(inventory_path, task, output_dir):
    inventory = load_inventory(inventory_path)
    items = select_task_files(inventory, task, output_dir)
    return {
        "node": inventory["node"],
        "task": task.upper(),
        "file_count": len(items),
        "total_bytes": sum(item.size_bytes for item in items),
        "files": [asdict(item) for item in items],
    }


def _response_status(response):
    status = getattr(response, "status", None)
    if status is not None:
        return int(status)
    getter = getattr(response, "getcode", None)
    return int(getter()) if getter is not None else 200


def download_one(item, chunk_size=8 * 1024 * 1024, max_attempts=5, opener=urlopen):
    """Download one file, keeping a persistent .part file across interruptions."""

    destination = Path(item.destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    if destination.exists():
        observed = destination.stat().st_size
        if observed != item.size_bytes:
            raise ValueError(
                f"existing file has the wrong size: {destination} "
                f"({observed} != {item.size_bytes})"
            )
        return "existing"
    if partial.exists() and partial.stat().st_size > item.size_bytes:
        raise ValueError(f"partial file is larger than expected: {partial}")

    attempts = 0
    while (partial.stat().st_size if partial.exists() else 0) < item.size_bytes:
        attempts += 1
        if attempts > max_attempts:
            observed = partial.stat().st_size if partial.exists() else 0
            raise RuntimeError(
                f"download remained incomplete after {max_attempts} attempts: "
                f"{destination.name} ({observed}/{item.size_bytes} bytes)"
            )
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "EEGText ZuCo downloader"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = Request(item.download_url, headers=headers)
        try:
            with opener(request, timeout=120) as response:
                status = _response_status(response)
                if offset and status != 206:
                    mode = "wb"
                    offset = 0
                else:
                    mode = "ab" if offset else "wb"
                received = 0
                last_report = offset
                with partial.open(mode) as handle:
                    while True:
                        block = response.read(chunk_size)
                        if not block:
                            break
                        handle.write(block)
                        received += len(block)
                        current = offset + received
                        if current - last_report >= 256 * 1024 * 1024:
                            print(
                                f"    {current / (1024 ** 3):.2f}/"
                                f"{item.size_bytes / (1024 ** 3):.2f} GiB",
                                flush=True,
                            )
                            last_report = current
                if received == 0 and partial.stat().st_size < item.size_bytes:
                    raise OSError(f"empty response while downloading {destination.name}")
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            if attempts == max_attempts:
                raise

    observed = partial.stat().st_size
    if observed != item.size_bytes:
        raise ValueError(
            f"completed partial file has the wrong size: {partial} "
            f"({observed} != {item.size_bytes})"
        )
    os.replace(partial, destination)
    return "downloaded"


def download_task(inventory_path, task, output_dir, status_file=None):
    """Download a locked task and update a small report after every file."""

    plan = task_plan(inventory_path, task, output_dir)
    items = [DownloadItem(**value) for value in plan["files"]]
    status_path = Path(status_file) if status_file else Path(output_dir) / "download_status.json"
    report = {
        "node": plan["node"],
        "task": plan["task"],
        "file_count": plan["file_count"],
        "total_bytes": plan["total_bytes"],
        "completed_files": 0,
        "completed_bytes": 0,
        "files": [],
    }
    _atomic_json(status_path, report)
    for index, item in enumerate(items, start=1):
        print(
            f"[{index}/{len(items)}] {item.name} "
            f"({item.size_bytes / (1024 ** 3):.2f} GiB)",
            flush=True,
        )
        try:
            action = download_one(item)
        except Exception as error:
            report["failed_file"] = item.name
            report["error"] = f"{type(error).__name__}: {error}"
            _atomic_json(status_path, report)
            raise
        report.pop("failed_file", None)
        report.pop("error", None)
        report["files"].append(
            {
                **asdict(item),
                "action": action,
                "validated_size_bytes": Path(item.destination).stat().st_size,
            }
        )
        report["completed_files"] = len(report["files"])
        report["completed_bytes"] = sum(
            value["validated_size_bytes"] for value in report["files"]
        )
        _atomic_json(status_path, report)
    report["complete"] = True
    _atomic_json(status_path, report)
    return {key: value for key, value in report.items() if key != "files"}
