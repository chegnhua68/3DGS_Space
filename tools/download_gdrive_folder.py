#!/usr/bin/env python3
"""Download a public Google Drive folder concurrently and resumably."""

import argparse
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

import gdown
import requests


def _destination(root, relative_path):
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("unsafe Google Drive path: {!r}".format(relative_path))
    destination = (root / Path(*relative.parts)).resolve()
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            "Google Drive path escapes output directory: {!r}".format(relative_path)
        ) from exc
    return destination


def download_folder(folder_id, output_dir, workers=8):
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    records = gdown.download_folder(
        id=folder_id,
        quiet=True,
        skip_download=True,
        use_cookies=False,
    )
    if not records:
        raise RuntimeError("Google Drive folder is empty or unavailable")

    jobs = []
    destinations = set()
    for record in records:
        destination = _destination(root, record.path)
        if destination in destinations:
            raise ValueError("duplicate output path: {}".format(destination))
        destinations.add(destination)
        if destination.is_file() and destination.stat().st_size > 0:
            continue
        jobs.append((record.id, destination))

    progress_lock = threading.Lock()
    completed = len(records) - len(jobs)

    def fetch(file_id, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".part")
        response = requests.get(
            "https://drive.usercontent.google.com/download",
            params={"id": file_id, "export": "download", "confirm": "t"},
            stream=True,
            timeout=(15, 120),
        )
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").lower()
        if "text/html" in content_type:
            raise RuntimeError("Drive returned an HTML response for {}".format(destination.name))
        expected_size = response.headers.get("Content-Length")
        written = 0
        with temporary.open("wb") as stream:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    stream.write(chunk)
                    written += len(chunk)
        response.close()
        if written == 0:
            raise RuntimeError("download is empty: {}".format(destination.name))
        if expected_size is not None and written != int(expected_size):
            raise RuntimeError(
                "size mismatch for {}: expected {}, got {}".format(
                    destination.name, expected_size, written
                )
            )
        os.replace(str(temporary), str(destination))
        return destination

    errors = []
    if jobs:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_destination = {
                executor.submit(fetch, file_id, destination): destination
                for file_id, destination in jobs
            }
            for future in as_completed(future_to_destination):
                destination = future_to_destination[future]
                try:
                    future.result()
                    with progress_lock:
                        completed += 1
                        if completed % 10 == 0 or completed == len(records):
                            print(
                                "downloaded {}/{}".format(completed, len(records)),
                                flush=True,
                            )
                except Exception as exc:  # Continue to report all failed files.
                    errors.append((destination, exc))

    if errors:
        details = ", ".join(
            "{}: {}".format(path.name, error) for path, error in errors[:5]
        )
        raise RuntimeError("{} download(s) failed: {}".format(len(errors), details))

    missing = [path for path in destinations if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError("download verification found {} missing file(s)".format(len(missing)))
    total_bytes = sum(path.stat().st_size for path in destinations)
    print("files={}".format(len(destinations)))
    print("bytes={}".format(total_bytes))
    print("output={}".format(root))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 32:
        parser.error("--workers must be between 1 and 32")
    download_folder(args.folder_id, args.output_dir, args.workers)


if __name__ == "__main__":
    main()
