from __future__ import annotations

import re
import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile


SAFE_FILENAME = re.compile(r"[^a-zA-Z0-9._-]+")


def sanitize_filename(filename: str | None, fallback: str) -> str:
    if not filename:
        return fallback
    cleaned = SAFE_FILENAME.sub("_", filename)
    return cleaned or fallback


async def persist_upload(
    upload: UploadFile,
    destination_dir: Path,
    fallback_name: str,
    max_size_mb: int,
) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    suffix = _detect_suffix(upload.filename)
    filename = sanitize_filename(upload.filename, fallback_name) + suffix
    destination = destination_dir / filename
    size_limit = max_size_mb * 1024 * 1024
    bytes_written = 0
    with destination.open("wb") as file_obj:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            bytes_written += len(chunk)
            if bytes_written > size_limit:
                raise HTTPException(status_code=413, detail="Uploaded file exceeds size limit")
            file_obj.write(chunk)
    await upload.close()
    return destination


def create_request_upload_dir(base_dir: Path) -> Path:
    request_id = uuid.uuid4().hex[:12]
    upload_dir = base_dir / request_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


def _detect_suffix(filename: str | None) -> str:
    if not filename:
        return ".nii.gz"
    lower = filename.lower()
    if lower.endswith(".nii.gz"):
        return ""
    if lower.endswith(".nii"):
        return ""
    raise HTTPException(status_code=400, detail=f"Unsupported file extension for '{filename}'")

