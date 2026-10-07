from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImageProbe:
    mime_type: str
    width: int | None
    height: int | None
    health_status: str


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
JPEG_SOF = {
    0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
    0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
}


def safe_filename(value: str) -> str:
    name = Path(value).name.strip().replace("\x00", "")
    if not name or name in {".", ".."}:
        raise ValueError("A valid image filename is required.")
    return name


def probe_image(data: bytes, filename: str = "") -> ImageProbe:
    if data.startswith(PNG_SIGNATURE):
        if len(data) < 24 or data[12:16] != b"IHDR":
            return ImageProbe("image/png", None, None, "corrupt")
        width, height = struct.unpack(">II", data[16:24])
        if width <= 0 or height <= 0:
            return ImageProbe("image/png", None, None, "corrupt")
        return ImageProbe("image/png", width, height, "ok")
    if data.startswith(b"\xff\xd8"):
        index = 2
        try:
            while index < len(data):
                while index < len(data) and data[index] != 0xFF:
                    index += 1
                while index < len(data) and data[index] == 0xFF:
                    index += 1
                if index >= len(data):
                    break
                marker = data[index]
                index += 1
                if marker in {0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
                    continue
                if index + 2 > len(data):
                    break
                length = struct.unpack(">H", data[index:index + 2])[0]
                if length < 2 or index + length > len(data):
                    break
                if marker in JPEG_SOF and length >= 7:
                    height, width = struct.unpack(">HH", data[index + 3:index + 7])
                    if width > 0 and height > 0:
                        return ImageProbe("image/jpeg", width, height, "ok")
                    break
                index += length
        except (IndexError, struct.error):
            pass
        return ImageProbe("image/jpeg", None, None, "corrupt")
    suffix = Path(filename).suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg" if suffix in {".jpg", ".jpeg"} else "application/octet-stream"
    return ImageProbe(mime, None, None, "unsupported")


def _probe_png(data: bytes) -> ImageProbe:
    if len(data) < 24 or not data.startswith(PNG_SIGNATURE) or data[12:16] != b"IHDR":
        return ImageProbe("image/png", None, None, "corrupt")
    width, height = struct.unpack(">II", data[16:24])
    if width <= 0 or height <= 0:
        return ImageProbe("image/png", None, None, "corrupt")
    return ImageProbe("image/png", width, height, "ok")


def _probe_jpeg(data: bytes) -> ImageProbe:
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        return ImageProbe("image/jpeg", None, None, "corrupt")
    offset = 2
    while offset + 3 < len(data):
        if data[offset] != 0xFF:
            offset += 1
            continue
        while offset < len(data) and data[offset] == 0xFF:
            offset += 1
        if offset >= len(data):
            break
        marker = data[offset]
        offset += 1
        if marker in {0xD8, 0xD9}:
            continue
        if marker == 0xDA:
            break
        if offset + 2 > len(data):
            break
        segment_length = int.from_bytes(data[offset:offset + 2], "big")
        if segment_length < 2 or offset + segment_length > len(data):
            break
        if marker in JPEG_SOF and segment_length >= 7:
            height = int.from_bytes(data[offset + 3:offset + 5], "big")
            width = int.from_bytes(data[offset + 5:offset + 7], "big")
            if width > 0 and height > 0:
                return ImageProbe("image/jpeg", width, height, "ok")
            break
        offset += segment_length
    return ImageProbe("image/jpeg", None, None, "corrupt")


def probe_image(data: bytes, filename: str) -> ImageProbe:
    suffix = Path(filename).suffix.casefold()
    if data.startswith(PNG_SIGNATURE):
        return _probe_png(data)
    if data.startswith(b"\xff\xd8"):
        return _probe_jpeg(data)
    mime = "application/octet-stream"
    if suffix == ".png":
        mime = "image/png"
    elif suffix in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    return ImageProbe(mime, None, None, "unsupported" if suffix not in {".png", ".jpg", ".jpeg"} else "corrupt")
