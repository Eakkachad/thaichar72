"""External public Thai character dataset fetching, mapping, and caching."""

from __future__ import annotations

import io
import json
import logging
import os
import re
import tarfile
import zipfile
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from PIL import Image

from thaichar.classes import (
    CLASS_CODES,
    _code_to_index,
    category,
    char_to_code,
    code_to_char,
    code_to_index,
    unicode_name,
)

logger = logging.getLogger(__name__)

# Explicit mapping from consonant/vowel/mark/digit names, transliterations,
# and aliases to their Unicode Thai characters.
NAME_TO_CHAR: dict[str, str] = {
    # 44 Consonants
    "ko kai": "ก",
    "kho khai": "ข",
    "khor khai": "ข",
    "kho khuat": "ฃ",
    "kho khwai": "ค",
    "kho khon": "ฅ",
    "kho rakhang": "ฆ",
    "ngo ngu": "ง",
    "cho chan": "จ",
    "cho ching": "ฉ",
    "cho chang": "ช",
    "so so": "ซ",
    "cho choe": "ฌ",
    "yo ying": "ญ",
    "do chada": "ฎ",
    "to patak": "ฏ",
    "tho than": "ฐ",
    "tho nangmontho": "ฑ",
    "tho nang montho": "ฑ",
    "tho phuthao": "ฒ",
    "tho phu thao": "ฒ",
    "no nen": "ณ",
    "do dek": "ด",
    "to tao": "ต",
    "tho thung": "ถ",
    "tho thahan": "ท",
    "tho thong": "ธ",
    "no nu": "น",
    "bo baimai": "บ",
    "bo bai mai": "บ",
    "po pla": "ป",
    "pho phung": "ผ",
    "fo fa": "ฝ",
    "pho phan": "พ",
    "fo fan": "ฟ",
    "pho samphao": "ภ",
    "pho sam phao": "ภ",
    "mo ma": "ม",
    "yo yak": "ย",
    "ro rua": "ร",
    "ru": "ฤ",
    "lo ling": "ล",
    "lu": "ฦ",
    "wo waen": "ว",
    "so sala": "ศ",
    "so rusi": "ษ",
    "so ru si": "ษ",
    "so sua": "ส",
    "ho hip": "ห",
    "lo chula": "ฬ",
    "o ang": "อ",
    "ho nokhuk": "ฮ",
    "ho nok huk": "ฮ",
    # Marks / Vowels
    "paiyannoi": "ฯ",
    "paiyan noi": "ฯ",
    "sara a": "ะ",
    "mai han": "ั",
    "mai han-akat": "ั",
    "mai han akat": "ั",
    "sara aa": "า",
    "sara am": "ำ",
    "sara i": "ิ",
    "sara ii": "ี",
    "sara ue": "ึ",
    "sara uee": "ื",
    "sara u": "ุ",
    "sara uu": "ู",
    "sara e": "เ",
    "sara ae": "แ",
    "sara o": "โ",
    "sara ai maimuan": "ใ",
    "sara ai mai muan": "ใ",
    "sara ai maimalai": "ไ",
    "sara ai mai malai": "ไ",
    "lakkhangyao": "ๅ",
    "lak khang yao": "ๅ",
    "maiyamok": "ๆ",
    "mai yamok": "ๆ",
    "maitaikhu": "็",
    "mai tai khu": "็",
    "mai ek": "่",
    "mai tho": "้",
    "mai tri": "๊",
    "mai chattawa": "๋",
    "thanthakhat": "์",
    "karan": "์",
    "nikhahit": "ํ",
    "phinthu": "ฺ",
    "baht": "฿",
    "baht sign": "฿",
    # Digits
    "thai digit zero": "๐",
    "thai digit one": "๑",
    "thai digit two": "๒",
    "thai digit three": "๓",
    "thai digit four": "๔",
    "thai digit five": "๕",
    "thai digit six": "๖",
    "thai digit seven": "๗",
    "thai digit eight": "๘",
    "thai digit nine": "๙",
    "0": "๐",
    "1": "๑",
    "2": "๒",
    "3": "๓",
    "4": "๔",
    "5": "๕",
    "6": "๖",
    "7": "๗",
    "8": "๘",
    "9": "๙",
    "zero": "๐",
    "one": "๑",
    "two": "๒",
    "three": "๓",
    "four": "๔",
    "five": "๕",
    "six": "๖",
    "seven": "๗",
    "eight": "๘",
    "nine": "๙",
}

# Normalised lookup dict
_NORM_NAME_TO_CHAR: dict[str, str] = {}


def _normalize_name(name: str) -> str:
    """Normalize a transliterated/Unicode name for lookup."""
    s = name.strip().lower()
    s = re.sub(r"^thai\s+character\s+", "", s)
    s = re.sub(r"^thai\s+digit\s+", "thai digit ", s)
    s = re.sub(r"[-_]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


for _k, _v in NAME_TO_CHAR.items():
    _NORM_NAME_TO_CHAR[_normalize_name(_k)] = _v


def print_transliteration_mapping() -> None:
    """Print the transliteration and name to character mapping table."""
    print("Explicit Transliteration to Character Mapping:")
    print("-" * 50)
    for name, ch in sorted(NAME_TO_CHAR.items()):
        cp = ord(ch)
        tis = cp - 0x0E00 + 0xA0
        mapped = tis in _code_to_index
        status = f"class {_code_to_index[tis]}" if mapped else "unmapped"
        print(f"  {name:25s} -> '{ch}' (U+{cp:04X}, TIS {tis}) [{status}]")


def map_char_to_class(ch: str) -> int | None:
    """Convert a Unicode Thai character or transliteration name to our 72-class index (0..71).

    Returns None if the character is not one of our 72 target classes
    (e.g. ฅ, ฆ, ฌ, ฎ, ฦ, ะ, ำ, ๋, ํ) or cannot be resolved.
    """
    if not ch:
        return None

    # Direct single character lookup
    if len(ch) == 1:
        cp = ord(ch)
        # Check Thai Unicode block: U+0E00 to U+0E7F
        if 0x0E00 <= cp <= 0x0E7F:
            tis_code = cp - 0x0E00 + 0xA0
            return _code_to_index.get(tis_code, None)
        # Handle ASCII digits if passed as single character
        if ch in NAME_TO_CHAR:
            thai_digit = NAME_TO_CHAR[ch]
            tis_code = ord(thai_digit) - 0x0E00 + 0xA0
            return _code_to_index.get(tis_code, None)
        return None

    # Handle transliterations / multi-character strings
    norm = _normalize_name(ch)
    if norm in _NORM_NAME_TO_CHAR:
        target_char = _NORM_NAME_TO_CHAR[norm]
        tis_code = ord(target_char) - 0x0E00 + 0xA0
        return _code_to_index.get(tis_code, None)

    return None


def binarize_for_cache(img: Image.Image | np.ndarray | str | Path) -> np.ndarray:
    """Binarize and tight-crop an image for glyph caching.

    Pipeline:
      1. Load / convert to grayscale uint8.
      2. Otsu thresholding -> {0, 255}.
      3. Ensure ink is dark (0) on white (255) background (check mean; invert if needed).
      4. Crop to ink bounding box with 0 margin.
      5. Reject image if ink fraction < 1% or > 90%.

    Returns
    -------
    np.ndarray
        Cropped uint8 array with values in {0, 255}.

    Raises
    ------
    ValueError
        If no ink pixels found or ink fraction is outside [0.01, 0.90].
    """
    if isinstance(img, (str, Path)):
        pil_img = Image.open(str(img)).convert("L")
        gray = np.asarray(pil_img, dtype=np.uint8)
    elif isinstance(img, Image.Image):
        gray = np.asarray(img.convert("L"), dtype=np.uint8)
    elif isinstance(img, np.ndarray):
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        elif img.ndim == 2:
            gray = img.astype(np.uint8)
        else:
            raise ValueError(f"Unsupported numpy array ndim: {img.ndim}")
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")

    # Otsu thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Ensure ink is dark (0) on white (255) background
    if thresh.mean() < 127.5:
        thresh = 255 - thresh

    # Crop to ink bounding box with 0 margin
    ink_mask = (thresh == 0)
    rows = np.where(ink_mask.any(axis=1))[0]
    cols = np.where(ink_mask.any(axis=0))[0]

    if len(rows) == 0 or len(cols) == 0:
        raise ValueError("No ink pixels found in image")

    r_min, r_max = int(rows[0]), int(rows[-1])
    c_min, c_max = int(cols[0]), int(cols[-1])

    cropped = thresh[r_min : r_max + 1, c_min : c_max + 1]

    # Check ink fraction on cropped glyph
    ink_frac = float(np.mean(cropped == 0))
    if ink_frac < 0.01 or ink_frac > 0.90:
        raise ValueError(f"Ink fraction {ink_frac:.4f} outside [0.01, 0.90]")

    return cropped


def _download_file(url: str, dest: Path, chunk_size: int = 1024 * 1024) -> None:
    """Download a file with streaming if not already downloaded."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        logger.info("File already exists: %s", dest)
        return

    logger.info("Downloading %s -> %s", url, dest)
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    temp_path = dest.with_suffix(dest.suffix + ".tmp")
    with open(temp_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
    temp_path.replace(dest)  # atomic overwrite on POSIX and Windows
    logger.info("Downloaded %s (%.1f MB)", dest, dest.stat().st_size / (1024 * 1024))


def fetch_alice(out_dir: str | Path = "data/external") -> pd.DataFrame:
    """Fetch and process ALICE-THI dataset.

    URL: https://www.ai.rug.nl/~mrolarik/ALICE-THI/ALICE-THI-Dataset.tar.gz
    Subsets: THI-C68 (characters) + THI-D10 (digits) -> 24,045 images.

    Returns
    -------
    pd.DataFrame
        Columns: src_path, source, orig_label, char, code, label, writer_id
    """
    out_dir = Path(out_dir)
    dl_dir = out_dir / "downloads"
    alice_img_dir = out_dir / "alice"
    alice_img_dir.mkdir(parents=True, exist_ok=True)

    tar_gz_path = dl_dir / "ALICE-THI-Dataset.tar.gz"
    url = "https://www.ai.rug.nl/~mrolarik/ALICE-THI/ALICE-THI-Dataset.tar.gz"
    _download_file(url, tar_gz_path)

    records: list[dict[str, Any]] = []

    with tarfile.open(tar_gz_path, "r:gz") as outer_tar:
        nested_file = outer_tar.extractfile("ALICE-THI Dataset/ALICE-THI Dataset.tar.gz")
        if nested_file is None:
            raise RuntimeError("Could not find inner ALICE-THI Dataset.tar.gz")
        with tarfile.open(fileobj=nested_file, mode="r:gz") as inner_tar:
            for member in inner_tar.getmembers():
                if not member.isfile() or not member.name.endswith(".png"):
                    continue

                fn = os.path.basename(member.name)
                parts = fn.split("-")
                if len(parts) < 3:
                    continue

                try:
                    tis_code = int(parts[1])
                except ValueError:
                    continue

                orig_label = parts[0]
                ch = code_to_char(tis_code)
                lbl = _code_to_index.get(tis_code, -1)

                # Determine subset (char vs digit)
                if "Thai_digit_sqr" in member.name:
                    subset = "digit"
                else:
                    subset = "char"

                # Extract sample / writer id if available
                m_writer = re.search(r"-(\d+)(?:\s*-\s*Copy)?\.png$", fn)
                writer_id = m_writer.group(1) if m_writer else ""

                # Target file path
                dest_sub = alice_img_dir / subset
                dest_sub.mkdir(parents=True, exist_ok=True)
                dest_path = dest_sub / fn

                if not dest_path.exists():
                    f = inner_tar.extractfile(member)
                    if f is not None:
                        img = Image.open(f).convert("L")
                        img.save(dest_path)

                records.append({
                    "src_path": str(dest_path),
                    "source": "alice",
                    "orig_label": orig_label,
                    "char": ch,
                    "code": tis_code,
                    "label": lbl,
                    "writer_id": writer_id,
                })

    df = pd.DataFrame(records)
    logger.info("ALICE-THI: %d images loaded (%d mapped, %d unmapped)",
                len(df), int((df["label"] >= 0).sum()), int((df["label"] < 0).sum()))
    return df


def fetch_burapha(out_dir: str | Path = "data/external") -> pd.DataFrame:
    """Fetch and process Burapha-TH dataset (character + digit).

    URL: https://services.informatics.buu.ac.th/datasets/Burapha-TH/
    Files: character/20210306-all.zip, digit/20210307-all.zip

    Returns
    -------
    pd.DataFrame
        Columns: src_path, source, orig_label, char, code, label, writer_id
    """
    out_dir = Path(out_dir)
    dl_dir = out_dir / "downloads"
    burapha_img_dir = out_dir / "burapha"
    burapha_img_dir.mkdir(parents=True, exist_ok=True)

    base_url = "https://services.informatics.buu.ac.th/datasets/Burapha-TH"
    archives = [
        ("character", "20210306-all.zip"),
        ("digit", "20210307-all.zip"),
    ]

    records: list[dict[str, Any]] = []

    for subset, zip_name in archives:
        url = f"{base_url}/{subset}/{zip_name}"
        zip_path = dl_dir / f"burapha_{subset}_{zip_name}"
        _download_file(url, zip_path)

        with zipfile.ZipFile(zip_path, "r") as zf:
            for zip_member in zf.namelist():
                if not zip_member.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                # Expected path format: all/<folder>/<filename>
                parts = zip_member.split("/")
                if len(parts) < 3:
                    continue

                folder = parts[1]
                fn = parts[2]
                folder_parts = folder.split("-")
                if len(folder_parts) < 4:
                    continue

                try:
                    tis_code = int(folder_parts[1])
                except ValueError:
                    continue

                orig_label = folder_parts[3]
                ch = code_to_char(tis_code)
                lbl = _code_to_index.get(tis_code, -1)

                # Writer ID from filename e.g. Set2_F1_P-0001_23.jpg
                m_writer = re.search(r"P-(\d+)", fn)
                writer_id = f"P-{m_writer.group(1)}" if m_writer else ""

                dest_folder = burapha_img_dir / subset / folder
                dest_folder.mkdir(parents=True, exist_ok=True)
                dest_path = dest_folder / fn

                if not dest_path.exists():
                    with zf.open(zip_member) as img_f:
                        img = Image.open(img_f).convert("L")
                        img.save(dest_path)

                records.append({
                    "src_path": str(dest_path),
                    "source": "burapha",
                    "orig_label": orig_label,
                    "char": ch,
                    "code": tis_code,
                    "label": lbl,
                    "writer_id": writer_id,
                })

    df = pd.DataFrame(records)
    logger.info("Burapha-TH: %d images loaded (%d mapped, %d unmapped)",
                len(df), int((df["label"] >= 0).sum()), int((df["label"] < 0).sum()))
    return df


def fetch_kvis(out_dir: str | Path = "data/external") -> pd.DataFrame:
    """Attempt to fetch KVIS Thai OCR dataset.

    Record outcome: Mendeley Data S3 zip returns 403 Forbidden;
    Mendeley Data web landing page requires interactive OAuth2 login.
    Skipped according to scope rules ('skip anything gated/login-only and say so').

    Returns
    -------
    pd.DataFrame
        Empty DataFrame with expected columns.
    """
    logger.warning("KVIS Thai OCR: Mendeley Data download link returns HTTP 403 Forbidden and "
                   "requires login. Skipping gated/login-only source.")
    return pd.DataFrame(columns=[
        "src_path", "source", "orig_label", "char", "code", "label", "writer_id"
    ])


def _process_one_external(args: tuple[int, str, int, int, str, str, str]) -> dict[str, Any] | None:
    """Worker function to binarize and tight-crop one image."""
    idx, path, code, label, ch, source, writer_id = args
    try:
        cropped = binarize_for_cache(path)
        h, w = cropped.shape
        ink_frac = float(np.mean(cropped == 0))
        cat = category(code)
        group = f"{source}_{writer_id}" if writer_id else f"{source}_all"
        return {
            "idx": idx,
            "path": path,
            "code": code,
            "label": label,
            "char": ch,
            "category": cat,
            "width": w,
            "height": h,
            "ink_frac": ink_frac,
            "group": group,
            "data": cropped.ravel(),
        }
    except Exception as e:
        logger.debug("Skipping %s: %s", path, e)
        return None


def build_external_cache(
    df: pd.DataFrame,
    out_npz: str | Path,
    index_csv: str | Path,
    n_workers: int = 8,
) -> tuple[int, int]:
    """Build the external glyph cache in the same .npz format as glyphs.npz.

    Filters df to mapped rows (label >= 0), binarizes, tight-crops,
    and writes out_npz and index_csv.

    Returns
    -------
    tuple[int, int]
        (n_saved, n_skipped)
    """
    out_npz = Path(out_npz)
    index_csv = Path(index_csv)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    index_csv.parent.mkdir(parents=True, exist_ok=True)

    # Filter to mapped rows only
    mapped_df = df[df["label"] >= 0].copy().reset_index(drop=True)
    total_mapped = len(mapped_df)
    logger.info("Building external cache for %d mapped images...", total_mapped)

    tasks = [
        (
            i,
            str(row["src_path"]),
            int(row["code"]),
            int(row["label"]),
            str(row["char"]),
            str(row["source"]),
            str(row.get("writer_id", "")),
        )
        for i, row in mapped_df.iterrows()
    ]

    results: list[dict[str, Any]] = []
    if n_workers > 1:
        with Pool(processes=n_workers) as pool:
            for res in pool.imap(_process_one_external, tasks, chunksize=256):
                if res is not None:
                    results.append(res)
    else:
        for t in tasks:
            res = _process_one_external(t)
            if res is not None:
                results.append(res)

    # Sort by original index to ensure deterministic order
    results.sort(key=lambda x: x["idx"])

    n_saved = len(results)
    n_skipped = total_mapped - n_saved
    logger.info("External cache: %d images kept, %d skipped (rejected ink fraction / empty)",
                n_saved, n_skipped)

    flat_parts: list[np.ndarray] = []
    offsets = np.zeros(n_saved, dtype=np.int64)
    widths = np.zeros(n_saved, dtype=np.int32)
    heights = np.zeros(n_saved, dtype=np.int32)
    paths: list[str] = []

    csv_rows: list[dict[str, Any]] = []
    offset = 0

    for i, res in enumerate(results):
        flat_parts.append(res["data"])
        offsets[i] = offset
        w = res["width"]
        h = res["height"]
        widths[i] = w
        heights[i] = h
        offset += h * w
        p = res["path"]
        paths.append(p)

        csv_rows.append({
            "path": p,
            "code": res["code"],
            "label": res["label"],
            "char": res["char"],
            "category": res["category"],
            "width": w,
            "height": h,
            "ink_frac": res["ink_frac"],
            "group": res["group"],
        })

    data = np.concatenate(flat_parts) if flat_parts else np.zeros(0, dtype=np.uint8)
    path_arr = np.array(paths, dtype=str)

    np.savez(
        str(out_npz),
        data=data,
        offsets=offsets,
        widths=widths,
        heights=heights,
        paths=path_arr,
    )

    index_df = pd.DataFrame(csv_rows)
    index_df.to_csv(str(index_csv), index=False)

    mb = out_npz.stat().st_size / (1024 * 1024)
    logger.info("Saved external cache to %s (%.1f MB, %d glyphs)", out_npz, mb, n_saved)
    logger.info("Saved index to %s (%d rows)", index_csv, len(index_df))

    return n_saved, n_skipped
