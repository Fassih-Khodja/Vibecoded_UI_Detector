"""
preprocess.py
Preprocessing pipeline for the "vibecoded vs not-vibecoded" website screenshot classifier.

Expected folder structure (run this script from the project root):
    vibecoded_project/
        preprocess.py
        vibecoded_dataset/          <- label 1
        notvibecoded_screenshots/   <- label 0

Outputs:
    processed_images/<label>/<filename>.png   resized, letterboxed, RGB, ready for training
    dataset.csv                                filename, label, path + metadata (use this for training)
    flagged_removed.csv                        everything removed/flagged, and why (REVIEW THIS FILE)

Install requirements (if needed):
    pip install pillow numpy
"""

import os
import csv
import hashlib
from pathlib import Path

from PIL import Image
import numpy as np

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
FOLDERS = {
    "notvibecoded_screenshots": 0,
    "vibecoded_dataset": 1,
}
OUTPUT_DIR = PROJECT_ROOT / "processed_images"
CSV_PATH = PROJECT_ROOT / "dataset.csv"
LOG_PATH = PROJECT_ROOT / "flagged_removed.csv"

TARGET_SIZE = 224           # standard input size for ResNet / EfficientNet-B0
BLANK_STD_THRESHOLD = 5.0   # grayscale std-dev below this => treated as near-solid-color / broken
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg"}
LETTERBOX_FILL = (255, 255, 255)  # white padding

# ---------------------------------------------------------------------------
# STEP 0 -- collect files
# ---------------------------------------------------------------------------
def collect_files():
    files = []
    for folder_name, label in FOLDERS.items():
        folder_path = PROJECT_ROOT / folder_name
        if not folder_path.exists():
            print(f"WARNING: folder not found: {folder_path}")
            continue
        for f in sorted(folder_path.iterdir()):
            if f.suffix.lower() in VALID_EXTENSIONS:
                files.append((f, label))
    return files


# ---------------------------------------------------------------------------
# STEP 1 -- exact duplicate removal, by file CONTENT hash (not just filename)
#   Two images with different names but identical pixels are still duplicates.
#   Special case flagged: the same image appearing in BOTH classes -> label conflict.
# ---------------------------------------------------------------------------
def md5_of_file(path, chunk_size=8192):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def remove_exact_duplicates(files, log_rows):
    seen = {}  # hash -> (path, label)
    kept = []
    for path, label in files:
        try:
            h = md5_of_file(path)
        except Exception as e:
            log_rows.append([str(path), label, "unreadable_for_hash", str(e)])
            continue

        if h in seen:
            prev_path, prev_label = seen[h]
            reason = "exact_duplicate"
            if prev_label != label:
                reason = "exact_duplicate_CROSS_LABEL_CONFLICT"  # same image, two labels -- investigate!
            log_rows.append([str(path), label, reason, f"duplicate_of={prev_path}"])
        else:
            seen[h] = (path, label)
            kept.append((path, label))
    return kept


# ---------------------------------------------------------------------------
# STEP 2 -- flag near-solid-color / blank images (broken or failed screenshots)
# ---------------------------------------------------------------------------
def is_blank_image(path, threshold=BLANK_STD_THRESHOLD):
    img = Image.open(path).convert("L")  # grayscale
    arr = np.asarray(img, dtype=np.float32)
    return arr.std() < threshold


def remove_blank_images(files, log_rows):
    kept = []
    for path, label in files:
        try:
            if is_blank_image(path):
                log_rows.append([str(path), label, "blank_or_solid_color", ""])
                continue
        except Exception as e:
            log_rows.append([str(path), label, "corrupted_or_unreadable", str(e)])
            continue
        kept.append((path, label))
    return kept


# ---------------------------------------------------------------------------
# STEP 3 -- resize, preserving aspect ratio, padded (letterboxed) to a square
# ---------------------------------------------------------------------------
def resize_letterbox(img, target_size=TARGET_SIZE, fill=LETTERBOX_FILL):
    img = img.convert("RGB")
    w, h = img.size
    scale = target_size / max(w, h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGB", (target_size, target_size), fill)
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    canvas.paste(img_resized, (paste_x, paste_y))
    return canvas


def parse_filename(stem, width, height):
    """
    Filenames look like: {site}_{mobile|desktop}_{number}
    e.g. '1page-in_mobile_001', '7minti_desktop_002'

    Site names may themselves contain underscores, so we split from the
    RIGHT and only take the last two underscore-separated fields as
    device + sequence number; everything else is the site name.

    Falls back to an aspect-ratio guess if the filename doesn't match
    the expected pattern (and logs it, so you can check how often that happens).
    """
    parts = stem.rsplit("_", 2)
    if len(parts) == 3:
        site, device, seq = parts
        device = device.lower()
        if device in ("mobile", "desktop"):
            return site, device, seq, False  # False = not a fallback guess

    # fallback: couldn't parse, guess from aspect ratio instead
    ratio = height / width if width else 0
    device_guess = "mobile" if ratio > 1.0 else "desktop"
    return stem, device_guess, "", True  # True = fallback was used


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------
def main():
    log_rows = []  # [path, label, reason, detail]

    print("Scanning folders...")
    files = collect_files()
    print(f"Found {len(files)} images total.")

    print("Removing exact duplicates (by content, not filename)...")
    files = remove_exact_duplicates(files, log_rows)
    print(f"{len(files)} remain after dedup.")

    print("Flagging blank / solid-color / broken screenshots...")
    files = remove_blank_images(files, log_rows)
    print(f"{len(files)} remain after blank-image removal.")

    print(f"Resizing + letterboxing to {TARGET_SIZE}x{TARGET_SIZE} and writing CSV...")
    OUTPUT_DIR.mkdir(exist_ok=True)
    for label in FOLDERS.values():
        (OUTPUT_DIR / str(label)).mkdir(exist_ok=True)

    csv_rows = []
    for path, label in files:
        try:
            img = Image.open(path)
            orig_w, orig_h = img.size
            site, device, seq, used_fallback = parse_filename(path.stem, orig_w, orig_h)
            processed = resize_letterbox(img)

            out_name = f"{path.stem}.png"
            out_path = OUTPUT_DIR / str(label) / out_name
            processed.save(out_path, "PNG")

            csv_rows.append([
                out_name, label, str(out_path),
                orig_w, orig_h, round(orig_h / orig_w, 3),
                site, device, seq, used_fallback
            ])

            if used_fallback:
                log_rows.append([str(path), label, "filename_pattern_not_matched",
                                  f"fell back to aspect-ratio guess: {device}"])
        except Exception as e:
            log_rows.append([str(path), label, "failed_during_resize", str(e)])

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label", "path", "orig_width", "orig_height",
                          "aspect_ratio_h_over_w", "source_site", "device", "seq",
                          "device_was_guessed_fallback"])
        writer.writerows(csv_rows)

    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "reason", "detail"])
        writer.writerows(log_rows)

    print(f"\nDone. {len(csv_rows)} images processed -> {CSV_PATH}")
    print(f"{len(log_rows)} images removed/flagged -> {LOG_PATH}  (review this before trusting it blindly)")


if __name__ == "__main__":
    main()
