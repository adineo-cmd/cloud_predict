# scripts/download_ccsn.py
# Download the CCSN (Cirrus Cumulus Stratus Nimbus) database from Harvard Dataverse
# and merge it into the project's binary dataset folders.
#
# Source:  doi:10.7910/DVN/CADDPD  (Zhang et al., "CloudNet: Ground-Based Cloud
#          Classification With Deep Learning", GRL 2018) — 2,543 ground-based
#          cloud photos at 256x256 across 11 WMO cloud categories.
# Mapping: Cb (cumulonimbus)  -> dataset/cumulonimbus/ccsn_<name>.jpg
#          every other class  -> dataset/other/ccsn_<name>.jpg
# The raw zip stays in data/ (gitignored); images land in dataset/ with an
# unambiguous ccsn_ prefix so the seeded split helper can treat sources
# differently and .gitignore keeps them out of version control.
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import urllib.request

DATAVERSE_FILE_ID = "3569338"          # CCSN.zip on Harvard Dataverse
EXPECTED_MIN_BYTES = 90 * 1024 * 1024  # published artifact is ~98.5 MB

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ZIP_PATH = DATA_DIR / "CCSN.zip"
DATASET_DIR = BASE_DIR / "dataset"

# CCSN class tokens seen in filenames/folders -> binary label folder
CLASS_TOKENS = {
    "Cb": "cumulonimbus",   # cumulonimbus
    "Ac": "other", "Sc": "other", "Ns": "other", "Cu": "other",
    "Ci": "other", "Cc": "other", "As": "other", "Ct": "other",
    "Cs": "other", "St": "other", "Mixed": "other", "Mi": "other",
}


def download_zip() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists() and ZIP_PATH.stat().st_size >= EXPECTED_MIN_BYTES:
        print(f"✓ Reusing existing {ZIP_PATH} ({ZIP_PATH.stat().st_size / 1e6:.1f} MB)")
        return ZIP_PATH
    url = f"https://dataverse.harvard.edu/api/access/datafile/{DATAVERSE_FILE_ID}"
    print(f"▶ Downloading CCSN.zip from {url} ...")
    tmp = ZIP_PATH.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    if tmp.stat().st_size < EXPECTED_MIN_BYTES:
        raise SystemExit(f"✗ Download too small ({tmp.stat().st_size} bytes) — aborting.")
    tmp.replace(ZIP_PATH)
    print(f"✓ Saved {ZIP_PATH} ({ZIP_PATH.stat().st_size / 1e6:.1f} MB)")
    return ZIP_PATH


def classify(rel_path: Path) -> str | None:
    """Return the binary class folder for a CCSN member path, or None to skip."""
    parts = [p.lower() for p in rel_path.parts]
    for token, target in CLASS_TOKENS.items():
        t = token.lower()
        if t in parts or any(part.startswith(t + "_") for part in parts):
            return target
    # Fall back to the filename stem, e.g. "Cb_0123.jpg" or "Cb_0123_...jpg"
    stem = rel_path.stem.lower()
    for token, target in CLASS_TOKENS.items():
        if stem.startswith(token.lower() + "_"):
            return target
    return None


def merge(zip_path: Path) -> dict:
    counts = {"cumulonimbus": 0, "other": 0}
    skipped = 0
    with tempfile.TemporaryDirectory(prefix="ccsn_") as tmp:
        with zipfile.ZipFile(zip_path) as zf:
            members = [n for n in zf.namelist()
                       if n.lower().endswith((".jpg", ".jpeg", ".png"))
                       and not n.startswith("__MACOSX")]
            print(f"▶ Extracting {len(members)} images ...")
            zf.extractall(tmp)
        root = Path(tmp)
        for member in members:
            src = root / member
            if not src.is_file():
                skipped += 1
                continue
            target_class = classify(Path(member))
            if target_class is None:
                skipped += 1
                continue
            out_dir = DATASET_DIR / target_class
            out_dir.mkdir(parents=True, exist_ok=True)
            # 'ccsn_' prefix + sanitized relative name keeps every image unique.
            safe = member.replace("\\", "/").lstrip("/")
            safe = "/".join(p for p in safe.split("/") if p.lower() not in
                            {t.lower() for t in CLASS_TOKENS} and p != "__MACOSX")
            dest_name = "ccsn_" + safe.replace("/", "_")
            dest = out_dir / dest_name
            if not dest.exists():
                shutil.copy2(src, dest)
            counts[target_class] += 1
    return {"counts": counts, "skipped": skipped}


def main():
    zip_path = download_zip()
    result = merge(zip_path)
    print("✓ CCSN merge complete:")
    for cls, n in result["counts"].items():
        total = len(list((DATASET_DIR / cls).glob("*.jpg")))
        print(f"    dataset/{cls:<13}: +{n:>4} ccsn images  ({total} total)")
    if result["skipped"]:
        print(f"    skipped/unclassified: {result['skipped']}")
    print("  Split into train/val/test happens automatically at training time.")


if __name__ == "__main__":
    main()
