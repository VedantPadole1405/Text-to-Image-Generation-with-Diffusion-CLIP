#!/usr/bin/env python
"""
dataset_setup.py  – download + prepare ImageNet‑128 or Tiny‑ImageNet
--------------------------------------------------------------------
USAGE EXAMPLES
--------------

# 1 ImageNet‑128 (≈13 GB)  – stream‑download + convert
python dataset_setup.py --dataset imagenet128 --out imagenet128

# 2 Tiny‑ImageNet (≈250 MB) – download zip + prepare
python dataset_setup.py --dataset tiny-imagenet --out tiny-imagenet

# 3 Re‑split any existing ImageFolder tree (e.g. Tiny‑ImageNet train)
python dataset_setup.py --split_only --src tiny-imagenet/train \
                        --out tiny-imagenet-split --val_pct 10
"""
import argparse, os, shutil, sys, tarfile, zipfile, random
from pathlib import Path

# ─────────────────────────────────────────────────── utils
def log(msg: str):
    print(f"[•] {msg}", flush=True)

# ─────────────────────────────────────────────────── download helpers
def download_file(url: str, dest: Path):
    import urllib.request, tqdm
    if dest.exists():
        log(f"{dest.name} already exists – skipping download.")
        return
    log(f"Downloading {url}")
    with urllib.request.urlopen(url) as resp, open(dest, "wb") as f, tqdm.tqdm(
        total=int(resp.headers["Content-Length"]), unit="B", unit_scale=True
    ) as pbar:
        while chunk := resp.read(8192):
            f.write(chunk)
            pbar.update(len(chunk))

# ─────────────────────────────────────────────────── imagenet‑128
# ─────────────────────────────────────────────────── imagenet‑128  (fixed)
def prepare_imagenet128(out_root: Path):
    """
    Stream‑download benjamin‑paine/imagenet‑1k‑128x128 and write to
    <out_root>/train/<class>/*.jpg   and   …/val/<class>/*.jpg
    """
    from datasets import load_dataset, Image as HFImage
    from PIL import Image
    from io import BytesIO
    import tqdm, itertools

    for split, sub in [("train", "train"), ("validation", "val")]:
        out_dir = out_root / sub
        out_dir.mkdir(parents=True, exist_ok=True)

        log(f"Streaming HF split “{split}” → {out_dir}")
        ds = load_dataset(
            "benjamin-paine/imagenet-1k-128x128",
            split=split,
            streaming=True,
            trust_remote_code=True,
        )

        # detect column name
        image_col = "img" if "img" in ds.features else "image"
        label_col = "label"
        label_names = ds.features[label_col].names

        # make sure Hugging Face treats the column as an image
        ds = ds.cast_column(image_col, HFImage())
        ds = ds.with_format("python")  # returns dict, img is bytes

        for i, sample in enumerate(tqdm.tqdm(ds, unit="img")):
            cls_name = label_names[sample[label_col]]
            cls_dir  = out_dir / cls_name
            cls_dir.mkdir(parents=True, exist_ok=True)

            img_data = sample[image_col]

            # img_data can be a dict with {"bytes": …, "path":…}
            if isinstance(img_data, dict):
                img = Image.open(BytesIO(img_data["bytes"]))
            elif isinstance(img_data, (bytes, bytearray)):
                img = Image.open(BytesIO(img_data))
            else:                       # already a PIL.Image
                img = img_data

            img.save(cls_dir / f"{i:08}.jpg", quality=95)

        log(f"✔ {split} split done – saved to {out_dir}")


# ─────────────────────────────────────────────────── tiny‑imagenet
TINY_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
def prepare_tiny_imagenet(out_root: Path):

    out_root = Path(out_root).expanduser().resolve()     # «~/data» → «/home/kpate271/data»
    out_root.mkdir(parents=True, exist_ok=True)          # create target dir if needed

    zip_path = out_root / "tiny-imagenet-200.zip"        # NOT at filesystem root
    download_file(TINY_URL, zip_path)

    log("Unzipping …")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_root.parent)
    src_root = out_root.parent / "tiny-imagenet-200"

    # ---- fix validation split (originally all images in val/images + annotations.tsv)
    val_dir   = src_root / "val"
    val_imgs  = val_dir / "images"
    val_map   = val_dir / "val_annotations.txt"
    log("Rewriting Tiny‑ImageNet val/ into per‑class sub‑folders …")
    labels = {}
    with open(val_map) as f:
        for line in f:
            img, cls, *_ = line.strip().split("\t")
            labels[img] = cls
    for img_file in val_imgs.iterdir():
        cls = labels[img_file.name]
        dest = val_dir / cls / "images"
        dest.mkdir(parents=True, exist_ok=True)
        shutil.move(img_file, dest / img_file.name)
    shutil.rmtree(val_imgs)  # empty now

    # final rename to user‑requested out_root
    if out_root.exists():
        shutil.rmtree(out_root)
    src_root.rename(out_root)
    log("✔ Tiny‑ImageNet ready.")

# ─────────────────────────────────────────────────── re‑split helper
def split_imagefolder(src: Path, out: Path, val_pct: float, seed: int = 42):
    random.seed(seed)
    train_dst = out/"train"; val_dst = out/"val"
    for cls_dir in src.iterdir():
        if not cls_dir.is_dir(): continue
        images = list(cls_dir.glob("*.[jp][pn]*g"))  # jpg/png
        random.shuffle(images)
        k = round(len(images)*val_pct/100)
        val_imgs, train_imgs = images[:k], images[k:]
        for subset, imgs in [("train", train_imgs), ("val", val_imgs)]:
            dest_cls = (out/subset/cls_dir.name)
            dest_cls.mkdir(parents=True, exist_ok=True)
            for img in imgs:
                shutil.copy2(img, dest_cls/img.name)
    log(f"✔ Split finished → {train_dst} + {val_dst}")

# ─────────────────────────────────────────────────── main
def main():
    ap = argparse.ArgumentParser(description="Dataset downloader / splitter")
    ap.add_argument("--dataset", choices=["imagenet128", "tiny-imagenet"],
                    help="which dataset to fetch & prepare")
    ap.add_argument("--out", required=True, help="destination root folder")
    ap.add_argument("--split_only", action="store_true",
                    help="skip download; only split an existing ImageFolder tree")
    ap.add_argument("--src", help="ImageFolder source for --split_only")
    ap.add_argument("--val_pct", type=float, default=5,
                    help="val percentage for --split_only")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_root = Path(args.out).expanduser()

    if args.split_only:
        if not args.src:
            sys.exit("Error: --src is required in --split_only mode.")
        split_imagefolder(Path(args.src).expanduser(), out_root,
                          val_pct=args.val_pct, seed=args.seed)
        return

    if args.dataset == "imagenet128":
        prepare_imagenet128(out_root)
    elif args.dataset == "tiny-imagenet":
        prepare_tiny_imagenet(out_root)
    else:
        sys.exit("Error: specify --dataset imagenet128 or tiny-imagenet")

if __name__ == "__main__":
    main()
