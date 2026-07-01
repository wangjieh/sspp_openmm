import argparse
from pathlib import Path

from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def build_image_map(image_dir):
    images = {}
    for path in sorted(Path(image_dir).rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            images.setdefault(path.stem, path)
    return images


def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.size


def is_meta_line(line):
    lowered = line.lower()
    return lowered.startswith("imagesource:") or lowered.startswith("gsd:")


def is_valid_hbb_line(line, image_size):
    parts = line.split()
    if len(parts) < 4:
        return False
    try:
        xmin, ymin, xmax, ymax = [float(v) for v in parts[:4]]
    except ValueError:
        return False

    img_w, img_h = image_size
    if xmax - xmin <= 1 or ymax - ymin <= 1:
        return False
    if min(xmin, xmax) < 0 or max(xmin, xmax) > img_w:
        return False
    if min(ymin, ymax) < 0 or max(ymin, ymax) > img_h:
        return False
    return True


def clean_annotation(ann_path, image_path):
    image_size = get_image_size(image_path)
    kept_lines = []
    removed = 0

    for raw_line in Path(ann_path).read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if is_meta_line(line):
            kept_lines.append(line)
            continue
        if is_valid_hbb_line(line, image_size):
            kept_lines.append(line)
        else:
            removed += 1

    return kept_lines, removed


def clean_dataset(image_dir, annotation_dir, output_dir):
    image_map = build_image_map(image_dir)
    annotation_dir = Path(annotation_dir)
    output_dir = Path(output_dir)

    checked = 0
    removed_boxes = 0
    written_files = 0

    for ann_path in sorted(annotation_dir.rglob("*.txt")):
        checked += 1
        rel_path = ann_path.relative_to(annotation_dir)
        image_path = image_map.get(ann_path.stem)
        if image_path is None:
            raise FileNotFoundError(f"missing image for annotation: {ann_path}")

        kept_lines, removed = clean_annotation(ann_path, image_path)
        removed_boxes += removed

        out_path = output_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            "\n".join(kept_lines) + ("\n" if kept_lines else ""),
            encoding="utf-8",
        )
        written_files += 1

    return {
        "summary": {
            "checked_annotations": checked,
            "written_annotations": written_files,
            "removed_boxes": removed_boxes,
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Clean invalid HBB DOTA annotation boxes.")
    parser.add_argument("image_dir", help="directory containing images")
    parser.add_argument("annotation_dir", help="directory containing DOTA-HBB txt annotations")
    parser.add_argument("output_dir", help="directory to save cleaned DOTA-HBB txt annotations")
    args = parser.parse_args()
    result = clean_dataset(args.image_dir, args.annotation_dir, args.output_dir)
    summary = result["summary"]
    print(f"checked {summary['checked_annotations']} annotations")
    print(f"wrote {summary['written_annotations']} cleaned annotations")
    print(f"removed {summary['removed_boxes']} invalid boxes")


if __name__ == "__main__":
    main()
