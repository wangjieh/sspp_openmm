import argparse
import json
import math
from pathlib import Path

from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def build_image_map(image_dir):
    image_dir = Path(image_dir)
    images = {}
    for path in sorted(image_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            images.setdefault(path.stem, path)
    return images


def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.size


def is_meta_line(line):
    lowered = line.lower()
    return lowered.startswith("imagesource:") or lowered.startswith("gsd:")


def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def check_line(line, line_no, image_size):
    parts = line.split()
    if len(parts) < 8:
        return [{
            "line": line_no,
            "reason": "parse_error",
            "message": "expected at least 8 coordinate values",
        }]

    try:
        coords = [float(v) for v in parts[:8]]
    except ValueError:
        return [{
            "line": line_no,
            "reason": "parse_error",
            "message": "first 8 fields must be numbers",
        }]

    points = list(zip(coords[0::2], coords[1::2]))
    img_w, img_h = image_size
    box_w = min(distance(points[0], points[1]), distance(points[2], points[3]))
    box_h = min(distance(points[1], points[2]), distance(points[3], points[0]))
    xs = coords[0::2]
    ys = coords[1::2]
    issues = []

    if box_w < 1 or box_h < 1:
        issues.append({
            "line": line_no,
            "reason": "box_size_lt_1",
            "bbox": coords,
            "box_width": box_w,
            "box_height": box_h,
        })

    if min(xs) < 0 or max(xs) > img_w or min(ys) < 0 or max(ys) > img_h:
        issues.append({
            "line": line_no,
            "reason": "box_out_of_bounds",
            "bbox": coords,
            "image_size": [img_w, img_h],
        })

    return issues


def validate_annotation(ann_path, image_path):
    image_size = get_image_size(image_path)
    issues = []
    for line_no, line in enumerate(Path(ann_path).read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or is_meta_line(line):
            continue
        issues.extend(check_line(line, line_no, image_size))
    return issues


def validate_dataset(image_dir, annotation_dir, output_json):
    image_map = build_image_map(image_dir)
    annotation_dir = Path(annotation_dir)
    details = {}

    ann_paths = sorted(annotation_dir.rglob("*.txt"))
    for ann_path in ann_paths:
        rel_name = ann_path.relative_to(annotation_dir).as_posix()
        image_path = image_map.get(ann_path.stem)
        if image_path is None:
            details[rel_name] = [{"reason": "missing_image"}]
            continue

        issues = validate_annotation(ann_path, image_path)
        if issues:
            details[rel_name] = issues

    result = {
        "summary": {
            "checked_annotations": len(ann_paths),
            "invalid_files": len(details),
        },
        "invalid_files": sorted(details),
        "details": {name: details[name] for name in sorted(details)},
    }

    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description="Find invalid OBB DOTA annotation files.")
    parser.add_argument("image_dir", help="directory containing images")
    parser.add_argument("annotation_dir", help="directory containing DOTA-OBB txt annotations")
    parser.add_argument("output_json", help="path to save invalid file report as JSON")
    args = parser.parse_args()
    result = validate_dataset(args.image_dir, args.annotation_dir, args.output_json)
    print(f"checked {result['summary']['checked_annotations']} annotations")
    print(f"found {result['summary']['invalid_files']} invalid files")


if __name__ == "__main__":
    main()
