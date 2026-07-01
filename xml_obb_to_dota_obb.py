import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


POINT_FIELDS = (
    "x_left_top",
    "y_left_top",
    "x_right_top",
    "y_right_top",
    "x_right_bottom",
    "y_right_bottom",
    "x_left_bottom",
    "y_left_bottom",
)


def text(node, path, default=None):
    found = node.find(path)
    if found is None or found.text is None:
        if default is not None:
            return default
        raise ValueError(f"missing XML field: {path}")
    return found.text.strip()


def parse_xml(xml_path):
    root = ET.parse(xml_path).getroot()
    lines = []
    for obj in root.findall("object"):
        name = text(obj, "name")
        difficult = text(obj, "difficult", "0")
        box = obj.find("robndbox")
        if box is None:
            raise ValueError(f"{xml_path}: object {name!r} has no robndbox")
        coords = [text(box, key) for key in POINT_FIELDS]
        lines.append(" ".join(coords + [name, difficult]))
    return lines


def convert_dir(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for xml_path in sorted(input_dir.glob("*.xml")):
        lines = parse_xml(xml_path)
        out_path = output_dir / f"{xml_path.stem}.txt"
        out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Convert OBB XML annotations to DOTA-OBB txt.")
    parser.add_argument("input_dir", help="directory containing .xml annotation files")
    parser.add_argument("output_dir", help="directory for converted .txt files")
    args = parser.parse_args()
    count = convert_dir(args.input_dir, args.output_dir)
    print(f"converted {count} xml files")


if __name__ == "__main__":
    main()
