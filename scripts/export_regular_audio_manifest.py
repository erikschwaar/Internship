"""Export audited regular mappings with paths relative to a chosen audio root."""
import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.audit.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames
        rows = []
        for row in reader:
            if row["matrix"] != "VocalizationMat":
                continue
            absolute = (args.audit_root / row["wav_path"]).resolve()
            row["wav_path"] = absolute.relative_to(args.audio_root.resolve()).as_posix()
            rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(rows)} regular mapping rows to {args.output}")


if __name__ == "__main__":
    main()
