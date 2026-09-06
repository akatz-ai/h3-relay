#!/usr/bin/env python3
"""Record hashes, stream contracts, and full-decode checks for validation media."""
import argparse
import hashlib
import json
import pathlib
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=pathlib.Path)
    parser.add_argument("--report", type=pathlib.Path, required=True)
    parser.add_argument("--pattern", default="*.mp4", help="Glob relative to root; select deliverables, not picture-only intermediates")
    args = parser.parse_args()
    records = []
    for path in sorted(args.root.rglob(args.pattern)):
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        probe = subprocess.run([
            "ffprobe", "-v", "error", "-show_streams", "-show_format",
            "-of", "json", str(path),
        ], capture_output=True, text=True, check=True)
        metadata = json.loads(probe.stdout)
        decode = subprocess.run([
            "ffmpeg", "-v", "error", "-xerror", "-i", str(path),
            "-map", "0:v", "-map", "0:a", "-f", "null", "-",
        ], capture_output=True, text=True)
        kinds = {s["codec_type"] for s in metadata["streams"]}
        records.append({
            "path": str(path), "sha256": digest, "probe": metadata,
            "decode_exit": decode.returncode, "decode_errors": decode.stderr,
            "passed": decode.returncode == 0 and {"video", "audio"} <= kinds,
        })
    with args.report.open("x") as handle:
        json.dump(records, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"media": len(records), "passed": sum(r["passed"] for r in records)}))
    return 0 if records and all(r["passed"] for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
