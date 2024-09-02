"""Build a TSV manifest from a directory of wavs.

Usage:
    python scripts/build_manifest.py /path/to/libritts > out.tsv
    python scripts/build_manifest.py /path/to/aishell  >> out.tsv
"""
import os
import sys
import wave


def duration(path):
    try:
        with wave.open(path) as w:
            return w.getnframes() / w.getframerate()
    except Exception:
        return -1.0


def main(root):
    for d, _, files in os.walk(root):
        for fn in files:
            if not (fn.endswith(".wav") or fn.endswith(".flac")):
                continue
            p = os.path.join(d, fn)
            dur = duration(p)
            if dur > 0.5:
                print(f"{p}\t{dur:.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
