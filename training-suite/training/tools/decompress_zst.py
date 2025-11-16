from __future__ import annotations

import argparse
import os
import sys

import zstandard as zstd


def decompress(input_path: str, output_path: str):
    input_path = os.path.abspath(os.path.expanduser(input_path))
    output_path = os.path.abspath(os.path.expanduser(output_path))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dctx = zstd.ZstdDecompressor()
    with open(input_path, "rb") as compressed, open(output_path, "wb") as decompressed:
        with dctx.stream_reader(compressed) as reader:
            while True:
                chunk = reader.read(1 << 20)
                if not chunk:
                    break
                decompressed.write(chunk)
    print(f"Decompressed {input_path} -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Decompress a .zst archive using zstandard.")
    parser.add_argument("--input", required=True, help="Path to the .zst file")
    parser.add_argument("--output", required=True, help="Destination path for the decompressed file")
    args = parser.parse_args()

    if not args.input.lower().endswith(".zst"):
        print("Warning: input file does not end with .zst", file=sys.stderr)

    decompress(args.input, args.output)


if __name__ == "__main__":
    main()
