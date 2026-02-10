#!/usr/bin/env python3
import argparse
import os
import pathlib
import re
import subprocess
import sys
from statistics import mean

def run(cmd, cwd=None):
    proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(proc.returncode)
    return proc.stdout

def parse_output(text):
    max_diff = None
    best_block = None
    m = re.findall(r"max diff:\s*([^\n]+)", text)
    if m:
        try:
            max_diff = max(float(x) for x in m)
        except ValueError:
            max_diff = m[-1].strip()
    m = re.search(r"best block:\s*(\d+)", text)
    if m:
        best_block = int(m.group(1))
    pairs = re.findall(r"([^:,\n]+):\s*([-+0-9.eE]+)\s*ms", text)
    # preserve order
    labels = [p[0].strip() for p in pairs]
    values = [float(p[1]) for p in pairs]
    return max_diff, best_block, labels, values

def main():
    parser = argparse.ArgumentParser(description="Compile and benchmark CUDA .cu files with warmup/repeat runs.")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--nvcc", default=os.environ.get("NVCC", "/usr/local/cuda/bin/nvcc"))
    parser.add_argument("--out", default="results.txt")
    args = parser.parse_args()

    root = pathlib.Path(__file__).resolve().parent
    cu_files = sorted(root.glob("*.cu"))
    if not cu_files:
        print("No .cu files found.", file=sys.stderr)
        return 1

    bin_dir = root / "bin"
    bin_dir.mkdir(exist_ok=True)

    # Compile
    for cu in cu_files:
        out = bin_dir / cu.stem
        cmd = [args.nvcc, "-O3", "-std=c++14", str(cu), "-o", str(out)]
        run(cmd)

    results = []

    for cu in cu_files:
        exe = bin_dir / cu.stem
        # warmup runs
        for _ in range(args.warmup):
            run([str(exe)])

        # measured runs
        label_order = None
        collected = {}
        best_blocks = []
        max_diffs = []
        for _ in range(args.repeat):
            out = run([str(exe)])
            max_diff, best_block, labels, values = parse_output(out)
            if max_diff is not None:
                max_diffs.append(max_diff)
            if best_block is not None:
                best_blocks.append(best_block)
            if label_order is None:
                label_order = labels
                for lbl in label_order:
                    collected[lbl] = []
            for lbl, val in zip(labels, values):
                if lbl not in collected:
                    collected[lbl] = []
                collected[lbl].append(val)

        # summarize
        summary = {
            "file": cu.stem,
            "max_diff": max(max_diffs) if max_diffs else None,
            "best_block": best_blocks[0] if best_blocks else None,
            "labels": label_order or [],
            "means": [mean(collected[lbl]) for lbl in (label_order or [])],
        }
        results.append(summary)

    # write results
    lines = []
    lines.append(f"# Benchmark results (average over {args.repeat} runs, {args.warmup} warmup runs)")
    for r in results:
        lines.append(f"=== {r['file']}")
        if r["max_diff"] is not None:
            lines.append(f"max diff: {r['max_diff']}")
        if r["best_block"] is not None:
            # keep best block line consistent with original output style
            if r["labels"]:
                # include time label if present
                if len(r["labels"]) == 1:
                    lines.append(f"best block: {r['best_block']}, {r['labels'][0]}: {r['means'][0]:.3f} ms")
                else:
                    parts = [f"{lbl}: {val:.3f} ms" for lbl, val in zip(r["labels"], r["means"])]
                    lines.append(f"best block: {r['best_block']}, " + ", ".join(parts))
            else:
                lines.append(f"best block: {r['best_block']}")
        else:
            if r["labels"]:
                parts = [f"{lbl}: {val:.3f} ms" for lbl, val in zip(r["labels"], r["means"])]
                lines.append(", ".join(parts))
        lines.append("")

    out_path = root / args.out
    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
