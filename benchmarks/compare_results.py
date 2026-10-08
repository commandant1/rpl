#!/usr/bin/env python3
"""Compare JSON result files and print a small table."""
import sys, json

def load(path):
    with open(path) as f: return json.load(f)

def main():
    if len(sys.argv) < 2:
        print("Usage: compare_results.py file1.json [file2.json ...]")
        sys.exit(1)
    rows = [load(p) for p in sys.argv[1:]]
    print("Framework | epochs | batch | time (s) | test_acc")
    print("---|---:|---:|---:|---:")
    for r in rows:
        print(f"{r.get('framework','?')} | {r.get('epochs', '?')} | {r.get('batch_size','?')} | {r.get('train_time_s','?'):.2f} | {r.get('test_acc','?')}")

if __name__ == '__main__':
    main()
