#!/usr/bin/env python3
"""
trace_parser.py  -  Parse MPIB NCCL TRACE logs into a flat per-message CSV.

Usage:
    python3 trace_parser.py -i nccl.log -o stage.csv

Output columns (Stage 1 flat table):
    host, pid, tid, comm, req, case, size, nreqs, dev, path, qp,
    post_send_ts, cqe_send_ts, latency_us, xput_gbps
"""

import argparse
import csv
import re
import sys

# ---------------------------------------------------------------------------
# Regex: matches any NCCL TRACE line regardless of optional timestamp prefix.
#
# Full line format (with NCCL_DEBUG_TIMESTAMP_FORMAT set):
#   [timestamp_prefix] host:pid:tid [cudaDev] elapsed_ms file:line NCCL TRACE payload
#
# Without timestamp prefix:
#   host:pid:tid [cudaDev] elapsed_ms file:line NCCL TRACE payload
#
# We anchor on the "host:pid:tid [cudaDev] elapsed_ms" portion which is
# always present for TRACE level lines.
# ---------------------------------------------------------------------------
LINE_RE = re.compile(
    r"(\S+):(\d+):(\d+)"          # host : pid : tid
    r"\s+\[\d+\]"                  # [cudaDev]  (ignored)
    r"\s+(\d+\.\d+)"               # elapsed_ms (NCCL epoch-relative)
    r"\s+\S+"                      # file:line  (ignored)
    r"\s+NCCL TRACE\s+"            # log level tag
    r"(.+)"                        # key=value payload
)

# (path, dev) → human-readable case label
CASE_LABEL = {
    (0, 0): "intra-sout",
    (0, 1): "intra-sup",
    (1, 0): "cross-sout",
    (1, 1): "cross-sup",
}

OUTPUT_FIELDS = [
    "host", "pid", "tid", "comm", "req",
    "case", "size", "nreqs", "dev", "path", "qp",
    "post_send_ts", "cqe_send_ts", "latency_us", "xput_gbps",
]


def parse_kv(payload: str) -> dict:
    """Split 'k=v k=v ...' into a dict. Skips tokens without '='."""
    result = {}
    for tok in payload.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            result[k] = v
    return result


def parse(infile, outfile):
    # pending[(host, pid, req)] = post_send record dict
    pending: dict = {}
    matched = 0
    dropped = 0

    writer = csv.DictWriter(outfile, fieldnames=OUTPUT_FIELDS)
    writer.writeheader()

    for lineno, line in enumerate(infile, 1):
        m = LINE_RE.search(line)
        if not m:
            continue

        host, pid, tid, ts_str, payload = m.groups()
        kv = parse_kv(payload)
        event = kv.get("event")

        if event not in ("post_send", "cqe_send"):
            continue

        ts = float(ts_str)
        req = kv.get("req")
        if req is None:
            continue

        key = (host, pid, kv.get("comm", ""), req)

        if event == "post_send":
            if key in pending:
                # A previous post_send for this slot was never completed —
                # could happen if the log was truncated or req was reused
                # without a matching cqe (shouldn't happen in normal flow,
                # but guard defensively).
                dropped += 1
            pending[key] = {
                "host": host,
                "pid": pid,
                "tid": tid,
                "comm": kv.get("comm", ""),
                "req": req,
                "post_send_ts": ts,
                "size": kv.get("size", ""),
                "nreqs": kv.get("nreqs", ""),
                "dev": kv.get("dev", ""),
                "path": kv.get("path", ""),
                "qp": kv.get("qp", ""),
            }

        elif event == "cqe_send":
            if key not in pending:
                # cqe_send with no matching post_send — log may be partial
                dropped += 1
                continue

            ps = pending.pop(key)
            cqe_ts = ts
            post_ts = ps["post_send_ts"]
            latency_us = (cqe_ts - post_ts) * 1000.0  # ms → µs

            try:
                size_bytes = int(ps["size"])
                xput_gbps = (size_bytes * 8) / (latency_us * 1e3) if latency_us > 0 else 0.0
            except (ValueError, ZeroDivisionError):
                xput_gbps = 0.0

            try:
                case = CASE_LABEL[(int(ps["path"]), int(ps["dev"]))]
            except (KeyError, ValueError):
                case = f"path{ps['path']}-dev{ps['dev']}"

            writer.writerow({
                "host":         ps["host"],
                "pid":          ps["pid"],
                "tid":          ps["tid"],
                "comm":         ps["comm"],
                "req":          ps["req"],
                "case":         case,
                "size":         ps["size"],
                "nreqs":        ps["nreqs"],
                "dev":          ps["dev"],
                "path":         ps["path"],
                "qp":           ps["qp"],
                "post_send_ts": f"{post_ts:.6f}",
                "cqe_send_ts":  f"{cqe_ts:.6f}",
                "latency_us":   f"{latency_us:.3f}",
                "xput_gbps":    f"{xput_gbps:.4f}",
            })
            matched += 1

    unmatched = len(pending)
    print(f"Matched pairs  : {matched}", file=sys.stderr)
    print(f"Unmatched posts: {unmatched}  (post_send with no cqe_send — truncated log?)", file=sys.stderr)
    print(f"Dropped        : {dropped}  (cqe_send with no post_send, or slot reuse)", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description="Parse MPIB NCCL traces into a flat CSV.")
    ap.add_argument("-i", "--input",  required=True, help="Input NCCL log file")
    ap.add_argument("-o", "--output", required=True, help="Output CSV file (Stage 1 table)")
    args = ap.parse_args()

    with open(args.input, "r", errors="replace") as infile, \
         open(args.output, "w", newline="") as outfile:
        parse(infile, outfile)

    print(f"Written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
