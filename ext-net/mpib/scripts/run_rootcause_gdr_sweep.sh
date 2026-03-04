#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LAUNCHER="${SCRIPT_DIR}/launch_nccl.sh"
[[ -x "${LAUNCHER}" ]] || { echo "ERROR: launcher missing/executable: ${LAUNCHER}" >&2; exit 2; }

TRIALS=${TRIALS:-3}
OUT_DIR=${OUT_DIR:-${SCRIPT_DIR}/log/rootcause}
mkdir -p "${OUT_DIR}"

# Clean previous root-cause sweep traces for reproducibility.
find "${OUT_DIR}" -maxdepth 1 -type f -name '*.log' -delete

echo "[INFO] Root-cause sweep: ${TRIALS} trials per config"
echo "[INFO] Output dir: ${OUT_DIR}"

run_case() {
  local case_name="$1"
  local gdr_mode="$2"
  local binding="$3"
  local qps="$4"
  local gdr_read="$5"
  local i out

  for ((i=1; i<=TRIALS; i++)); do
    out="${OUT_DIR}/${case_name}_trial$(printf '%02d' "${i}").log"
    echo "[INFO] ${case_name} trial ${i}/${TRIALS}"

    local -a env_cmd=(
      "GDR_MODE=${gdr_mode}"
      "MPIRUN_BINDING=${binding}"
      "NCCL_IB_QPS_PER_CONNECTION=${qps}"
      "NCCL_NET_GDR_READ=${gdr_read}"
    )

    # Use env -u to avoid carrying knobs from shell when intentionally unset.
    env -u NCCL_IB_QPS_PER_CONNECTION -u NCCL_NET_GDR_READ \
      "${env_cmd[@]}" "${LAUNCHER}" > "${out}" 2>&1
  done
}

# ---------- Binding policy matrix (GDR vs no-GDR) ----------
run_case "bind_none_gdr"   "on"  "--bind-to none" "" ""
run_case "bind_core_gdr"   "on"  "--bind-to core" "" ""
run_case "bind_numa_gdr"   "on"  "--bind-to numa" "" ""
run_case "bind_none_nogdr" "off" "--bind-to none" "" ""
run_case "bind_core_nogdr" "off" "--bind-to core" "" ""
run_case "bind_numa_nogdr" "off" "--bind-to numa" "" ""

# ---------- Other likely GDR limiters ----------
# QPs per connection on GDR path
run_case "qps1_gdr" "on" "--bind-to none" "1" ""
run_case "qps2_gdr" "on" "--bind-to none" "2" ""
run_case "qps4_gdr" "on" "--bind-to none" "4" ""

# GDR read path toggle
run_case "gdrread0_gdr" "on" "--bind-to none" "" "0"
run_case "gdrread1_gdr" "on" "--bind-to none" "" "1"

python3 - "$OUT_DIR" "$TRIALS" <<'PY'
import collections
import pathlib
import re
import statistics
import sys

out_dir = pathlib.Path(sys.argv[1])
trials = int(sys.argv[2])

avg_re = re.compile(r"#\s*Avg bus bandwidth\s*:\s*([0-9.]+)")
gdr_re = re.compile(r"Connected all rings, use ring PXN\s+\d+\s+GDR\s+(\d+)")
line_1g_re = re.compile(r"^\s*1073741824\s+\d+\s+\w+\s+\w+\s+-?\d+\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+\d+\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)")
chan_re = re.compile(r"(\d+) coll channels")


def parse(path: pathlib.Path):
    txt = path.read_text(errors="ignore")

    avg = avg_re.findall(txt)
    if not avg:
        raise RuntimeError(f"Missing Avg bus BW in {path}")
    avg_bw = float(avg[-1])

    oneg = None
    for line in txt.splitlines():
        m = line_1g_re.match(line)
        if m:
            o_time, o_alg, o_bus, i_time, i_alg, i_bus = map(float, m.groups())
            oneg = {
                "o_alg": o_alg,
                "o_bus": o_bus,
                "i_alg": i_alg,
                "i_bus": i_bus,
            }
            break
    if oneg is None:
        raise RuntimeError(f"Missing 1GiB row in {path}")

    gdr_marks = [int(x) for x in gdr_re.findall(txt)]
    gdr_seen = max(gdr_marks) if gdr_marks else None

    chans = [int(x) for x in chan_re.findall(txt)]
    coll_channels = chans[0] if chans else None

    return {
        "avg_bw": avg_bw,
        "oneg": oneg,
        "gdr_seen": gdr_seen,
        "coll_channels": coll_channels,
    }


def p90(vals):
    return statistics.quantiles(vals, n=10, method="inclusive")[8] if len(vals) >= 2 else vals[0]

cases = collections.defaultdict(list)
for p in sorted(out_dir.glob("*_trial*.log")):
    case = p.name.split("_trial")[0]
    cases[case].append((p, parse(p)))

rows = []
for case, items in sorted(cases.items()):
    vals_avg = [x[1]["avg_bw"] for x in items]
    vals_1g_o = [x[1]["oneg"]["o_alg"] for x in items]
    vals_1g_i = [x[1]["oneg"]["i_alg"] for x in items]
    gdr_vals = [x[1]["gdr_seen"] for x in items]
    channels = [x[1]["coll_channels"] for x in items]

    rows.append({
        "case": case,
        "n": len(items),
        "avg_mean": statistics.mean(vals_avg),
        "avg_p50": statistics.median(vals_avg),
        "avg_p90": p90(vals_avg),
        "o1g_mean": statistics.mean(vals_1g_o),
        "o1g_p50": statistics.median(vals_1g_o),
        "o1g_p90": p90(vals_1g_o),
        "i1g_mean": statistics.mean(vals_1g_i),
        "i1g_p50": statistics.median(vals_1g_i),
        "i1g_p90": p90(vals_1g_i),
        "gdr_vals": gdr_vals,
        "channels": channels,
    })

csv_path = out_dir / "rootcause_summary.csv"
with csv_path.open("w", encoding="utf-8") as f:
    f.write("case,trials,avg_bw_mean,avg_bw_p50,avg_bw_p90,oneg_o_mean,oneg_o_p50,oneg_o_p90,oneg_i_mean,oneg_i_p50,oneg_i_p90,gdr_vals,coll_channels\n")
    for r in rows:
        f.write(
            f"{r['case']},{r['n']},{r['avg_mean']:.5f},{r['avg_p50']:.5f},{r['avg_p90']:.5f},{r['o1g_mean']:.3f},{r['o1g_p50']:.3f},{r['o1g_p90']:.3f},{r['i1g_mean']:.3f},{r['i1g_p50']:.3f},{r['i1g_p90']:.3f},\"{r['gdr_vals']}\",\"{r['channels']}\"\n"
        )

md_path = out_dir / "rootcause_summary.md"
with md_path.open("w", encoding="utf-8") as f:
    f.write("# GDR root-cause sweep summary\n\n")
    f.write(f"- trials per config: {trials}\n")
    f.write("- metrics from nccl-tests all_reduce_perf logs\n\n")
    f.write("| case | avg BW mean | 1GiB out mean | 1GiB in mean | GDR markers | coll channels |\n")
    f.write("|---|---:|---:|---:|---|---|\n")
    for r in rows:
        f.write(
            f"| {r['case']} | {r['avg_mean']:.5f} | {r['o1g_mean']:.3f} | {r['i1g_mean']:.3f} | {r['gdr_vals']} | {r['channels']} |\n"
        )

print(f"[INFO] Wrote {csv_path}")
print(f"[INFO] Wrote {md_path}")
PY

echo "[INFO] Sweep complete"
