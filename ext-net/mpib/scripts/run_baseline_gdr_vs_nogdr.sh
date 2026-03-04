#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LAUNCHER="${SCRIPT_DIR}/launch_nccl.sh"
[[ -x "${LAUNCHER}" ]] || { echo "ERROR: launcher missing/executable: ${LAUNCHER}" >&2; exit 2; }

TRIALS=${TRIALS:-5}
LOG_DIR=${LOG_DIR:-${SCRIPT_DIR}/log}
mkdir -p "${LOG_DIR}"

echo "[INFO] Running ${TRIALS} trials for each mode"
echo "[INFO] Logs will be written under: ${LOG_DIR}"

run_mode() {
  local mode="$1" # gdr|nogdr
  local gdr_mode="$2" # on|off
  local i out
  for ((i=1; i<=TRIALS; i++)); do
    out="${LOG_DIR}/${mode}_trial$(printf '%02d' "${i}").log"
    echo "[INFO] ${mode} trial ${i}/${TRIALS} -> ${out}"
    GDR_MODE="${gdr_mode}" "${LAUNCHER}" > "${out}" 2>&1
  done
}

run_mode gdr on
run_mode nogdr off

python3 - "$LOG_DIR" "$TRIALS" <<'PY'
import pathlib, re, statistics, sys

log_dir = pathlib.Path(sys.argv[1])
trials = int(sys.argv[2])

avg_re = re.compile(r"#\s*Avg bus bandwidth\s*:\s*([0-9.]+)")
gdr_re = re.compile(r"Connected all rings, use ring PXN\s+\d+\s+GDR\s+(\d+)")
line_1g_re = re.compile(r"^\s*1073741824\s+\d+\s+\w+\s+\w+\s+-?\d+\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+\d+\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)")


def parse_log(path: pathlib.Path):
    txt = path.read_text(errors="ignore")
    avg = avg_re.findall(txt)
    if not avg:
        raise RuntimeError(f"Missing avg bus bandwidth in {path}")
    avg_bw = float(avg[-1])

    one_g = None
    for line in txt.splitlines():
        m = line_1g_re.match(line)
        if m:
            o_time, o_alg, o_bus, i_time, i_alg, i_bus = map(float, m.groups())
            one_g = {
                "o_time_us": o_time,
                "o_alg_gbs": o_alg,
                "o_bus_gbs": o_bus,
                "i_time_us": i_time,
                "i_alg_gbs": i_alg,
                "i_bus_gbs": i_bus,
            }
            break
    if one_g is None:
        raise RuntimeError(f"Missing 1GiB line in {path}")

    gdr_flags = [int(v) for v in gdr_re.findall(txt)]
    gdr_seen = max(gdr_flags) if gdr_flags else None

    return {
        "avg_bw": avg_bw,
        "one_g": one_g,
        "gdr_seen": gdr_seen,
    }


def summarize(mode):
    rows = []
    for i in range(1, trials + 1):
        p = log_dir / f"{mode}_trial{i:02d}.log"
        rows.append((i, parse_log(p)))

    avg_vals = [r[1]["avg_bw"] for r in rows]
    oneg_o = [r[1]["one_g"]["o_alg_gbs"] for r in rows]
    oneg_i = [r[1]["one_g"]["i_alg_gbs"] for r in rows]
    gdr_vals = [r[1]["gdr_seen"] for r in rows]

    return {
        "rows": rows,
        "avg_mean": statistics.mean(avg_vals),
        "avg_p50": statistics.median(avg_vals),
        "avg_p90": statistics.quantiles(avg_vals, n=10, method="inclusive")[8] if len(avg_vals) >= 2 else avg_vals[0],
        "o_mean": statistics.mean(oneg_o),
        "o_p50": statistics.median(oneg_o),
        "o_p90": statistics.quantiles(oneg_o, n=10, method="inclusive")[8] if len(oneg_o) >= 2 else oneg_o[0],
        "i_mean": statistics.mean(oneg_i),
        "i_p50": statistics.median(oneg_i),
        "i_p90": statistics.quantiles(oneg_i, n=10, method="inclusive")[8] if len(oneg_i) >= 2 else oneg_i[0],
        "gdr_vals": gdr_vals,
    }


gdr = summarize("gdr")
nogdr = summarize("nogdr")

summary_csv = log_dir / "summary.csv"
with summary_csv.open("w", encoding="utf-8") as f:
    f.write("mode,trial,avg_bus_bw,one_gib_o_algbw,one_gib_o_busbw,one_gib_i_algbw,one_gib_i_busbw,gdr_seen\n")
    for mode, obj in (("gdr", gdr), ("nogdr", nogdr)):
        for trial, item in obj["rows"]:
            one = item["one_g"]
            f.write(
                f"{mode},{trial},{item['avg_bw']:.5f},{one['o_alg_gbs']:.3f},{one['o_bus_gbs']:.3f},{one['i_alg_gbs']:.3f},{one['i_bus_gbs']:.3f},{item['gdr_seen']}\n"
            )

summary_md = log_dir / "summary.md"
with summary_md.open("w", encoding="utf-8") as f:
    f.write("# GDR vs non-GDR run summary\n\n")
    f.write(f"- Trials per mode: {trials}\n")
    f.write("- Metrics extracted from nccl-tests `all_reduce_perf` logs\n\n")

    f.write("## Aggregated metrics\n\n")
    f.write("| mode | avg bus BW mean | avg bus BW p50 | avg bus BW p90 | 1GiB out-of-place algBW mean | 1GiB in-place algBW mean | GDR marker values |\n")
    f.write("|---|---:|---:|---:|---:|---:|---|\n")
    f.write(
        f"| gdr | {gdr['avg_mean']:.5f} | {gdr['avg_p50']:.5f} | {gdr['avg_p90']:.5f} | {gdr['o_mean']:.3f} | {gdr['i_mean']:.3f} | {gdr['gdr_vals']} |\n"
    )
    f.write(
        f"| nogdr | {nogdr['avg_mean']:.5f} | {nogdr['avg_p50']:.5f} | {nogdr['avg_p90']:.5f} | {nogdr['o_mean']:.3f} | {nogdr['i_mean']:.3f} | {nogdr['gdr_vals']} |\n"
    )

    f.write("\n## Trial details\n\n")
    f.write("| mode | trial | avg bus BW | 1GiB out algBW | 1GiB in algBW | GDR marker |\n")
    f.write("|---|---:|---:|---:|---:|---:|\n")
    for mode, obj in (("gdr", gdr), ("nogdr", nogdr)):
        for trial, item in obj["rows"]:
            one = item["one_g"]
            f.write(
                f"| {mode} | {trial} | {item['avg_bw']:.5f} | {one['o_alg_gbs']:.3f} | {one['i_alg_gbs']:.3f} | {item['gdr_seen']} |\n"
            )

print(f"[INFO] Wrote {summary_csv}")
print(f"[INFO] Wrote {summary_md}")
PY

echo "[INFO] Done."
