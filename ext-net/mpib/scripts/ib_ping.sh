#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Cross-island relay ping prep across all 8 containers in parallel
#
# Run this script directly on c1. Commands targeting c1 execute locally;
# commands targeting the other containers use SSH.
# =============================================================================

GID_IDX=3
IB_PORT=1
BASE_TCP_PORT=24000  # distinct from other scripts to avoid collisions
MTU=1024
ITERS=100
SIZE=4096
TIMEOUT=30

LOG_DIR="/tmp/ping_prep_n8_$(date +%Y%m%d_%H%M%S)"
STATUS_DIR="${LOG_DIR}/status"
mkdir -p "$LOG_DIR" "$STATUS_DIR"

# ---------------------------------------------------------------------------
# Topology helpers (macOS Bash 3.2 compatible)
# ---------------------------------------------------------------------------
declare -a VF1_RDMA

auto_fail() {
  echo "ERROR: $*" >&2
  exit 1
}

ctr_id() {
  local ctr=$1
  echo "${ctr#c}"
}

ns_name() {
  local id
  id=$(ctr_id "$1")
  echo "ns${id}"
}

pf1_ip() {
  local id
  id=$(ctr_id "$1")
  if (( id <= 4 )); then
    echo "10.9.1.${id}"
  else
    echo "10.9.2.${id}"
  fi
}

run_in_container() {
  local ctr=$1
  local cmd=$2

  if [[ "$ctr" == "c1" ]]; then
    bash -lc "$cmd"
  else
    ssh "$ctr" "bash -lc $(printf '%q' "$cmd")"
  fi
}

resolve_rdma_dev() {
  local ctr=$1
  case "$ctr" in
    c1|c5) echo "mlx5_9" ;;
    c2|c6) echo "mlx5_11" ;;
    c3|c7) echo "mlx5_13" ;;
    c4|c8) echo "mlx5_15" ;;
    *) auto_fail "Unknown container: ${ctr}" ;;
  esac
}

wait_with_timeout() {
  local pid=$1 timeout=$2
  local elapsed=0
  while kill -0 "$pid" 2>/dev/null; do
    if (( elapsed >= timeout )); then
      kill "$pid" 2>/dev/null || true
      return 1
    fi
    sleep 1
    (( elapsed++ ))
  done
  wait "$pid" 2>/dev/null
  return $?
}

write_status() {
  local status_file=$1
  local status=$2
  local detail=$3
  {
    echo "status=${status}"
    echo "detail=${detail}"
  } > "$status_file"
}

# ---------------------------------------------------------------------------
# Resolve VF1 RDMA devices from static topology table
# ---------------------------------------------------------------------------
echo "=== Resolving VF1 RDMA device names from containers ==="
echo "=== Local container: c1 ==="
for i in 1 2 3 4 5 6 7 8; do
  ctr="c${i}"
  ns="$(ns_name "$ctr")"
  vf1_rdma=$(resolve_rdma_dev "$ctr")

  if [[ -z "$vf1_rdma" ]]; then
    auto_fail "Failed to resolve VF1 RDMA device for ${ctr} (${ns}vf1)"
  fi

  VF1_RDMA[$i]="$vf1_rdma"
  echo "  ${ctr}: vf1=${ns}vf1 -> ${vf1_rdma} ip=$(pf1_ip "$ctr")"
done

# ---------------------------------------------------------------------------
# Cross-island relay pairs only
# ---------------------------------------------------------------------------
ISLAND_A_CTRS=(c1 c2 c3 c4)
ISLAND_B_CTRS=(c5 c6 c7 c8)

declare -a VF1_CROSS_PAIRS=()
declare -a PAIR_PIDS=()
declare -a PAIR_TAGS=()

for src in "${ISLAND_A_CTRS[@]}"; do
  for dst in "${ISLAND_B_CTRS[@]}"; do
    VF1_CROSS_PAIRS+=("${src}:${dst}")
  done
done

run_cross_island_pingpong() {
  local server_ctr=$1
  local client_ctr=$2
  local tcp_port=$3

  local server_id client_id
  server_id=$(ctr_id "$server_ctr")
  client_id=$(ctr_id "$client_ctr")

  local server_ip client_ip server_rdma client_rdma
  server_ip="$(pf1_ip "$server_ctr")"
  client_ip="$(pf1_ip "$client_ctr")"
  server_rdma="${VF1_RDMA[$server_id]}"
  client_rdma="${VF1_RDMA[$client_id]}"

  local pair_tag="vf1_cross_${server_ctr}_${client_ctr}"
  local server_log="${LOG_DIR}/${pair_tag}_server.log"
  local client_log="${LOG_DIR}/${pair_tag}_client.log"
  local status_file="${STATUS_DIR}/${pair_tag}.status"

  echo "[START] ${pair_tag}: ${server_ctr}(${server_ip}, ${server_rdma}) <-> ${client_ctr}(${client_ip}, ${client_rdma}) port=${tcp_port}"

  run_in_container "$server_ctr" "
    ibv_rc_pingpong \
      -d $server_rdma \
      -i $IB_PORT \
      -g $GID_IDX \
      -p $tcp_port \
      -s $SIZE \
      -m $MTU \
      -n $ITERS
  " > "$server_log" 2>&1 &
  local server_pid=$!

  sleep 2

  run_in_container "$client_ctr" "
    ibv_rc_pingpong \
      -d $client_rdma \
      -i $IB_PORT \
      -g $GID_IDX \
      -p $tcp_port \
      -s $SIZE \
      -m $MTU \
      -n $ITERS \
      $server_ip
  " > "$client_log" 2>&1 &
  local client_pid=$!

  local ok=true
  local reason=""
  if ! wait_with_timeout "$client_pid" "$TIMEOUT"; then
    ok=false
    reason="client_timeout"
  fi
  if ! wait_with_timeout "$server_pid" "$TIMEOUT"; then
    ok=false
    if [[ -n "$reason" ]]; then
      reason="${reason},server_timeout"
    else
      reason="server_timeout"
    fi
  fi

  if $ok; then
    if grep -q "bytes" "$client_log" 2>/dev/null; then
      local bw
      bw=$(grep "bytes" "$client_log" | tail -1)
      write_status "$status_file" "PASS" "$bw"
      echo "[PASS]  ${pair_tag}: ${bw}"
      return 0
    fi

    write_status "$status_file" "FAIL" "no_result_line (server_log=${server_log}, client_log=${client_log})"
    echo "[FAIL]  ${pair_tag}: no result line"
    return 1
  fi

  kill "$server_pid" 2>/dev/null || true
  kill "$client_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
  wait "$client_pid" 2>/dev/null || true

  write_status "$status_file" "FAIL" "${reason} (server_log=${server_log}, client_log=${client_log})"
  echo "[FAIL]  ${pair_tag}: ${reason}"
  return 1
}

# ---------------------------------------------------------------------------
# Launch all relay pairs in parallel
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "VF1 Cross-Island Relay Ping Prep — ${#VF1_CROSS_PAIRS[@]} pairs"
echo "================================================================"
echo "Launching all pairs in parallel..."

tcp_port=$BASE_TCP_PORT
for pair in "${VF1_CROSS_PAIRS[@]}"; do
  server="${pair%%:*}"
  client="${pair##*:}"
  (( tcp_port++ ))

  pair_tag="vf1_cross_${server}_${client}"
  run_cross_island_pingpong "$server" "$client" "$tcp_port" &
  PAIR_PIDS+=("$!")
  PAIR_TAGS+=("$pair_tag")
done

for pid in "${PAIR_PIDS[@]}"; do
  wait "$pid" || true
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
TOTAL=${#VF1_CROSS_PAIRS[@]}
PASSED=0
FAILED=0
FAILED_PAIRS=()

echo ""
echo "================================================================"
echo "SUMMARY"
echo "================================================================"
echo "  Total relay pairs  : $TOTAL"

echo ""
for pair_tag in "${PAIR_TAGS[@]}"; do
  status_file="${STATUS_DIR}/${pair_tag}.status"
  if [[ ! -f "$status_file" ]]; then
    (( FAILED++ ))
    FAILED_PAIRS+=("${pair_tag}:missing_status")
    echo "  [FAIL] ${pair_tag} -> missing status file"
    continue
  fi

  status=$(grep '^status=' "$status_file" | cut -d= -f2-)
  detail=$(grep '^detail=' "$status_file" | cut -d= -f2-)

  if [[ "$status" == "PASS" ]]; then
    (( PASSED++ ))
    echo "  [PASS] ${pair_tag} -> ${detail}"
  else
    (( FAILED++ ))
    FAILED_PAIRS+=("${pair_tag}:${detail}")
    echo "  [FAIL] ${pair_tag} -> ${detail}"
  fi
done

echo ""
echo "  Passed             : $PASSED"
echo "  Failed             : $FAILED"
echo "  Logs               : $LOG_DIR"

if (( FAILED > 0 )); then
  echo ""
  echo "  Failed pairs:"
  for fp in "${FAILED_PAIRS[@]}"; do
    echo "    - $fp"
  done
  echo ""
  exit 1
fi

echo ""
echo "  All relay pairs PASSED ✓"
echo ""
exit 0
