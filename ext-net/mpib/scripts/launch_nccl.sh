#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MPIB_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
MPIB_SO=${MPIB_SO:-"${MPIB_DIR}/libnccl-net-mpib.so"}
NCCL_TESTS_BIN=${NCCL_TESTS_BIN:-/home/suweigao/benchmark_utils/nccl-tests/build/all_reduce_perf}
[[ -x "${NCCL_TESTS_BIN}" ]] || { echo "ERROR: NCCL_TESTS_BIN not found/executable: ${NCCL_TESTS_BIN}" >&2; exit 2; }

NP=4
N_PER_NODE=1
HOSTS=${HOSTS:-vm1,vm2,vm3,vm4,vm5,vm6,vm7,vm8}

MODE=${1:-}
if [[ "${MODE}" == "--help" || "${MODE}" == "-h" ]]; then
  cat <<'EOF'
Usage:
  launch_nccl.sh                 # one-shot run (default)
  launch_nccl.sh -p              # continuous runs for manual policy testing

Env vars:
  NP, N_PER_NODE, HOSTS, NCCL_TESTS_BIN (or BINARY), NCCL_LIB_DIR,
  MPIB_HCA_SOUT, MPIB_HCA_SUP, MPIB_OOB_IF, MPIB_IB_GID_INDEX,
  NCCL_DEBUG, NCCL_DEBUG_SUBSYS
EOF
  exit 0
fi

# NCCL build lib dir in this repo; must exist at the same path on all nodes.
NCCL_LIB_DIR=${NCCL_LIB_DIR:-/home/suweigao/mynccl/build/lib}

# Debug
NCCL_DEBUG=${NCCL_DEBUG:-INFO}
NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,NET}

# MPIB dual-rail device selection (all required)
MPIB_HCA_SOUT=${MPIB_HCA_SOUT:-mlx5_0}  # Scaleout NIC
MPIB_HCA_SUP=${MPIB_HCA_SUP:-mlx5_1}    # Scaleup NIC
MPIB_OOB_IF=${MPIB_OOB_IF:-enp8s0np0}   # OOB TCP interface (reuse SOUT NIC)
MPIB_IB_GID_INDEX=${MPIB_IB_GID_INDEX:-3}

UCX_NET_DEVICES=${MPIB_OOB_IF}

NCCL_NET_PLUGIN=mpib
NCCL_SOCKET_IFNAME=enp8s0np0

# Build the runtime env we will export to ranks via mpirun.
LD_LIBRARY_PATH_LAUNCH="${MPIB_DIR}:${NCCL_LIB_DIR}:${LD_LIBRARY_PATH:-}"

MPIRUN_BASE=(
  mpirun
  -x "LD_LIBRARY_PATH=${LD_LIBRARY_PATH_LAUNCH}"
  -x "NCCL_DEBUG=${NCCL_DEBUG}"
  -x "NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS}"
  -x "MPIB_HCA_SOUT=${MPIB_HCA_SOUT}"
  -x "MPIB_HCA_SUP=${MPIB_HCA_SUP}"
  -x "MPIB_OOB_IF=${MPIB_OOB_IF}"
  -x "MPIB_IB_GID_INDEX=${MPIB_IB_GID_INDEX}"
  -x "NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN}"
  -x "NCCL_IB_HCA=${MPIB_HCA_SUP}"
  -x "UCX_NET_DEVICES=${UCX_NET_DEVICES}"
  -x "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
)

if [[ "${MODE}" == "-p" ]]; then
  SIZE=${SIZE:-$((1024*1024*1024))}
  WARMUP=${WARMUP:-0}
  PERSISTENT_NCCL_DEBUG=${PERSISTENT_NCCL_DEBUG:-VERSION}
  PERSISTENT_NCCL_DEBUG_SUBSYS=${PERSISTENT_NCCL_DEBUG_SUBSYS:-}

  while true; do
    "${MPIRUN_BASE[@]}" \
      -x "NCCL_DEBUG=${PERSISTENT_NCCL_DEBUG}" \
      -x "NCCL_DEBUG_SUBSYS=${PERSISTENT_NCCL_DEBUG_SUBSYS}" \
      -H "${HOSTS}" -np "${NP}" \
      "${NCCL_TESTS_BIN}" -b "${SIZE}" -e "${SIZE}" -f 1 -w "${WARMUP}" -n 1 \
      2>/dev/null | awk '/^[[:space:]]*[0-9]+[[:space:]]/ {print; fflush();}' || break
  done
  exit 0
fi

# Default one-shot mode
exec "${MPIRUN_BASE[@]}" -H "${HOSTS}" -np "${NP}" -N "${N_PER_NODE}" \
  "${NCCL_TESTS_BIN}" -b 8 -e 1G -f 2 -g 1 -c 1