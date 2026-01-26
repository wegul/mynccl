#!/usr/bin/env bash
set -euo pipefail

# Multi-node NCCL launch helper.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MPIB_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
MPIB_SO=${MPIB_SO:-"${MPIB_DIR}/libnccl-net-mpib.so"}
HOSTFILE=${HOSTFILE:-"${SCRIPT_DIR}/hostfile"}
NCCL_TESTS_BIN=${NCCL_TESTS_BIN:-/home/suweigao/benchmark_utils/nccl-tests/build/all_reduce_perf}
[[ -f "${HOSTFILE}" ]] || { echo "ERROR: hostfile not found: ${HOSTFILE}" >&2; exit 2; }
[[ -x "${NCCL_TESTS_BIN}" ]] || { echo "ERROR: NCCL_TESTS_BIN not found/executable: ${NCCL_TESTS_BIN}" >&2; exit 2; }

NP=${NP:-4}
N_PER_NODE=${N_PER_NODE:-1}


# NCCL build lib dir in this repo; must exist at the same path on all nodes.
NCCL_LIB_DIR=${NCCL_LIB_DIR:-/home/suweigao/mynccl/build/lib}
# Debug
NCCL_DEBUG=${NCCL_DEBUG:-INFO}
NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,NET}
# Device selection
NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_3}
UCX_NET_DEVICES=${UCX_NET_DEVICES:-mlx5_1:1}

NCCL_NET_PLUGIN=mpib

# Build the runtime env we will export to ranks via mpirun.
LD_LIBRARY_PATH_LAUNCH="${MPIB_DIR}:${NCCL_LIB_DIR}:${LD_LIBRARY_PATH:-}"

MPIRUN=(
  mpirun --hostfile "${HOSTFILE}" -np "${NP}" -N "${N_PER_NODE}"
  -x "LD_LIBRARY_PATH=${LD_LIBRARY_PATH_LAUNCH}"
  -x "NCCL_DEBUG=${NCCL_DEBUG}"
  -x "NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS}"
  -x "NCCL_IB_HCA=${NCCL_IB_HCA}"
  -x "UCX_NET_DEVICES=${UCX_NET_DEVICES}"
  -x "NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN}"
)

exec "${MPIRUN[@]}" "${NCCL_TESTS_BIN}" -b 4096 -e 1G -f 2 -g 1 -c 1