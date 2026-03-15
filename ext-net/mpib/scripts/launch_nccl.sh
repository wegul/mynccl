#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MPIB_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
RANK_WRAPPER=${RANK_WRAPPER:-"${SCRIPT_DIR}/rank_wrapper.sh"}
HOSTFILE=${HOSTFILE:-"${SCRIPT_DIR}/hostfile"}
HOST_MAP=${HOST_MAP:-"${SCRIPT_DIR}/host_map.conf"}
NCCL_TESTS_BIN=${NCCL_TESTS_BIN:-/home/suweigao/benchmark_utils/nccl-tests/build/all_reduce_perf}
[[ -x "${NCCL_TESTS_BIN}" ]] || { echo "ERROR: NCCL_TESTS_BIN not found/executable: ${NCCL_TESTS_BIN}" >&2; exit 2; }
[[ -f "${RANK_WRAPPER}" ]] || { echo "ERROR: rank wrapper not found: ${RANK_WRAPPER}" >&2; exit 2; }
[[ -f "${HOSTFILE}" ]] || { echo "ERROR: hostfile not found: ${HOSTFILE}" >&2; exit 2; }
[[ -f "${HOST_MAP}" ]] || { echo "ERROR: host map not found: ${HOST_MAP}" >&2; exit 2; }

NP=${NP:-8}
N_PER_NODE=${N_PER_NODE:-1}

# Keep Open MPI from forcing all ranks onto CPU0 in constrained containers.
MPI_BIND_TO=${MPI_BIND_TO:-none}
MPI_MAP_BY=${MPI_MAP_BY:-slot}
MPI_REPORT_BINDINGS=${MPI_REPORT_BINDINGS:-0}

# NCCL build lib dir in this repo; must exist at the same path on all nodes.
NCCL_LIB_DIR=${NCCL_LIB_DIR:-/home/suweigao/mynccl/build/lib}
# Build the runtime env we will export to ranks via mpirun.
LD_LIBRARY_PATH_LAUNCH="${MPIB_DIR}:${NCCL_LIB_DIR}:${LD_LIBRARY_PATH:-}"
# Force-disable local-node transports for cross-node-only debugging:
NCCL_GIN_TYPE=${NCCL_GIN_TYPE:-0}

# Debug
NCCL_DEBUG=${NCCL_DEBUG:-WARNING}
NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,NET}

MPIRUN_BASE=(
  mpirun
  --bind-to "${MPI_BIND_TO}"
  --map-by "${MPI_MAP_BY}"
  -x "LD_LIBRARY_PATH=${LD_LIBRARY_PATH_LAUNCH}"
  -x "HOST_MAP=${HOST_MAP}"
  -x "NCCL_DEBUG=${NCCL_DEBUG}"
  -x "NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS}"
  -x "NCCL_GIN_TYPE=${NCCL_GIN_TYPE}"

  -x "NCCL_NET_PLUGIN=mpib"
  -x "MPIB_MODE=1" # 0=vanilla, 1=advanced
  # -x "NCCL_ALGO=TREE" # NCCL cannot adjust ALGO adaptively.
)

if [[ "${MPI_REPORT_BINDINGS}" == "1" ]]; then
  MPIRUN_BASE+=(--report-bindings)
fi

# Default one-shot mode
exec "${MPIRUN_BASE[@]}" --hostfile "${HOSTFILE}" -np "${NP}" -N "${N_PER_NODE}" \
  bash "${RANK_WRAPPER}" "${NCCL_TESTS_BIN}" -b 8K -e 1G -f 2 -g 1 -c 1