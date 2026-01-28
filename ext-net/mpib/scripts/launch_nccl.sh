#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MPIB_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
MPIB_SO=${MPIB_SO:-"${MPIB_DIR}/libnccl-net-mpib.so"}
NCCL_TESTS_BIN=${NCCL_TESTS_BIN:-/home/suweigao/benchmark_utils/nccl-tests/build/all_reduce_perf}
[[ -x "${NCCL_TESTS_BIN}" ]] || { echo "ERROR: NCCL_TESTS_BIN not found/executable: ${NCCL_TESTS_BIN}" >&2; exit 2; }

NP=2
N_PER_NODE=1

# NCCL build lib dir in this repo; must exist at the same path on all nodes.
NCCL_LIB_DIR=${NCCL_LIB_DIR:-/home/suweigao/mynccl/build/lib}

# Debug
NCCL_DEBUG=${NCCL_DEBUG:-INFO}
NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,NET}

# MPIB dual-rail device selection (all required)
MPIB_HCA_SOUT=${MPIB_HCA_SOUT:-mlx5_3}  # Scaleout NIC
MPIB_HCA_SUP=${MPIB_HCA_SUP:-mlx5_4}    # Scaleup NIC
MPIB_OOB_IF=${MPIB_OOB_IF:-ens1f0np0}   # OOB TCP interface (independent of RDMA)
MPIB_IB_GID_INDEX=${MPIB_IB_GID_INDEX:-3}

UCX_NET_DEVICES=${MPIB_OOB_IF}

NCCL_NET_PLUGIN=mpib

# Build the runtime env we will export to ranks via mpirun.
LD_LIBRARY_PATH_LAUNCH="${MPIB_DIR}:${NCCL_LIB_DIR}:${LD_LIBRARY_PATH:-}"

MPIRUN=(
  mpirun -H "accord1,accord3,accord2,accord4" -np "${NP}" -N "${N_PER_NODE}"
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
)

exec "${MPIRUN[@]}" "${NCCL_TESTS_BIN}" -b 8 -e 1G -f 2 -g 1 -c 1