#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MPIB_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
NCCL_TESTS_BIN=${NCCL_TESTS_BIN:-/home/suweigao/benchmark_utils/nccl-tests/build/all_reduce_perf}
[[ -x "${NCCL_TESTS_BIN}" ]] || { echo "ERROR: NCCL_TESTS_BIN not found/executable: ${NCCL_TESTS_BIN}" >&2; exit 2; }

NP=${NP:-2}
N_PER_NODE=${N_PER_NODE:-1}
HOSTS=${HOSTS:-syrax49,taco}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage:
  launch_nccl.sh                 # one-shot run

Env vars:
  NP, N_PER_NODE, HOSTS, NCCL_TESTS_BIN (or BINARY), NCCL_LIB_DIR,
  MPIB_HCA_SOUT, MPIB_HCA_SUP, MPIB_OOB_IF, MPIB_IB_GID_INDEX,
  NCCL_DEBUG, NCCL_DEBUG_SUBSYS, OMPI_OOB_IF_INCLUDE, OMPI_BTL_IF_INCLUDE,
  GDR_MODE(auto|on|off), CUDA_VISIBLE_DEVICES (default 3 = GPU3, PIX-local to mlx5_6 SOUT),
  MPIRUN_BINDING (default '--bind-to numa --map-by ppr:1:node'),
  NCCL_IB_QPS_PER_CONNECTION, NCCL_NET_GDR_READ,
  NCCL_MIN_NCHANNELS, NCCL_MAX_NCHANNELS
EOF
  exit 0
fi

# NCCL build lib dir in this repo; must exist at the same path on all nodes.
NCCL_LIB_DIR=${NCCL_LIB_DIR:-/home/suweigao/mynccl/build/lib}

# Debug
NCCL_DEBUG=${NCCL_DEBUG:-INFO}
NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,NET,GRAPH}

# MPIB dual-rail device selection (all required)
MPIB_HCA_SOUT=${MPIB_HCA_SOUT:-mlx5_2}  # Scaleout NIC
MPIB_HCA_SUP=${MPIB_HCA_SUP:-mlx5_1}    # Scaleup NIC
MPIB_OOB_IF=${MPIB_OOB_IF:-ens28f0np0}   # OOB TCP interface (reuse SOUT NIC)
MPIB_IB_GID_INDEX=${MPIB_IB_GID_INDEX:-3}
MPIB_MODE=${MPIB_MODE:-0}  # 0=vanilla (strict path isolation), 1=advanced (agent-driven)
GDR_MODE=${GDR_MODE:-auto} # auto|on|off

# GPU selection: GPU3 is PIX-local to mlx5_6 (SOUT) on this host.
# mlx5_6 → GPU3 (PIX), GPU2 (NODE)  |  mlx5_1 → GPU0 (PIX), GPU1 (NODE)
# Override if your SOUT NIC or target node topology differs.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

# Open MPI TCP control/data interfaces. Keep these aligned with MPIB_OOB_IF
# so Open MPI does not accept peer connections on unexpected NICs.
OMPI_OOB_IF_INCLUDE=${OMPI_OOB_IF_INCLUDE:-${MPIB_OOB_IF}}
OMPI_BTL_IF_INCLUDE=${OMPI_BTL_IF_INCLUDE:-${MPIB_OOB_IF}}

UCX_NET_DEVICES=${MPIB_OOB_IF}

NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-${MPIB_OOB_IF}}

# Build the runtime env we will export to ranks via mpirun.
LD_LIBRARY_PATH_LAUNCH="${MPIB_DIR}:${NCCL_LIB_DIR}:${LD_LIBRARY_PATH:-}"

MPIRUN_BASE=(
  mpirun
  --mca oob_tcp_if_include "${OMPI_OOB_IF_INCLUDE}"
  --mca btl_tcp_if_include "${OMPI_BTL_IF_INCLUDE}"
  -x "LD_LIBRARY_PATH=${LD_LIBRARY_PATH_LAUNCH}"
  -x "NCCL_DEBUG=${NCCL_DEBUG}"
  -x "NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS}"
  -x "MPIB_HCA_SOUT=${MPIB_HCA_SOUT}"
  -x "MPIB_HCA_SUP=${MPIB_HCA_SUP}"
  -x "MPIB_OOB_IF=${MPIB_OOB_IF}"
  -x "MPIB_IB_GID_INDEX=${MPIB_IB_GID_INDEX}"
  -x "MPIB_MODE=${MPIB_MODE}"
  -x "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  -x "NCCL_IB_HCA=${MPIB_HCA_SOUT}"
  -x "UCX_NET_DEVICES=${UCX_NET_DEVICES}"
  # -x "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
)

case "${GDR_MODE}" in
  auto)
    ;;
  on)
    # Force-enable GDR path selection up to SYS distance.
    MPIRUN_BASE+=( -x "NCCL_NET_GDR_LEVEL=SYS" )
    ;;
  off)
    # Disable GDR for clean A/B comparison.
    MPIRUN_BASE+=( -x "NCCL_NET_GDR_LEVEL=0" )
    ;;
  *)
    echo "ERROR: Invalid GDR_MODE='${GDR_MODE}'. Expected: auto|on|off" >&2
    exit 2
    ;;
esac

# CPU binding policy. -N is already translated to ppr:N:node internally by
# this version of Open MPI, so any explicit --map-by causes a "too many
# directives" conflict. Only use --bind-to here.
#   MPIRUN_BINDING="--bind-to core"   # physical cores only (no HT)
#   MPIRUN_BINDING=""                 # disable binding entirely
MPIRUN_BINDING=${MPIRUN_BINDING:-"--bind-to numa"}
if [[ -n "${MPIRUN_BINDING:-}" ]]; then
  # shellcheck disable=SC2206
  _mpirun_binding=( ${MPIRUN_BINDING} )
  MPIRUN_BASE+=( "${_mpirun_binding[@]}" )
fi

# Optional tuning env exports used in deeper A/B sweeps.
if [[ -n "${NCCL_IB_QPS_PER_CONNECTION:-}" ]]; then
  MPIRUN_BASE+=( -x "NCCL_IB_QPS_PER_CONNECTION=${NCCL_IB_QPS_PER_CONNECTION}" )
fi
if [[ -n "${NCCL_NET_GDR_READ:-}" ]]; then
  MPIRUN_BASE+=( -x "NCCL_NET_GDR_READ=${NCCL_NET_GDR_READ}" )
fi
if [[ -n "${NCCL_MIN_NCHANNELS:-}" ]]; then
  MPIRUN_BASE+=( -x "NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS}" )
fi
if [[ -n "${NCCL_MAX_NCHANNELS:-}" ]]; then
  MPIRUN_BASE+=( -x "NCCL_MAX_NCHANNELS=${NCCL_MAX_NCHANNELS}" )
fi

exec "${MPIRUN_BASE[@]}" -H "${HOSTS}" -np "${NP}" -N "${N_PER_NODE}" \
  "${NCCL_TESTS_BIN}" -b 8 -e 1G -f 2 -g 1 -c 1