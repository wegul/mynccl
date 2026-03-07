#!/usr/bin/env bash
set -euo pipefail

[[ $# -gt 0 ]] || { echo "Usage: $0 <binary> [args...]" >&2; exit 2; }

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HOST_MAP=${HOST_MAP:-"${SCRIPT_DIR}/host_map.conf"}
host=$(hostname -s)

[[ -f "${HOST_MAP}" ]] || { echo "ERROR: host map not found: ${HOST_MAP}" >&2; exit 2; }

# host_map.conf format (whitespace separated):
# host vf0_if vf1_if hca_sout hca_sup cpu_set
entry=$(awk -v h="${host}" '!/^[[:space:]]*#/ && NF && $1==h {print; exit}' "${HOST_MAP}")
[[ -n "${entry}" ]] || { echo "ERROR: no host-map entry for ${host} in ${HOST_MAP}" >&2; exit 2; }

read -r map_host vf0_if vf1_if hca_sout hca_sup cpu_core <<< "${entry}"
[[ -n "${vf0_if}" && -n "${vf1_if}" && -n "${hca_sout}" && -n "${hca_sup}" && -n "${cpu_core}" ]] || {
  echo "ERROR: bad host-map row for ${host}: ${entry}" >&2
  exit 2
}

export MPIB_OOB_IF="${MPIB_OOB_IF:-${vf0_if}}"
export MPIB_HCA_SOUT="${MPIB_HCA_SOUT:-${hca_sout}}"
export MPIB_HCA_SUP="${MPIB_HCA_SUP:-${hca_sup}}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-${MPIB_HCA_SOUT}}"
export UCX_NET_DEVICES="${UCX_NET_DEVICES:-${MPIB_OOB_IF}}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${MPIB_OOB_IF}}"

# Pin each rank to the configured CPU set from the host map.
# Examples: "16" or "16,17" or "16-17".
if [[ "${MPIB_CPU_PIN:-1}" == "1" ]] && command -v taskset >/dev/null 2>&1; then
  taskset -pc "${cpu_core}" $$ >/dev/null 2>&1 || true
fi

if [[ "${WRAPPER_VERBOSE:-0}" == "1" ]]; then
  echo "[${host}] vf0=${vf0_if}->${MPIB_HCA_SOUT} vf1=${vf1_if}->${MPIB_HCA_SUP} cpu=${cpu_core} NCCL_IB_HCA=${NCCL_IB_HCA}" >&2
fi

exec "$@"
