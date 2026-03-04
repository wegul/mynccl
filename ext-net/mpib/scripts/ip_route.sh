#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ip_route.sh -add
  ip_route.sh -del

Notes:
  - Adds/removes only mlx5_0 path routes used by launch_nccl.sh.
  - Runs on local host and peer host automatically via SSH.
  - Optional: PEER_HOST=<ssh-hostname> overrides default peer target.
EOF
}

ACTION=${1:-}
REMOTE_ONLY=${2:-}

if [[ "${ACTION}" != "-add" && "${ACTION}" != "-del" ]]; then
  usage
  exit 2
fi

apply_local() {
  local host dev remote_cidr gw peer_ip

  host=$(hostname -s)
  case "${host}" in
    syrax-49|syrax49)
      dev="ns2pf0"
      remote_cidr="10.0.2.0/24"
      gw="10.0.1.199"
      peer_ip="10.0.2.5"
      ;;
    eat-tacos|taco)
      dev="ns5pf0"
      remote_cidr="10.0.1.0/24"
      gw="10.0.2.199"
      peer_ip="10.0.1.2"
      ;;
    *)
      echo "ERROR: unsupported host '${host}'" >&2
      echo "Expected syrax-49/syrax49 or eat-tacos/taco." >&2
      exit 2
      ;;
  esac

  ip link show dev "${dev}" >/dev/null
  sudo -n true >/dev/null

  if [[ "${ACTION}" == "-add" ]]; then
    sudo ip route replace "${remote_cidr}" via "${gw}" dev "${dev}"
    echo "[${host}] Added route: ${remote_cidr} via ${gw} dev ${dev}"
    ip route show "${remote_cidr}"
    echo "[${host}] Ping check (${dev} -> ${peer_ip})"
    if ! ping -I "${dev}" -c 2 -W 1 "${peer_ip}"; then
      echo "[${host}] WARN: ping failed (route installed, but path may be missing)" >&2
    fi
  else
    sudo ip route del "${remote_cidr}" via "${gw}" dev "${dev}" 2>/dev/null || true
    echo "[${host}] Deleted route: ${remote_cidr} via ${gw} dev ${dev}"
    ip route show "${remote_cidr}" || true
  fi
}

apply_local

if [[ "${REMOTE_ONLY}" == "--remote-internal" ]]; then
  exit 0
fi

LOCAL_HOST=$(hostname -s)
case "${LOCAL_HOST}" in
  syrax-49|syrax49)
    PEER_HOST=${PEER_HOST:-taco}
    ;;
  eat-tacos|taco)
    PEER_HOST=${PEER_HOST:-syrax49}
    ;;
  *)
    echo "ERROR: unsupported host '${LOCAL_HOST}'" >&2
    exit 2
    ;;
esac

echo "Applying '${ACTION}' on peer host: ${PEER_HOST}"
ssh -o BatchMode=yes -o ConnectTimeout=8 "${PEER_HOST}" 'bash -s -- "'"${ACTION}"'" --remote-internal' <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
ACTION=${1:-}
if [[ "${ACTION}" != "-add" && "${ACTION}" != "-del" ]]; then
  exit 2
fi

host=$(hostname -s)
case "${host}" in
  syrax-49|syrax49)
    dev="ns2pf0"; remote_cidr="10.0.2.0/24"; gw="10.0.1.199"; peer_ip="10.0.2.5"
    ;;
  eat-tacos|taco)
    dev="ns5pf0"; remote_cidr="10.0.1.0/24"; gw="10.0.2.199"; peer_ip="10.0.1.2"
    ;;
  *)
    echo "ERROR: unsupported host '${host}'" >&2
    exit 2
    ;;
esac

ip link show dev "${dev}" >/dev/null
sudo -n true >/dev/null

if [[ "${ACTION}" == "-add" ]]; then
  sudo ip route replace "${remote_cidr}" via "${gw}" dev "${dev}"
  echo "[${host}] Added route: ${remote_cidr} via ${gw} dev ${dev}"
  ip route show "${remote_cidr}"
  echo "[${host}] Ping check (${dev} -> ${peer_ip})"
  if ! ping -I "${dev}" -c 2 -W 1 "${peer_ip}"; then
    echo "[${host}] WARN: ping failed (route installed, but path may be missing)" >&2
  fi
else
  sudo ip route del "${remote_cidr}" via "${gw}" dev "${dev}" 2>/dev/null || true
  echo "[${host}] Deleted route: ${remote_cidr} via ${gw} dev ${dev}"
  ip route show "${remote_cidr}" || true
fi
EOF
