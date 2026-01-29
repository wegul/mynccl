# MPIB Transport Agent

A per-host daemon that enables adaptive multi-path data splitting for the MPIB NetPlugin.

## Overview

The Transport Agent manages split hints for the MPIB plugin via shared memory, allowing dynamic traffic distribution between SOUT (scaleout) and SUP (scaleup) network paths.

## Quick Start

### Build

```bash
# Build the agent
cd ext-net/transport_agent
make

# Build MPIB plugin (with agent integration)
cd ext-net/mpib
make
```

### Run

**1. Start the agent on each node:**

```bash
# Manual mode (default): read commands from stdin
./bin/mpib_agent --manual

# Or alternating mode for testing:
./bin/mpib_agent --manual  # Manual control via stdin
```

**2. Run NCCL with MPIB plugin:**

```bash
export NCCL_NET_PLUGIN=mpib
export NCCL_NET=mpib
export MPIB_HCA_SOUT=mlx5_3    # Scaleout NIC
export MPIB_HCA_SUP=mlx5_4     # Scaleup NIC  
export MPIB_OOB_IF=eth0        # Out-of-band interface

# Run nccl-tests
mpirun -np 2 -H node1,node2 \
  -x NCCL_NET_PLUGIN -x NCCL_NET \
  -x MPIB_HCA_SOUT -x MPIB_HCA_SUP -x MPIB_OOB_IF \
  all_reduce_perf -b 1G -e 1G -n 100
```

## Agent Mode

| Mode | Command | Description |
|------|---------|-------------|
| Manual | `--manual` | Read "slot bw" lines interactively |

### Examples

```bash
# Interactive control
./bin/mpib_agent --manual
> all 1     # Set all slots to bw=1 (50/50)
> 0 2       # Set slot 0 to bw=2 (SUP ratio=2/3)
```

## Architecture

```
┌─────────────────┐         ┌─────────────────────────┐
│  NCCL Process   │         │    Transport Agent      │
│  ┌───────────┐  │  reg/   │  ┌───────────────────┐  │
│  │   MPIB    │──┼─dereg──▶│  │  Registration     │  │
│  │  Plugin   │  │  (IPC)  │  │  Handler          │  │
│  └─────┬─────┘  │         │  └─────────┬─────────┘  │
│        │        │         │            │            │
│   read │        │         │  ┌─────────▼─────────┐  │
│  hints │        │         │  │  Hint Writer      │  │
│        │        │         │  │  (policy-driven)  │  │
│        ▼        │         │  └─────────┬─────────┘  │
│  ┌───────────┐  │         │            │            │
│  │  Shared   │◀─┼─────────┼────────────┘            │
│  │  Memory   │  │  mmap   │                         │
│  └───────────┘  │         │                         │
└─────────────────┘         └─────────────────────────┘
```

## Interface Details

### Shared Memory (`/tmp/mpib/hints`)

- **Header**: magic (0x4D504948), max_entries (256), aligned to 16 bytes
- **Entries**: 256 × {sup_bw, seq, src_ip, dst_ip}
- **Total size**: 4112 bytes

The plugin reads hints using a seqlock pattern for lock-free consistency.

Hint semantics:
- `sup_bw` is an unsigned multiplier of SOUT bandwidth.
- Plugin computes `sup_ratio = sup_bw / (1 + sup_bw)`.

### Registration Socket (`/tmp/mpib/agent.sock`)

Unix domain socket for register/deregister messages:

| Message | Fields |
|---------|--------|
| Register | conn_id, sout_src_ip, sout_dst_ip, sup_src_ip, sup_dst_ip |
| Response | status, hint_slot |
| Deregister | conn_id |

Connections with the same SOUT (src_ip, dst_ip) share a hint slot.

## Testing

```bash
# Run local test (agent + registration check)
./scripts/test_phase1.sh

# Monitor traffic distribution
watch -n1 'ethtool -S mlx5_3 | grep tx_bytes; ethtool -S mlx5_4 | grep tx_bytes'
```

## Files

```
ext-net/transport_agent/
├── doc/
│   ├── agent_design.md       # Full design document
│   ├── phase1_plan.md        # Implementation plan
│   └── phase1_progress.md    # Implementation log
├── include/
│   └── mpib_agent_iface.h    # Shared interface definitions
├── src/
│   └── agent.cc              # Agent implementation
├── scripts/
│   └── test_phase1.sh        # Test script
├── Makefile
└── README.md                 # This file
```

## Troubleshooting

### "Failed to open hint SHM"

The agent must be running before starting NCCL:
```bash
./bin/mpib_agent --manual &
```

### "Agent registration failed"

Check if the socket exists:
```bash
ls -la /tmp/mpib/agent.sock
```

### Traffic not splitting

1. Verify hint value: `xxd /tmp/mpib/hints | head`
2. Check MPIB logs: `NCCL_DEBUG=TRACE ./your_app 2>&1 | grep hint_slot`
