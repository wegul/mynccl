# Transport Agent Design Document

## 1. Executive Summary

The Transport Agent is a per-host daemon that enables adaptive multi-path data splitting for the MPIB NetPlugin. It bridges three information domains:

1. **NCCL Plugin** — Registers connections, consumes split hints
2. **Switch Telemetry** — Receives per-path congestion signals via UDP
3. **Local NIC Monitoring** — Reports intra-host congestion to the network

**Key Design Principles:**

- **Decoupled policy**: The agent owns the "how much traffic on each path" decision; the plugin is a stateless executor.
- **Low-latency adaptation**: Hints update at millisecond granularity; backpressure signals trigger immediate changes.
- **Minimal plugin overhead**: Shared-memory interface avoids IPC syscalls on the hot path.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                   HOST                                      │
│  ┌─────────────────┐         ┌─────────────────────────────────────────┐   │
│  │  NCCL Process   │         │          Transport Agent (daemon)       │   │
│  │  ┌───────────┐  │  reg/   │  ┌─────────────┐    ┌───────────────┐  │   │
│  │  │   MPIB    │──┼─dereg──▶│  │ Connection  │    │  Policy       │  │   │
│  │  │  Plugin   │  │  (IPC)  │  │  Registry   │───▶│  Engine       │  │   │
│  │  └─────┬─────┘  │         │  └─────────────┘    └───────┬───────┘  │   │
│  │        │        │         │                             │          │   │
│  │   read │        │         │  ┌─────────────┐    ┌───────▼───────┐  │   │
│  │  hints │        │         │  │  NIC        │───▶│  Hint         │  │   │
│  │        │        │         │  │  Monitor    │    │  Writer       │  │   │
│  │        ▼        │         │  └─────────────┘    └───────┬───────┘  │   │
│  │  ┌───────────┐  │         │                             │          │   │
│  │  │  Shared   │◀─┼─────────┼─────────────────────────────┘          │   │
│  │  │  Memory   │  │  mmap   │                                        │   │
│  │  └───────────┘  │         │  ┌─────────────┐    ┌───────────────┐  │   │
│  └─────────────────┘         │  │  Telemetry  │◀───│  UDP Recv     │  │   │
│                              │  │  Processor  │    │  (from switch)│  │   │
│                              │  └──────┬──────┘    └───────────────┘  │   │
│                              │         │                              │   │
│                              │  ┌──────▼──────┐    ┌───────────────┐  │   │
│                              │  │  Congestion │───▶│  UDP Send     │  │   │
│                              │  │  Reporter   │    │  (to switch)  │  │   │
│                              │  └─────────────┘    └───────────────┘  │   │
│                              └─────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                          ──────────────┼──────────────
                                        │ UDP
                                        ▼
                              ┌─────────────────┐
                              │     Switch      │
                              │   (Telemetry)   │
                              └─────────────────┘
```

### 2.1 Components

| Component | Description |
|-----------|-------------|
| **Connection Registry** | Tracks active MPIB connections (src/dst IP, hostname) |
| **Policy Engine** | Computes split hints from telemetry + local congestion |
| **Hint Writer** | Updates shared memory with latest split hints |
| **NIC Monitor** | Polls local NIC queue depths (interface TBD) |
| **Telemetry Processor** | Parses incoming switch UDP packets |
| **Congestion Reporter** | Sends local congestion info to switch |

### 2.2 Deployment Model

- **One agent per host**, manually started before NCCL jobs
- Serves all NCCL processes on the host
- Agent provides an IPC endpoint for registration and a UDP port for telemetry

---

## 3. Communication Interfaces

### 3.1 Plugin ↔ Agent: Registration (IPC)

The plugin registers each connection at setup time and deregisters at teardown.

**IPC Options (TBD):**
- Unix domain socket (simple request-response)
- Shared memory ring buffer (lower latency, more complex)
- Named pipe (simple, unidirectional)

For initial implementation, Unix socket is likely simplest. Can revisit if registration overhead matters.

**Registration Data (conceptual):**
- Connection ID (unique identifier from plugin)
- Source IP address
- Destination IP address
- Source hostname (if easily available)
- Destination hostname (exchanged in MPIB metadata)
- Number of devices (2 for SOUT+SUP)

**Registration Response:**
- Status (success/failure)
- Hint slot index (where plugin should read hints from shared memory)

### 3.2 Plugin ↔ Agent: Hint Consumption (Shared Memory)

The plugin reads split hints from shared memory on every `isend()`. No syscall overhead.

**Shared Memory Layout (conceptual):**
- Magic number for validation
- Array of hint entries, indexed by slot
- Each entry contains:
     - SUP bandwidth multiplier `sup_bw` (unsigned integer, multiple of SOUT bandwidth)
  - Version counter (for consistency checking)
     - Flags (e.g., backpressure active)

**Read Pattern:** Lock-free with version checking (seqlock style) to avoid tearing.

**Performance Considerations:**
- Reading on every `isend()` adds minimal overhead (~10ns for atomic loads)
- If profiling shows issues, can cache hint for N requests
- Relaxed atomics may be acceptable since the hint is a single word

**Hint Semantics:**

The hint is an unsigned multiplier of SOUT bandwidth:
- `sup_bw * sout_bw`
- Plugin converts it to a byte split ratio:

$$
	ext{sup\_ratio} = \frac{\text{sup\_bw}}{1 + \text{sup\_bw}}
$$

Examples: `sup_bw=0` → 0% SUP, `sup_bw=1` → 50% SUP, `sup_bw=2` → 66.7% SUP.

### 3.3 Agent ↔ Switch: Telemetry (UDP)

#### 3.3.1 Incoming: Path Congestion from Switch

Switch sends congestion info for paths terminating at this host.

**Data (conceptual):**
- Source IP (sender)
- Destination IP (this host)
- Path ID (SOUT=0, SUP=1)
- Congestion level (0-255 scale)
- Queue depth (optional, more granular)
- Timestamp

#### 3.3.2 Outgoing: Local Congestion to Switch

Agent periodically reports its NIC queue depths so the switch can inform remote senders.

**Data (conceptual):**
- Host IP
- Per-NIC stats:
  - NIC ID (SOUT=0, SUP=1)
  - TX queue depth
  - RX queue depth (backpressure indicator)

---

## 4. Workflows

### 4.1 Connection Setup (Registration)

```
NCCL Process                    Transport Agent                    Switch
     │                                │                               │
     │ [MPIB connect() completes]     │                               │
     │                                │                               │
     │──── register_req ─────────────▶│                               │
     │     {src_ip, dst_ip,           │                               │
     │      hostnames, conn_id}       │                               │
     │                                │                               │
     │                                │── UDP "new flow" ────────────▶│
     │                                │   {src, dst, path_ids}        │
     │                                │                               │
     │                                │   (switch adds to monitor)    │
     │                                │                               │
     │◀─── register_resp ─────────────│                               │
     │     {hint_slot}                │                               │
     │                                │                               │
     │ [store hint_slot in comm]      │                               │
```

### 4.2 Steady State (Data Transfer)

```
NCCL Process                    Transport Agent                    Switch
     │                                │                               │
     │ [isend() called]               │                               │
     │                                │                               │
     │◀─── read shm[hint_slot] ───────│                               │
     │     (lock-free, no IPC)        │                               │
     │                                │                               │
     │ [apply sup_ratio to split]     │                               │
     │ [post WRs to SOUT/SUP]         │                               │
     │                                │                               │
     │                                │◀── UDP telemetry ─────────────│
     │                                │    {path congestion levels}   │
     │                                │                               │
     │                                │ [policy engine computes       │
     │                                │  new sup_bw per conn]         │
     │                                │                               │
     │◀─── update shm[hint_slot] ─────│                               │
     │     (atomic write)             │                               │
     │                                │                               │
     │                                │── UDP report ─────────────────▶│
     │                                │   {local NIC queue depths}    │
```

### 4.3 Backpressure Event (Async)

When switch detects sudden congestion:

```
NCCL Process                    Transport Agent                    Switch
     │                                │                               │
     │                                │◀── UDP telemetry ─────────────│
     │                                │    {path=SUP, congestion=HIGH}│
     │                                │                               │
     │                                │ [policy: sup_bw = 0]          │
     │                                │                               │
     │◀─── update shm (immediate) ────│                               │
     │                                │                               │
     │ [next isend() sees ratio=0]    │                               │
```

### 4.4 Connection Teardown (Deregistration)

```
NCCL Process                    Transport Agent                    Switch
     │                                │                               │
     │ [MPIB closeRecv/closeSend]     │                               │
     │                                │                               │
     │──── deregister_req ───────────▶│                               │
     │     {conn_id}                  │                               │
     │                                │                               │
     │                                │── UDP "flow removed" ────────▶│
     │                                │                               │
     │                                │ [clear shm entry]             │
     │                                │                               │
     │◀─── deregister_resp ──────────│                               │
```

---

## 5. Policy Engine

### 5.1 Inputs

| Source | Data | Update Frequency |
|--------|------|------------------|
| Switch telemetry | Per-path congestion level (0-255) | ~1ms (configurable) |
| Local NIC monitor | TX/RX queue depths | ~1ms |
| Connection registry | Active flows (src/dst pairs) | On registration |

### 5.2 Policy Algorithm (Initial)

Simple proportional control based on congestion differential:

- If SUP path is more congested than SOUT, reduce SUP share (decrease `sup_bw`)
- If SOUT path is more congested, increase SUP share (increase `sup_bw`)
- If both are equally congested (or clear), use 50/50 split
- Apply EWMA smoothing to avoid oscillation

### 5.3 Backpressure Override

When congestion level exceeds a threshold (e.g., 240/255):
- Immediately set the congested path's share to 0 (set `sup_bw=0`)
- Set a "backpressure active" flag in the hint entry
- Resume normal policy when congestion drops below threshold

### 5.4 Future Enhancements

- **Per-flow fairness**: Consider other flows sharing the path
- **Bandwidth estimation**: Use throughput measurements, not just queue depth
- **Predictive control**: Anticipate congestion from trends

---

## 6. Intra-Host NIC Monitoring

### 6.1 What We Need

The agent needs to report "how busy is the TX/RX path" for each NIC (SOUT, SUP). This information is sent to the switch so it can inform remote senders.

### 6.2 Potential Data Sources (TBD)

| Approach | Pros | Cons |
|----------|------|------|
| **ethtool -S counters** | Standard, widely available | May not expose queue depth directly |
| **mlx5 devlink** | Mellanox-specific detailed stats | Vendor lock-in |
| **PFC pause frame counters** | Direct backpressure indicator | Only shows when already congested |
| **TX/RX byte rate delta** | Simple load estimation | Indirect, doesn't show queuing |
| **eBPF tracepoints** | Flexible, kernel-level | Requires BPF, more complex |

### 6.3 Initial Approach

Start with what's easily available:
1. Poll `ethtool -S <nic>` for tx/rx counters
2. Look for `rx_pause` / `tx_pause` (PFC indicators)
3. If queue depth counters exist (`tx_queue_*`), use those

Leave this as a pluggable interface so we can swap implementations later.

---

## 7. Open Questions

1. **Hostname resolution**: Should the plugin resolve remote hostname via DNS, or exchange it in MPIB connection metadata? (Metadata exchange is simpler.)

2. **Agent failure handling**: If agent is unavailable at registration time, should plugin:
   - (a) Fail the connection?
   - (b) Fall back to static 50/50 split?
   - (c) Retry with backoff?

3. **IPC mechanism**: Unix socket is simplest for request-response. Is there a case for shared-memory ring buffer for registration too? (Probably overkill.)

4. **Switch protocol**: Exact packet formats TBD when switch telemetry is developed. Current design is placeholder.

5. **Multiple NCCL jobs**: If multiple jobs share the host, their connections share the hint pool. conn_id disambiguates. Is this sufficient?

---

## 8. Implementation Phases

### Phase 1: Hint Reading + Registration

**Goal:** Complete local data path — MPIB registers with agent, reads hints from shared memory, agent writes hints.

**Agent:**
- Create `/tmp/mpib/` directory and shared memory file
- Listen on Unix socket for registration
- Handle register/deregister requests, assign hint slots
- Maintain connection registry with slot sharing (same src-dst → same slot)
- Write hints (static or simulated policy for testing)

**Plugin:**
- On init: mmap the shared memory file (fail if unavailable)
- On connect: send registration request, store returned hint_slot
- On isend: read hint from assigned slot, compute ratio, apply to split
- On close: send deregistration request

**Validation:**
- Agent starts, creates SHM
- MPIB connects, registers, gets slot
- Multiple connections to same peer share slot
- Traffic shifts when agent changes hints
- Deregistration works

**Details:** See [phase1_plan.md](phase1_plan.md)

### Phase 2: Telemetry & Dynamic Hints

**Goal:** Agent receives switch telemetry and updates hints accordingly.

**Agent:**
- Add UDP listener for incoming telemetry
- Implement policy engine (simple proportional control)
- Update hints based on congestion levels

**Validation:**
- Inject mock telemetry packets
- Observe hint changes
- Verify MPIB traffic shifts in response

### Phase 3: Local Monitoring & Reporting

**Goal:** Agent monitors local NIC congestion and reports to switch.

**Agent:**
- Implement NIC monitor (ethtool-based initially)
- Add UDP reporter to send local stats
- Integrate local congestion into policy

**Validation:**
- Agent sends reports to switch (or mock receiver)
- Reports contain accurate NIC stats

---

## 9. Directory Structure

```
ext-net/transport_agent/
├── doc/
│   └── agent_design.md          # This document
├── src/
│   └── (implementation TBD)
├── include/
│   └── (headers TBD)
└── Makefile
```
