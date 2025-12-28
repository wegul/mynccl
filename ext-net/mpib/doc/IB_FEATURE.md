# NCCL Net/IB Advanced Features

This document summarizes several "fancy" InfiniBand/RoCE features used by the NCCL `net_ib` transport, focusing on how they work conceptually and which knobs control them.

---

## 1. Merged Devices (Virtual NIC / Multi‑Rail)

**Goal:** Present multiple physical ports/NICs as one logical "virtual" NIC and use their combined bandwidth.

### What it does

- `net_ib` discovers multiple IB/RoCE devices/ports and can **merge** them into a single **virtual device** (see `ncclIbMergedDev` in `src/transport/net_ib/common.h`).
- The virtual device exposes:
  - `vProps.devs[]`: the list of underlying physical device indices.
  - `vProps.ndevs`: how many physical devices are in this virtual NIC.
  - `speed`: the sum of the member links' speeds.
  - `devName`: a composite name like `mlx5_0+mlx5_1`.
- When a communicator is created on such a virtual device, NCCL can use multiple underlying rails (ports/NICs) for the same logical connection, improving aggregate bandwidth and resilience.

### How it is built

- The core logic is in `ncclIbMakeVDeviceInternal` in `src/transport/net_ib/init.cc`:
  - Validates that a virtual device has at least one member and does not exceed `NCCL_IB_MAX_DEVS_PER_NIC`.
  - Sums speed and concatenates names for each underlying device.
  - Checks that all member devices share the same link layer (all IB or all RoCE). If not, merging is rejected.
- The table of merged devices is stored in `ncclIbMergedDevs[MAX_IB_VDEVS]` (`src/transport/net_ib/common.cc` / `common.h`).
- Later in connection setup (`src/transport/net_ib/connect.cc`), the send/recv comms copy `mergedDev->vProps` into their base structure so the multi‑rail layout is known on both sides.

### Configuration knobs

- **`NCCL_IB_MERGE_NICS`** (param name `IbMergeNics`):
  - When `0`, merging across multiple NICs/ports is disabled.
  - When non‑zero, `net_ib` is allowed to build virtual devices that span several physical devices.
- **`NCCL_IB_HCA`**:
  - Filters which HCAs/ports are even considered for merging (whitelist/blacklist syntax).
  - Useful to restrict merging to a subset of NICs or to force only a given rail to be used.

**Effect on behavior:** once merged devices are enabled, NCCL will create communicators where each logical `net_ib` device can internally fan out over up to `NCCL_IB_MAX_DEVS_PER_NIC` underlying rails, and the reported link speed is the aggregated speed.

---

## 2. Multi‑QP per Connection and Data Striping

**Goal:** Use multiple RDMA QPs for a single logical connection to increase throughput and hide CQ latency.

### What it does

- Each send/recv communicator maintains an array of QPs, not just one (see `ncclIbQp` and `NCCL_IB_MAX_QPS` in `src/transport/net_ib/common.h`).
- For a given send request, data can be **split across several QPs**:
  - In `ncclIbMultiSend` (`src/transport/net_ib/p2p.cc`), `nqps = ncclIbCommBaseGetNqpsPerRequest(&comm->base)` decides how many QPs to use.
  - For each `qp`, the code computes a `chunkSize` and assigns a portion of the buffer to that QP (`length` per QP).
  - As a result, a single logical Isend/Irecv can be carried over multiple HW queues in parallel.

### How it is selected

- `ncclIbCommBaseGetNqpsPerRequest` returns how many QPs are effectively used per request. The mapping is managed in the `ncclIbNetCommBase` and `ncclIbQp` structures.
- When multi‑QP is in use, the plugin also takes care to align writes (multiples of 128B) to keep lower‑latency protocols (LL/LL128) working correctly.

### Configuration knobs

- **`NCCL_IB_QPS_PER_CONNECTION`** (param `IbQpsPerConnection`):
  - Number of QPs allocated per connection (upper bounded by `NCCL_IB_MAX_QPS`).
  - Higher values allow more striping, at the cost of more CQ/QP resources.
- **`NCCL_IB_SPLIT_DATA_ON_QPS`** (param `IbSplitDataOnQps`):
  - When `0`, only a subset of QPs might actually carry data (others may remain unused or used only for specific purposes).
  - When `1`, data for a request is intentionally divided across all available QPs returned by `ncclIbCommBaseGetNqpsPerRequest`.

**Effect on behavior:** with multi‑QP + striping, a single logical send runs multiple IB work queues in parallel, improving throughput and better hiding per‑QP latencies, especially for large messages.

---

## 3. Adaptive‑Routing‑Friendly Send Protocol (AR)

**Goal:** Make the wire protocol friendly to fabrics that use adaptive routing, while maintaining correct completion semantics.

### What it does

- In `ncclIbMultiSend` (`src/transport/net_ib/p2p.cc`), there are two modes for the last work request:
  1. **Simple mode:** a single `IBV_WR_RDMA_WRITE_WITH_IMM` sends payload and signals completion.
  2. **AR mode:**
     - First, one or more `IBV_WR_RDMA_WRITE` operations send the bulk of the data.
     - Then an extra 0‑byte `IBV_WR_RDMA_WRITE_WITH_IMM` is posted purely to act as a completion signal.
- For multi‑recv (batched sends):
  - If `nreqs > 1`, the sender also writes a completion record array (sizes for each sub‑recv) and then posts the `RDMA_WRITE_WITH_IMM` as the final fence.
- This separation of *data transfer* and *completion signal* improves robustness with adaptive routing, because path changes affect only the timing of completions, not the correctness of data placement.

### When it is used

```cpp
if (nreqs > 1 || (comm->ar && reqs[0]->send.size > ncclParamIbArThreshold())) {
  // ... add extra WRs, then final RDMA_WRITE_WITH_IMM
}
```

- AR mode is triggered if:
  - There is a multi‑send (`nreqs > 1`), or
  - The device has `comm->ar` set **and** the message size exceeds `IB_AR_THRESHOLD`.

### Configuration knobs

- **`NCCL_IB_ADAPTIVE_ROUTING`** (param `IbAdaptiveRouting`):
  - Controls whether the device marks itself as AR‑capable (`ibDev->ar`).
  - Typically:
    - `-2` (auto) = enable AR only on InfiniBand (not on RoCE).
    - `0` = force AR off.
    - `1` = force AR on.
- **`NCCL_IB_AR_THRESHOLD`** (param `IbArThreshold`, default `8192` bytes):
  - Minimum message size for which AR mode is used when `comm->ar` is set.
  - Smaller messages still use the simple single‑WR form.

**Effect on behavior:** with AR enabled, large sends (and all multi‑sends) are decomposed into a data phase and a separate zero‑byte completion phase, making it safer for the fabric to reroute traffic without violating NCCL’s ordering expectations.

---

## 4. Enhanced Connection Establishment (ECE)

**Goal:** Negotiate and enable advanced IBTA/vendor features (e.g., congestion control profiles) on a per‑QP basis when both ends support them.

### What it does

- Each QP has an `ncclIbQpInfo` entry (`src/transport/net_ib/connect.cc`):

```cpp
struct ncclIbQpInfo {
  uint32_t qpn;
  struct ibv_ece ece;      // ECE fields
  int ece_supported;       // whether ECE is usable
  int devIndex;
};
```

- During connection setup, the plugin:
  - Checks if the device and driver support `ibv_ece`.
  - If ECE is enabled, populates the `ece` struct with the desired profile and marks `ece_supported`.
  - Exchanges this information with the peer via `ncclIbConnectionMetadata` so both sides agree on ECE settings.
  - Programs the QP with ECE attributes before moving it to RTS.

### Configuration knobs

- **`NCCL_IB_ECE_ENABLE`** (param `IbEceEnable`, default `1`):
  - `1` = attempt to enable ECE on supported devices.
  - `0` = disable ECE even if the NIC and verbs library support it.

**Effect on behavior:** when enabled and supported by hardware, QPs use ECE, which can provide better congestion control and fabric behavior on modern IB networks. When disabled or unsupported, connections fall back to classic RC settings.

---

## 5. Net/IB Profiling Hooks

**Goal:** Allow NCCL’s profiler to capture detailed per‑QP and per‑request timing for `net_ib` operations.

### What it does

- The profiling structures and callback live in `src/transport/net_ib/common.h` and `common.cc`:
  - `struct ncclProfilerInfo` stores per‑QP event handles and a `ncclProfilerNetIbDescr_v1_t` description.
  - `ncclProfilerCallback_t ncclProfilerFunction` is a function pointer set at initialization time.
- Each `ncclIbRequest` has an optional array `pInfo[NCCL_NET_IB_MAX_RECVS]` when `NCCL_ENABLE_NET_PROFILING` is defined at build time.
- In the send/recv paths (`src/transport/net_ib/p2p.cc`), guarded by `#ifdef NCCL_ENABLE_NET_PROFILING`:
  - The plugin fills in metadata: QP indices, direction (send/recv), sizes, ranks, etc.
  - Then calls `ncclProfilerFunction(..., ncclProfilerNetEventStart, ...)` when posting WRs.
  - Later, when completions arrive or requests are completed, it calls `ncclProfilerFunction(..., ncclProfilerNetEventStop, ...)`.
- The profiler uses these callbacks to construct a timeline of `net_ib` activity with QP‑level resolution.

### How it is enabled

- **Build‑time:** the code is guarded by the macro `NCCL_ENABLE_NET_PROFILING`.
  - When this macro is not defined, the `pInfo` arrays and profiling calls are compiled out.
- **Runtime:** NCCL’s profiler decides whether to collect and record these events when it passes a non‑NULL `profFunction` into `ncclIbInitDevices` / `ncclIbInit` (`src/transport/net_ib/init.cc`).

**Effect on behavior:** when both build‑time and runtime profiling are enabled, every `net_ib` request reports start/stop events for each QP it uses, allowing detailed analysis of transport behavior (latency, concurrency, striping efficiency, etc.) with minimal changes to the data path.
