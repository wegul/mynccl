# Virtual devices (vDev) and multi-rail merging in net_ib

This note explains how NCCL’s built-in `net_ib` plugin uses **virtual devices** (vDevs) to represent **one or more physical IB ports** (“rails”), and how that drives connection setup, QP layout, memory registration keys, and send/recv striping.

The goal is to clarify what it means when NCCL “merges” NICs/ports, and what code paths actually consume `vProps`.

## Terminology

- **Physical device**: an actual IB HCA port, represented in `net_ib` as an entry in `ncclIbDevs[]`.
- **Virtual device (vDev / mergedDev)**: what NCCL core enumerates as a “network device”. It is represented in `net_ib` as an entry in `ncclIbMergedDevs[]`.
- **`vProps`**: `ncclNetVDeviceProps_t` describing a vDev’s physical membership:
  - `vProps.ndevs`: number of rails
  - `vProps.devs[i]`: index into `ncclIbDevs[]` for rail `i`

## High-level picture (textual graph)

```
Hardware (ports)
  ncclIbDevs[0]  ncclIbDevs[1]  ncclIbDevs[2]  ...
      |              |              |
      +--------------+--------------+
                     |
Virtualization layer (what NCCL sees)
  ncclIbMergedDevs[vdev0].vProps = { ndevs=1, devs=[0] }
  ncclIbMergedDevs[vdev1].vProps = { ndevs=1, devs=[1] }
  ncclIbMergedDevs[vdev2].vProps = { ndevs=2, devs=[0,2] }   (merged multi-rail vDev)
                     |
                     v
Connection + data path allocate/use per-rail state based on vProps.ndevs
```

A key point: **a vDev is the unit NCCL selects**, but **a rail is the unit the plugin allocates verbs objects for**.

## Lifecycle and ownership (mergedDev vs comm->vProps)

This is the most common point of confusion, so here is the lifecycle spelled out explicitly.

- `ncclIbMergedDevs[]` is a **process-global vDev table** owned by the plugin implementation.
  - It is populated during device enumeration (singleton vDevs) and optionally extended later (multi-rail vDevs).
  - It is *not* per-communicator and not per-connection.
- Each `sendComm` / `recvComm` stores `comm->base.vProps` as a **copy** of one selected vDev entry.
  - In `net_ib` connect, the plugin does `mergedDev = ncclIbMergedDevs + dev; comm->base.vProps = mergedDev->vProps;`.
  - After that, the comm allocates per-rail state and QPs using `comm->base.vProps.ndevs`.
  - The comm does not “point back” to mergedDev; it only depends on the copied `vProps` (plus other per-comm state it creates).

Text graph for the ownership:

```
Process-global (plugin)
  ncclIbMergedDevs[vDevIndex].vProps  (membership definition)
                 |
                 |  selected by NCCL core as an integer device index
                 v
Per-connection object (plugin)
  ncclIbSendComm / ncclIbRecvComm
    base.vProps = COPY( ncclIbMergedDevs[vDevIndex].vProps )
    devs[0..ndevs-1] verbs state (PD/CQ/GID/etc per rail)
    qps[0..nqps-1]   striped across rails
```

## Where merging happens

### 1) Building a vDev (`makeVDevice`)

`net_ib` can create a multi-rail vDev by copying the requested physical device indices into a new mergedDev.

- The merge routine is `ncclIbMakeVDeviceInternal()` in:
  - ../../../../src/transport/net_ib/init.cc

It effectively does:

- validate the request (`props->ndevs`, link-layer compatibility, etc.)
- `mDev->vProps.devs[mDev->vProps.ndevs++] = props->devs[i]`
- compute merged metadata (name/speed)

The resulting vDev index is returned to NCCL core.

### 1b) Who calls `makeVDevice` (NCCL core topo / NIC fusion)

NCCL core calls `makeVDevice()` while building the network portion of its topology graph.

- The core constructs a candidate `ncclNetVDeviceProps_t vProps` by grouping *physical NICs*.
- If `vProps.ndevs == 1`, it skips (no reason to create a “virtual” NIC of size 1).
- If `vProps.ndevs > 1`, it calls the plugin’s `makeVDevice(&vDevIndex, &vProps)`.

Concrete code:

- `ncclTopoMakeVnic()` calls `netInfo->makeVDevice(&vDevIndex, vProps)` in [src/graph/topo.cc](src/graph/topo.cc#L1009-L1044).
- The `makeVDevice` function pointer is wired from the selected plugin via `netInfo.makeVDevice = comm->ncclNet->makeVDevice` in [src/graph/topo.cc](src/graph/topo.cc#L1469-L1498).

How the core decides which physical NICs to group:

- **Auto-merge**: uses `NCCL_NET_MERGE_LEVEL` (aka “mergeLevel”).
  - The core computes a “path distance” between every pair of physical NIC nodes in the topology XML (e.g., same PCI switch vs farther apart).
  - Starting from the first unplaced NIC, it adds other NICs whose distance is `<= mergeLevel` until hitting `NCCL_NET_MAX_DEVS_PER_NIC`.
  - See `ncclTopoAutoMerge()` in [src/graph/topo.cc](src/graph/topo.cc#L1120-L1164).
- **Force-merge**: uses `NCCL_NET_FORCE_MERGE` (explicit groups), bypassing auto grouping.
  - See `ncclTopoForceMerge()` in [src/graph/topo.cc](src/graph/topo.cc#L1039-L1118).

Important fact check about indices / ordering:

- In `net_ib`, singleton vDevs are created first during enumeration (one per physical device/port).
- Each successful call to `makeVDevice()` **appends one new vDev entry** to the end of the vDev table.
- So the vDev index returned by `makeVDevice()` is typically **>= number of physical NICs**, because it is a new entry, not a modification of index 0.
- Not every entry “contains all vDevs”; rather, each vDev entry contains a *membership list* of physical devices for that one vDev.

### How auto-merge starts even though groups begin at size 1

In `ncclTopoAutoMerge()`, every candidate group starts as “just the root NIC `i`”, so initially `vProps.ndevs == 1`.

The merge begins when the inner loop finds at least one additional NIC `j` satisfying:

- `paths[i*nPhysDevs + j] <= mergeLevel`
- `placedDevs[j] == 0`
- `j != i`

When that happens, the core does `vProps.devs[vProps.ndevs++] = j;`, so `vProps.ndevs` becomes 2 (or more). Only then will `ncclTopoMakeVnic()` call `makeVDevice()`.

So the lifecycle looks like:

```
start group:  vProps = [i]         (ndevs=1)
add neighbor: vProps = [i, j]      (ndevs=2)  -> will call makeVDevice()
no neighbor:  vProps = [i]         (ndevs=1)  -> ncclTopoMakeVnic() skips makeVDevice()
```

### 2) The default “singleton” vDevs

Even without merging, `net_ib` creates singleton vDevs (ndevs=1) so each physical port is selectable.

This is why “device indices” exposed to NCCL core can be thought of as:

```
vDev indices 0..(nPhysical-1)          : singleton vDevs (one physical device each)
vDev indices nPhysical..(nTotal-1)     : merged vDevs created by makeVDevice()
```

## How vDevs are *used* (the important part)

Once NCCL chooses a vDev index (the `dev` argument in connect/listen), the plugin uses `mergedDev->vProps` to size and allocate resources.

### How the core chooses the vDev index for a connection/comm

NCCL core ultimately selects a single integer `netDev` for each network connection it is about to establish (per channel/peer/transport flow). That integer is a vDev index in the plugin’s `devices()` namespace.

You can see the “device index is an argument” behavior clearly in the proxy path:

- The proxy stores the chosen device index as `resources->netDev` and passes it to the plugin as the `dev` argument when connecting/listening.
- For connect, that call is in [src/transport/net.cc](src/transport/net.cc#L842-L855):
  - `proxyState->ncclNet->connect(proxyState->netContext, resources->netDev, ...)`

So, yes: the comm object is reactive. The core designates `dev` (vDev index), and the plugin builds the comm (rails/QPs/keys) based on the corresponding `vProps`.

## End-to-end worked example (3 physical NICs)

This example is meant to be “index-accurate” and cover the full lifecycle: enumeration → topo auto-merge → vDev table growth → core selection (`netDev`) → connect building `comm->base.vProps`.

Assume the plugin enumerates 3 physical NIC ports and exposes them as physical device indices:

```
Physical indices (plugin namespace): 0, 1, 2
```

### Step A — net_ib enumeration creates singleton vDevs

During IB device enumeration, `net_ib` creates singleton vDevs for each physical device/port.

After enumeration:

```
ncclIbDevs (physical)
  [0] NIC0
  [1] NIC1
  [2] NIC2

ncclIbMergedDevs (vDev table exposed to core via net->devices())
  [0] vProps=[0]   (singleton)
  [1] vProps=[1]   (singleton)
  [2] vProps=[2]   (singleton)

ncclNMergedIbDevs = 3
```

### Step B — topology auto-merge forms groups using mergeLevel

The core computes a path-distance matrix `paths[i,j]` between physical NIC nodes.

Suppose, for your machine/topology, the distances are such that:

- NIC0 is not within `mergeLevel` of NIC1 or NIC2
- NIC1 and NIC2 are within `mergeLevel` of each other

Then the loop in `ncclTopoAutoMerge()` behaves like:

```
i=0: start vProps=[0]
     inner loop finds no eligible j
     => vProps.ndevs stays 1
     => ncclTopoMakeVnic() SKIPS makeVDevice()

i=1: start vProps=[1]
     inner loop finds j=2 eligible
     => vProps becomes [1,2]
     => ncclTopoMakeVnic() CALLS makeVDevice(vProps=[1,2])
```

### Step C — makeVDevice appends a new vDev entry

When the core calls `makeVDevice` with `vProps=[1,2]`, net_ib appends one new vDev entry and returns its index.

After that call succeeds:

```
ncclIbMergedDevs
  [0] vProps=[0]
  [1] vProps=[1]
  [2] vProps=[2]
  [3] vProps=[1,2]   (NEW merged vDev)

ncclNMergedIbDevs = 4
```

Core-side, the topology marks NIC1 and NIC2 as `keep=0` (hidden as standalone choices), and vDev3 becomes the “preferred representation” of that fused group.

### Step D — core picks a device index (netDev) for a connection

At connection-planning time, NCCL core chooses one topology NET node to use for a specific connection, and that node carries a single integer `netDev` (the plugin device index).

In this example, for a connection that wants to use the fused link:

```
req->netDev = 3
```

For a connection that uses the lone NIC0:

```
req->netDev = 0
```

### Step E — connect builds comm->base.vProps from the selected vDev index

When the proxy calls the plugin:

```
connect(..., dev=req->netDev, ...)
```

net_ib does:

```
mergedDev = ncclIbMergedDevs + dev
comm->base.vProps = mergedDev->vProps   (COPY)
```

So if `dev=3`, then:

```
comm->base.vProps.ndevs = 2
comm->base.vProps.devs  = [1,2]

comm allocates per-rail verbs state for rail0->phys1 and rail1->phys2
comm creates nqps = IB_QPS_PER_CONNECTION * 2
data path stripes requests across those QPs/rails
```

This is the key “handoff”: core chooses one integer `dev`, and the plugin’s comm is built around the copied `vProps`.

### A) Connection setup allocates per-rail state

Sender-side connect path (simplified):

1) Select vDev and send vProps to peer
2) Receive peer vProps and choose a QP count compatible with both sides
3) For each rail in `comm->base.vProps.ndevs`:
   - initialize the per-rail comm base (PD/CQ/GID, etc.)
4) Create QPs and transition to RTS

Concrete code:

- Assign vDev membership into the comm:
  - ../../../../src/transport/net_ib/connect.cc (see `comm->base.vProps = mergedDev->vProps;`)
- Size QPs using number of rails:
  - `localNqps = ncclParamIbQpsPerConn() * comm->base.vProps.ndevs;`
- Per-rail init loop:
  - `for (int i = 0; i < comm->base.vProps.ndevs; i++) { ... ncclIbInitCommDevBase(ibDevN, ...) }`

Receiver-side accept path does the same per-rail init with its chosen vProps (and can reduce the list to match the peer).

### B) Memory registration is per-rail (lkeys/rkeys arrays)

Because each rail can have a different PD, the MR handle stores one `ibv_mr*` per rail.

- `struct ncclIbMrHandle { ibv_mr* mrs[NCCL_IB_MAX_DEVS_PER_NIC]; };`
  - ../../../../src/transport/net_ib/common.h

Send path stores lkeys for each rail:

- `for (int i = 0; i < comm->base.vProps.ndevs; i++) req->send.lkeys[i] = mhandleWrapper->mrs[i]->lkey;`
  - ../../../../src/transport/net_ib/p2p.cc

Recv path advertises rkeys for each rail into the CTS FIFO:

- `for (int j = 0; j < comm->base.vProps.ndevs; j++) localElem[i].rkeys[j] = mhandleWrapper->mrs[j]->rkey;`
  - ../../../../src/transport/net_ib/p2p.cc

### C) Striping: requests are spread across QPs, and QPs are associated with rails

The striping is done at the QP level: for a given request, the code iterates over the QPs chosen by `ncclIbCommBaseGetQpForRequest()`.

- QP selection per request:
  - ../../../../src/transport/net_ib/common.h

In the send data posting loop:

- pick a QP for each “lane” `i` (0..nqpsPerRequest-1)
- infer which rail is used via `qp->devIndex`
- choose the correct remote key index via `qp->remDevIdx`
- choose the correct local lkey via `req->send.lkeys[devIndex]`

Concrete code:

- ../../../../src/transport/net_ib/p2p.cc
  - `NCCLCHECK(ncclIbCommBaseGetQpForRequest(..., &qp, ...));`
  - `int devIndex = qp->devIndex;`
  - `comm->wrs[r].wr.rdma.rkey = slots[r].rkeys[qp->remDevIdx];`
  - `comm->sges[r].lkey = reqs[r]->send.lkeys[devIndex];`

This is the “real” place where a multi-rail vDev changes behavior: **more rails ⇒ more per-rail QPs and per-rail keys ⇒ more striping opportunities**.

## MPIB today (merged vDevs are effectively disabled)

`mpib` reuses the same *shape* as `net_ib` (a physical device list + a “merged device” table containing `vProps`), but today it intentionally exposes only a singleton vDev.

### What MPIB actually builds at init

- Enumeration is intentionally constrained: it requires `NCCL_IB_HCA` and opens exactly that HCA, uses port 1 only, and requires the Ethernet link-layer (RoCE). It creates exactly one physical entry: `mpibDevs[0]`.
- It also initializes a merged-vDev table, but only with one singleton entry:

```
mpibNIbDevs = 1
mpibNMergedIbDevs = 1

mpibMergedDevs[0].vProps = { ndevs=1, devs=[0] }
```

### What NCCL core can see

- `devices()` returns `mpibNIbDevs` (the physical count), so NCCL core only sees device index `0`.
- `getProperties(dev)` only accepts `dev < mpibNIbDevs` and reports `props->vProps = { ndevs=1, devs=[dev] }`.

### makeVDevice is not implemented

- `mpibMakeVDevice()` returns `ncclInvalidUsage`, so NCCL core NIC fusion via `ncclTopoMakeVnic()` cannot create a multi-rail vDev when `mpib` is the selected plugin.

### Practical consequence

- With `mpib` today, every connection sees `vProps.ndevs == 1` and there is no merged multi-rail behavior.
- Even though multi-rail is disabled at the topology/fusion layer, the connect/accept handshake and the data path are still written in a net_ib-like way (they copy `mergedDev->vProps` into `comm->base.vProps` and size loops/QPs using `comm->base.vProps.ndevs`). That means the internal structure is compatible with multi-rail, but the plugin never exposes or creates `ndevs>1` vDevs.
