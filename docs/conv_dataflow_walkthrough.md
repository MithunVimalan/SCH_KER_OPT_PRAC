# Conv2D 3×3 on the Core Ultra 7 270K Plus: plan first, then follow every byte

Problem: `N=16, C=3, H=W=128, K=8, 3×3 kernel, stride 1, pad 1`. Code: [`conv_fast.c`](../conv_fast.c)
(UW=12, nChw8c output, the default build).

This document has four parts. Read them in order: Part A uses no hardware facts at all.

| Part | Question it answers |
|---|---|
| **A. The approach** | Start from the output, tile it, find the inputs each tile needs, size it, and chunk again until one tile is one FMA. |
| **B. The hardware** | What each component (registers, FMA, ports, L0/L1/L2/L3, DRAM, pages, cores) allows, and which number in Part A it fixes. |
| **C. The data's path** | Every step from DRAM → L3 → L2 → L1 → L0 → registers → FMA → back down, with bytes per step and what each memory holds at each moment. |
| **D. What the numbers show** | The bottleneck, what is idle, and what to change next. |

Terms used throughout:

- **L0 / L1 / L2 / L3**: Intel's names for the Lion Cove P-core caches. L0 = 48 KiB data cache next to the
  load units, L1 = 192 KiB, L2 = 3 MiB per core, L3 = shared by all cores. (Linux tools may label the
  48 KiB L0 as "L1d"; see the appendix.)
- **line**: one 64 B cache line, the unit every transfer below the registers moves in.
- **register / ymm**: one 256-bit AVX2 register = 32 B = 8 floats.
- **halo**: the extra input border a tile needs. For a 3×3 kernel that is R−1 = 2 more rows and S−1 = 2 more columns than the output tile.
- **RFO (read-for-ownership)**: before a core may write into a line, it must fetch that line. So every output line is *read* once before it is written.
- KiB/MiB are powers of 1024. GB/s is 10⁹ B/s.

---

## Part A: the approach (no hardware yet)

### A1. Write the problem down

```
for n in 0..15                         image
 for k in 0..7                         output channel
  for p in 0..127                      output row
   for q in 0..127                     output column
     y = bias[k]
     for c in 0..2, r in 0..2, s in 0..2                          27 taps
        y += x[n][c][p-1+r][q-1+s] * w[k][c][r][s]                (skip taps that land in the padding)
```

| Tensor | Shape | Elements | Bytes | Useful sub-sizes |
|---|---|---|---|---|
| input `x` | 16×3×128×128 (NCHW) | 786,432 | **3 MiB** | 192 KiB per image, 64 KiB per channel plane, **512 B per row** (= 8 lines) |
| weights `w` | 8×3×3×3 | 216 | **864 B** | 27 taps × 8 k; repacked to `wv[c][r][s][k]` = 27 vectors of 32 B |
| bias | 8 | 8 | 32 B | one ymm |
| output `y` | 16×128×128×8 (nChw8c) | 2,097,152 | **8 MiB** | 512 KiB per image, **4 KiB per output row** (= 64 lines), 32 B per pixel |

### A2. Count the work, the minimum traffic and the reuse

**Work.** For each channel, 3×128 = 384 (output row, kernel row) pairs exist, and 2 of them land in the padding
(row 0 with r=0, row 127 with r=2), leaving 382. Columns work the same way. So:

- MACs on real pixels = 16 × 8 × 3 × 382 × 382 = **56,034,816**, which is **112 MFLOP**
- one FMA instruction does 8 MACs (8 output channels at once), so **7,004,352 FMA instructions** (F11 in the code)

**Minimum traffic.** Every input byte must be read at least once and every output byte written once:
3 MiB + 8 MiB = **11 MiB** (+864 B of weights).

**Arithmetic intensity** = 112 MFLOP / 11.5 MB ≈ **9.7 FLOP per byte**. Part B compares this number with the machine.

**Reuse.** For each operand, how many times one value is used:

| Operand | Uses per value | Made of |
|---|---|---|
| one input value `x[n][c][i][j]` | **72** | 3 kernel rows (r) × 3 kernel columns (s) × 8 output channels (k) |
| one weight `w[k][c][r][s]` | **≈262,144** | every pixel of every image: 16 × 128 × 128 |
| one output value `y` | **27** accumulations, then written **once** | 3 c × 3 r × 3 s |

The rest of Part A decides, for each of these reuse factors, **which memory level captures it**. The closer
to the FMA a reuse is captured, the less data has to travel.

### A3. Start from the output and tile it

Repeat this loop until one tile is one instruction:

```
pick an output tile
  -> inputs it needs?        footprint = tile + halo (R-1 rows, S-1 columns), all C channels
  -> weights it needs?
  -> working set = out + in + w
  -> does it fit in the memory it should live in?
        no  -> split the tile again (which dimension: see A4)
        yes -> pick the order to walk the tiles so that neighbours share data,
               and count what is NEW per step. Only the new data has to travel.
```

#### Level 1: the whole problem
- out 8 MiB, in 3 MiB, w 864 B, so working set **11 MiB**. Only DRAM or a large L3 can hold this.
- **Split by image `n`.** Images share nothing except the 864 B of weights: no halo and no dependency.
  This becomes the thread split.

#### Level 2: one image
- out 512 KiB, in 192 KiB, w 864 B, so working set **≈705 KiB**.
- Fits only in a cache of several hundred KiB or more (L2-sized). **Split by output row `p`.**

#### Level 3: one output row `p` (all 128 columns, all 8 k)
- out = 128 px × 8 k × 4 B = **4 KiB**
- in = input rows p−1, p, p+1 of all 3 channels = 9 rows × 512 B = **4.5 KiB**
- w = 864 B, so working set **≈9.4 KiB**. This fits easily in a small (tens of KiB) cache.
- **Neighbour sharing:** row p+1 needs input rows p, p+1, p+2. Six of the nine rows are already present,
  so only **3 new rows = 1.5 KiB** must arrive per output row. This is a **sliding window**: each input row
  arrives once and is used by 3 output rows, so the **×3 (r) reuse is captured at this level**.
- 9.4 KiB is still far more than the register file can hold. **Split along `q`.**

#### Level 4: one register block of 12 output pixels × 8 k
- out = 12 × 32 B = **384 B = 12 registers**
- in = 3 c × 3 rows × (12 + 2 halo) columns = 126 floats = **504 B**
- w = 864 B, so working set **≈1.75 KiB**, against a register file that holds 512 B. **It does not fit.**
- The three parts behave differently, so split the *work*, not the output:
  - the **output** (384 B) is updated 27 times per value, so it **stays** in 12 registers for the whole block (output-stationary)
  - each **input** value is needed only briefly, so stream them through **1 register**
  - the **weights**: each tap vector serves all 12 pixels of one row pass, so stream them **3 at a time**
- **Split the reduction (c, r) into 9 passes.**

#### Level 5: one (c, r) pass inside the block
- 14 input values from one input row, 3 weight vectors (s = 0, 1, 2 of this c, r), 12 accumulators
- 12 + 3 + 1 = **16 registers**, which is the whole register file. 9 passes complete the block.

#### Level 6: one input column `t`
- One value is broadcast to all 8 lanes and fed to **up to 3 FMAs**. It is tap s=2 of pixel t−2, s=1 of pixel t−1
  and s=0 of pixel t. **The ×3 (s) reuse is captured in a register.**

#### Level 7: one FMA
- `acc[j] += x × wv[c][r][s][0..7]`: one input value times 8 output channels. **The ×8 (k) reuse is captured
  inside the vector lanes.**

#### The ladder on one page

| Level | Output tile | Input it needs | Weights | Working set | Should live in | Shared with next tile | New data per step |
|---|---|---|---|---|---|---|---|
| 1 problem | 8 MiB | 3 MiB | 864 B | 11 MiB | DRAM / L3 | n/a | n/a |
| 2 image | 512 KiB | 192 KiB | 864 B | 705 KiB | L2 | weights only | 192 KiB |
| 3 output row | 4 KiB | 4.5 KiB (9 rows) | 864 B | 9.4 KiB | L0 | 6 of 9 input rows | **1.5 KiB** |
| 4 block, 12 px | 384 B | 504 B (3×3×14) | 864 B | 1.75 KiB | registers + L0 | 2 columns × 9 rows | 12 columns × 9 rows |
| 5 (c,r) pass | 12 partial sums | 14 values | 3 vectors | 16 registers | registers | the accumulators | 14 x + 3 w |
| 6 column t | ≤ 3 pixels | 1 value | 3 vectors | n/a | registers | n/a | 1 x |
| 7 FMA | 1 px × 8 k | 1 value | 1 vector | 3 registers | FMA unit | n/a | n/a |

**Where each reuse factor is captured:**

| Operand | Captured where | How |
|---|---|---|
| input ×8 (k) | inside one FMA | 8 lanes share one broadcast value |
| input ×3 (s) | register | one broadcast feeds 3 FMAs |
| input ×3 (r) | L0 | an input row stays for 3 consecutive output rows |
| weights | registers + L0 | 3 vectors stay in registers for 12 pixels, and the 864 B never leaves L0 |
| output ×27 | register | an accumulator is written to memory only when finished: **zero partial-sum traffic** |

8 × 3 × 3 = 72, so every input reuse is captured above L1. Below L0, each byte moves exactly once.

**The cost** of this plan: the 2-column halo means 148 broadcasts per row pass instead of 128 (+15.6%), and
weights are reloaded into registers 3 vectors per pass. Both are L0 → register loads, which are cheap. Neither adds traffic below L0.

### A4. Why split along these dimensions and not the others

| Choice | Reason |
|---|---|
| **N first** | No halo, no dependency, equal work per image, so it is the thread split. |
| **Do not split K** | K = 8 is exactly one ymm. Splitting K means reading every input once per k-group, which loses the ×8 reuse. |
| **Do not split C** | C is part of the reduction. Splitting it forces partial sums to be stored and reloaded (8 MiB written and read again per split). All 27 taps stay inside the block, so a pixel finishes in its register. |
| **OH next** | Halo of 2 rows, and walking `p` in order turns that halo into the sliding window, so no traffic is wasted. |
| **OW last** (12-px blocks) | A 2-column halo only costs extra L0 loads, not extra traffic from below. |
| **Vectorise over K** | One broadcast input × one weight vector gives 8 outputs, with no shuffles and no horizontal sums. Vectorising W would need shifted loads per tap, and C has only 3 lanes. |
| **Padding is never stored** | Compile-time variants handle it: top row (`RLO=1`), bottom row (`RHI=1`), left block (skip `t=0`), right block (skip `t=NW+1`). There is no padded copy of the input and no wasted FMA. |
| **Bit-exact order** | Each pixel gets bias first, then taps in (c, r, s) order, exactly like the reference loop. Inside a pass, pixel j gets s=0 at `t=j`, s=1 at `t=j+1`, s=2 at `t=j+2`. The roundings happen in the same order, so the result has the same bits. |
| **Layout follows the tile** | Weights are repacked to `[c][r][s][k]` so each tap is one aligned 32 B load. Output is nChw8c so the 12 pixels × 8 k of a block are one contiguous 384 B run: exactly 6 cache lines. |

### A5. The same checklist for any kernel

1. Shapes, bytes, MACs, minimum traffic, intensity.
2. The reuse factor of every operand, split into the dimensions it comes from.
3. Output-first tiling ladder: footprint, working set, fits or not, neighbour sharing, new data per step.
4. To choose a split: independent dimensions first, then small-halo dimensions. Never split the reduction if
   that pushes partial sums out of registers, and never split the vector dimension.
5. Choose the walk order so that halos become sliding windows.
6. **Only now** look at the hardware and fix the exact sizes (Part B).

---

## Part B: the hardware

### B1. The components (Lion Cove P-core, Arrow Lake)

The values are approximate public figures. Check them on your machine (appendix). The conclusions below have
wide margins, so ±20% changes nothing.

| Component | Size / rate | What it constrains |
|---|---|---|
| vector registers | **16 ymm × 32 B = 512 B** (no AVX-512 on this chip, so 16 not 32) | accumulators + weights + broadcast ≤ 16 |
| FMA units | **2 per cycle**, 256-bit, **latency 4 cycles** | at least 2 × 4 = **8 independent accumulators** must be in flight |
| load ports | **3 loads / cycle** (up to 32 B each) | loads per FMA ≤ 1.5 |
| store ports | 2 stores / cycle | not a concern here |
| **L0** data cache | **48 KiB**, ≈4–5 cycles | the per-row working set (9.4 KiB) must live here |
| **L1** data cache | **192 KiB**, ≈9 cycles | buffer between L0 and L2 |
| **L2** | **3 MiB per core**, ≈17 cycles | one core's images |
| **L3** | **36 MiB shared** (check `lscpu` on your chip), tens of ns | the whole problem (11 MiB) |
| DRAM | DDR5 dual channel: ≈102 GB/s peak at 6400 MT/s, ≈100 ns | matters only when data is cold |
| cache line | 64 B | the unit of every transfer below registers |
| pages | 4 KiB, or 2 MiB with `madvise(MADV_HUGEPAGE)` | TLB reach |
| cores | **8 P-cores, no Hyper-Threading** | at most 8 useful threads on P-cores |
| out-of-order window | ROB ≈ 576 µops | hides L0/L1/L2 misses; L3/DRAM latency needs the prefetchers |
| prefetchers | L2 streamer (sequential, within 4 KiB), L0/L1 next-line and stride | our pattern is 3 input streams + 1 output stream, which they handle easily |

### B2. Each constraint fixes one number

**1. Block width UW = 12, from latency × throughput and the register count.**
- Each pixel's 27 FMAs form a dependency chain: each needs the previous result. With 4-cycle latency and 2 FMAs
  issuing per cycle, 8 chains must be in flight, so **UW ≥ 8**.
- With 16 registers, UW + 3 weights + 1 broadcast ≤ 16, so **UW ≤ 12**.
- UW = 12: 128 = 10 × 12 + 8, so 10 blocks of 12 and one tail of 8. The tail still meets the ≥ 8 rule.
- Chain check: per block, one accumulator's chain is 27 × 4 = 108 cycles, but the block needs 324 / 2 = 162
  cycles of FMA throughput. The chains are not the limit.

**2. Loads per FMA, against 3 load ports.**
- *Source-level intent* (`WREG=0`, every FMA takes its weight as a memory operand): per pass, 14 broadcasts +
  36 weight loads = 50 loads for 36 FMAs, or 1.39 per FMA. That is below 1.5, but uses ≈92% of the load ports.
- *What the compilers emit* (gcc 13 and clang 18, `-O3 -march=alderlake` or `x86-64-v3`, checked in the appendix):
  they load the 3 weight vectors **once per pass** into registers, giving 14 + 3 = **17 loads per 36 FMAs = 0.47 per FMA**
  (≈32% of the load ports). Bias is folded into the first FMA of each pixel. The register file is full, so both
  compilers spill a few values to the stack (an accumulator or the bias). The stack is in L0, so this costs almost nothing.
- Either way the **FMA ports are the limit**, not the loads.

**3. L0 (48 KiB) against the 9.4 KiB per-row working set.** It fits 5× over, so every input line enters L0
exactly once per image. The simulation confirms 11 MiB per rep into L0, which is exactly the minimum.

**4. L2 (3 MiB) against one core's share.**
- `-t 1`: one core walks all 16 images, 11 MiB per rep, more than 3 MiB. Every rep comes back from L3.
- `-t 8`: 2 images per core = 2 × 705 KiB = 1.4 MiB, less than 3 MiB. **The data stays in L2 across reps and L3 sees no traffic.**

**5. L3 (36 MiB) against 11 MiB.** After the first rep, the benchmark never touches DRAM.

**6. DRAM roofline (cold data only).**
- Machine peak: 8 cores × 2 FMA × 16 FLOP × 3.7 GHz = **947 GFLOP/s**. DRAM ≈ 102 GB/s. The balance point is 947 / 102 ≈ **9.3 FLOP/B**.
- This kernel: 9.7 FLOP/B if the output could be written without the RFO read, or **5.6 FLOP/B with it**
  (normal stores read each output line first: 3 + 8 + 8 = 19 MiB).
- So **with 8 threads and cold data this kernel is memory-bound**. It would need 19.9 MB in 0.118 ms = 168 GB/s.
  With 1 thread it needs only 21 GB/s, which DRAM supplies.

**7. Cache-line alignment.**
- An input row is 512 B = 8 lines and starts on a line boundary.
- An output block is 384 B at byte offset 384·b inside the row, so it is line-aligned and covers exactly 6 full lines
  (the tail covers 4). No line is written by two blocks. Images are 512 KiB apart, so no two threads ever share a line.

**8. Pages.** With 2 MiB pages, the input spans 2 pages and the output 4, so the TLB never misses.
With 4 KiB pages, each output row (4 KiB) is exactly one page, so there is a new TLB entry every output row, and
each row's input touches 3 pages (one per channel).

**9. Threads.** `read_pcores` reads `/sys/devices/cpu_core/cpus` (8 CPUs). `-t 16` maps `pc[t % 8]`, which puts two
threads on each core. Without SMT they take turns. Use `-t 8` (2 images per core) or `-t 1`.

---

## Part C: the data's path, step by step

### C0. Before timing starts (setup)

1. `big_alloc` allocates 2 MiB-aligned buffers, asks for huge pages, and `memset`s them to 0. This touches
   every page, so the page faults happen here and not inside the timed loop. Input: 4 MiB (3 MiB rounded up).
   Output: 8 MiB.
2. `rand()` writes 3 MiB of input, plus weights (864 B) and bias (32 B). `pack_weights` reorders `wt[k][c][r][s]` into `wv[c][r][s][k]`.
3. Result: the data was just written, so most of it is still in the caches (mostly L3) when timing starts.
   **The benchmark is warm from the first rep**, and the first 10% of reps are discarded anyway. The DRAM → L3
   step below describes a real pipeline, or a run after a cache flush.

### C1. The map

```
            capacity        shared?     latency       what moves between levels
DRAM        GBs             all cores   ~100 ns       64 B lines (one DDR5 burst each)
  |  ^
L3          36 MiB          all cores   tens of ns    64 B lines
  |  ^
L2          3 MiB           per core    ~17 cycles    64 B lines
  |  ^
L1          192 KiB         per core    ~9 cycles     64 B lines
  |  ^
L0          48 KiB          per core    ~4-5 cycles   64 B lines in from L1
  |  load ports: 3/cycle    ^ store ports: 2/cycle
  v                         |
REGISTERS   16 x 32 B = 512 B           4 B broadcast (copied to 8 lanes), 32 B weight vector, 32 B store
  |  ^
  v  |
FMA x 2     acc[k] = x * w[k] + acc[k] for k = 0..7, 4-cycle latency, 2 per cycle
```

Going down: lines are **filled** from the level below on a miss. Going up: **dirty** (written) lines are
**written back** when they are evicted.

### C2. One image, cold, single thread: every step with counts

#### The input path

| Step | What moves | Amount per image | When |
|---|---|---|---|
| 1. DRAM → L3 | image n's input, 3 channel planes | **192 KiB = 3,072 lines** | streamed: 1.5 KiB (24 lines) per output row |
| 2. L3 → L2 | same lines | 192 KiB | the L2 streamer requests them ahead of use |
| 3. L2 → L1 → L0 | same lines | 192 KiB | each line exactly **once** (sliding window) |
| 4. L0 → register | 4 B broadcasts, each copied to 8 lanes | **169,608 broadcasts** (663 KiB of 4 B reads) | each value is broadcast 3.47× on average |

The demand pattern is **3 interleaved sequential streams**, one per channel plane (the planes are 64 KiB apart). Each stream
moves forward 512 B (8 lines) per output row. Timeline inside one image:

- before output row 0: rows 0 and 1 of the 3 channels = 6 × 512 B = **3 KiB**
- before each output row 1..126: one new row per channel = **1.5 KiB**
- before output row 127: nothing new
- total: 3 KiB + 126 × 1.5 KiB = **192 KiB**, the whole image read exactly once

The 3.47 broadcasts per value: each input row is used by 3 output rows (one per r), and within one row pass
adjacent blocks overlap by 2 columns (148 broadcasts for 128 columns): 3 × 148/128 ≈ 3.47.

#### The weight path

| Step | What moves | Amount | When |
|---|---|---|---|
| 1–3. DRAM → … → L0 | `wv`, 27 vectors | 864 B = **14 lines**, once | first block of the first image; afterwards it stays in L0, touched every few cycles, so it is never evicted |
| 4. L0 → register | 3 vectors per (c, r) pass per block | **37,818 vector loads per image** (1.15 MiB of reads, all of them from the same 864 B) | 3 per pass |

At source level (`WREG=0`, one memory operand per FMA) this would be 437,772 loads, 13.4 MiB per image. It all hits L0 either way.

#### The output path (the part most people forget)

| Step | What moves | Amount per image | When |
|---|---|---|---|
| 1. DRAM → L3 → L2 → L1 → L0 (**RFO**) | the *old* contents of each output line, read only so it can be overwritten | **512 KiB = 8,192 lines** | on the first store to each line |
| 2. register → L0 | 32 B stores, 12 per block (8 for the tail) | **16,384 stores = 512 KiB** | at the end of each block |
| 3. L0 → L1 → L2 → L3 (→ DRAM) | dirty lines written back on eviction | **512 KiB** | L0 keeps them about 9 rows, L1 about 37 rows, then they move on |

The output is 2.7× larger than the input (K = 8 > C = 3), and the RFO doubles it again: **per image, 704 KiB
comes up and 512 KiB goes down**, and 73% of the upward traffic is RFO of output lines.

#### Per-image totals (cold, `-t 1`)

| Boundary | Up (fill) | Down (write-back) |
|---|---|---|
| DRAM ↔ L3 | 704 KiB (192 in + 512 RFO) + 864 B weights | 512 KiB (eventually, only when L3 evicts it) |
| L3 ↔ L2 | 704 KiB | 512 KiB |
| L2 ↔ L1 | 704 KiB | 512 KiB |
| L1 ↔ L0 | 704 KiB | 512 KiB |
| L0 ↔ registers | 169,608 broadcasts + 37,818 weight vectors | 16,384 vector stores (512 KiB) |
| registers ↔ FMA | 437,772 FMAs × (3 registers in, 1 out) | n/a |

### C3. Zoom: one interior output row (p = 64)

**Before the row starts,** L0 already holds input rows 63 and 64 of each channel (6 rows = 3 KiB, left
over from rows 62 and 63) and the 864 B of weights.

**New data in:** input row 65 of channels 0, 1, 2 = 3 × 512 B = **1.5 KiB = 24 lines**. The L2 streamer has usually
fetched it already. The **64 output lines** of row 64 (4 KiB) arrive by RFO as the stores reach them.

**Work:** 11 blocks × 9 passes. 9 × 382 = **3,438 FMAs = 1,719 cycles** at the FMA limit (≈0.46 µs at 3.7 GHz).
Loads: 9 × 148 = 1,332 broadcasts + 9 × 11 × 3 = 297 weight vectors. Stores: 128.

**Which input lines each block reads.** An input row is 128 floats = lines #0..#7 of 16 floats each:

| Block b | q0 | Pixels | Input columns read | Row lines touched | Output lines written (within the 64-line row) |
|---|---|---|---|---|---|
| 0 (left) | 0 | 0–11 | 0–12 (column −1 is padding) | #0 | 0–5 |
| 1 | 12 | 12–23 | 11–24 | #0 #1 | 6–11 |
| 2 | 24 | 24–35 | 23–36 | #1 #2 | 12–17 |
| 3 | 36 | 36–47 | 35–48 | #2 #3 | 18–23 |
| 4 | 48 | 48–59 | 47–60 | #2 #3 | 24–29 |
| 5 | 60 | 60–71 | 59–72 | #3 #4 | 30–35 |
| 6 | 72 | 72–83 | 71–84 | #4 #5 | 36–41 |
| 7 | 84 | 84–95 | 83–96 | #5 #6 | 42–47 |
| 8 | 96 | 96–107 | 95–108 | #5 #6 | 48–53 |
| 9 | 108 | 108–119 | 107–120 | #6 #7 | 54–59 |
| 10 (tail) | 120 | 120–127 | 119–127 (column 128 is padding) | #7 | 60–63 |

Each block reads these lines in 9 input rows: (c, p−1), (c, p), (c, p+1) for c = 0, 1, 2.

**After the row:** the 4 KiB of results sit dirty in L0. They stay about 9 more rows and then move to L1
(snapshot in C6).

### C4. Zoom: one block (b = 5: pixels 60–71 of row p)

**Register map (16 of 16 in use):**

```
acc0 .. acc11   y[n][p][60..71][k=0..7]       12 registers   stay for the whole block
w0, w1, w2      wv[c][r][s=0,1,2][k=0..7]      3 registers   reloaded every pass
x               one input value, copied to 8 lanes   1 register   reloaded every column
(bias: folded into the first FMA of each pixel by the compiler)
```

**One pass (c, r)**. The row pointer is `&x[n][c][p-1+r][59]` (`q0 − 1`), and `t` walks 14 columns:

| t | input column | FMAs issued, in code order |
|---|---|---|
| 0 | 59 | acc0 += x·w0 |
| 1 | 60 | acc0 += x·w1, acc1 += x·w0 |
| 2 | 61 | acc0 += x·w2, acc1 += x·w1, acc2 += x·w0 |
| 3 | 62 | acc1 += x·w2, acc2 += x·w1, acc3 += x·w0 |
| … | … | acc[t−2] += x·w2, acc[t−1] += x·w1, acc[t] += x·w0 |
| 11 | 70 | acc9 += x·w2, acc10 += x·w1, acc11 += x·w0 |
| 12 | 71 | acc10 += x·w2, acc11 += x·w1 |
| 13 | 72 | acc11 += x·w2 |

That is 1 + 2 + 3×10 + 2 + 1 = **36 FMAs**, **14 broadcasts**, **3 weight loads**, **18 cycles** at the FMA limit.
The 17 loads would need only about 6 cycles of the 3 load ports.
Pixel 60 (acc0) gets column 59 with s=0, then 60 with s=1, then 61 with s=2: the reference order.

**Nine passes:** (c, r) = (0,0) (0,1) (0,2) (1,0) … (2,2). After pass 9, each accumulator holds its pixel's 8
finished output channels. Total: **324 FMAs, 162 cycles**, 126 broadcasts + 27 weight loads.

Inside one block, each value of the 3×3×14 input footprint is broadcast **exactly once** (126 broadcasts = 126
values). The 3.47× repeats come from neighbouring blocks (the halo) and neighbouring rows (r).

**Store:** 12 × `vmovaps` (aligned 32 B), 384 B written to output lines 30–35 of row p. That is 6 whole lines, so no other block touches them.

**Bytes moved for this block:** in from L0: 504 B of input values + 864 B of weight vectors. Out to L0: 384 B. All of
it happens between L0 and the registers. Nothing below L0 is involved unless that row's lines arrive for the first time.

### C5. Zoom: one FMA

`vfmadd231ps acc, x, w`: for k = 0..7, `acc[k] = x × w[k] + acc[k]`, rounded once.
3 registers in (96 B), 1 register out (32 B), **8 MACs**, 4-cycle latency, 2 can start every cycle.
The bias came first, then the taps in (c, r, s) order, so this is the same sequence of `fmaf` calls as the
reference loop. That is why the result is bit-exact.

### C6. What each memory holds at one moment

Simulated snapshot (single thread, right after output row 64 of an image; produced by
`tools/conv_traffic_sim.py snapshot 64`):

| Memory | Capacity | What is in it | Of that, still needed |
|---|---|---|---|
| registers | 512 B | 12 accumulators + 3 weight vectors + 1 broadcast | all of it |
| **L0** | 48 KiB (full) | weights 14 lines; input rows 56–65 (≈13 KiB); output rows 56–64 (≈34 KiB, dirty) | weights + input rows 64–65 (≈3.9 KiB). Row 66 is next to arrive, and the current output row is being written |
| **L1** | 192 KiB (full) | input rows ≈34–65; output rows ≈28–64 (dirty) | **nothing**. With one thread, L1 only buffers lines evicted from L0 |
| **L2** | 3 MiB | `-t 1`: current image's input + outputs of about the last 4 images. `-t 8`: both of this core's images (1.4 MiB) | `-t 8`: all of it, reused next rep |
| **L3** | 36 MiB | the whole problem: 3 MiB in + 8 MiB out | all of it next rep (`-t 1`) |
| DRAM | n/a | the master copy | touched only on a cold run |

The live working set is about 9.4 KiB, out of 48 KiB in L0 and about 3.2 MiB of cache per core (48 KiB + 192 KiB + 3 MiB). Everything else in the caches is old data that has not been evicted yet.

### C7. The totals per rep (all 16 images, simulated)

From `tools/conv_traffic_sim.py 1 2` and `… 8 2` (write-back, write-allocate, LRU, Lion Cove sizes):

| Boundary | `-t 1`, first rep, cold | `-t 1`, warm rep | `-t 8`, warm rep |
|---|---|---|---|
| L0 → registers | 2,713,728 broadcasts + 605,088 weight vectors | same | same (split over 8 cores) |
| registers → L0 | 262,144 stores × 32 B = 8 MiB | same | same |
| L1 ↔ L0 | 11 MiB up / 8 MiB down | 11 / 8 MiB | 11 / 8 MiB |
| L2 ↔ L1 | 11 / 8 MiB | 11 / 8 MiB | 11 / 8 MiB |
| L3 ↔ L2 | 11 / 5.75 MiB (the rest is still in L2) | 11 / 8 MiB | **0 / 0** |
| DRAM ↔ L3 | 11 MiB / 0 (dirty lines stay in L3) | **0 / 0** | **0 / 0** |

**Rates at the FMA limit** (`-t 1`: 3.50 M cycles per rep, 0.946 ms at 3.7 GHz):

| Resource | Used per cycle | Limit | Use |
|---|---|---|---|
| FMA | 2.00 | 2 | **100%, the bottleneck** |
| loads (as compiled) | 0.95 | 3 | 32% (source intent `WREG=0`: 2.77, 92%) |
| stores | 0.075 | 2 | 4% |
| L1 → L0 fill / L0 → L1 write-back | 3.3 B / 2.4 B | tens of B | low |
| L3 ↔ L2 | 5.7 B | tens of B | low |
| DRAM, cold, `-t 1` | 12.2 GB/s read | ≈102 GB/s | 12% |
| **DRAM, cold, `-t 8`** | 97 GB/s read + 71 GB/s write = **168 GB/s** | ≈102 GB/s | **over the limit, memory-bound** |

### C8. The way back, in detail

1. The FMA writes its result into the accumulator register.
2. After pass 9, `vmovaps` puts 32 B into the **store buffer**, and the instruction retires.
3. The store buffer writes into L0 once the core owns the line. The RFO for that line was started by the first
   store to it.
4. The line stays dirty in L0 for about 9 output rows (≈15,000 cycles), is evicted to L1, then about 37 rows later to L2,
   then to L3 (`-t 1`). It reaches DRAM only when L3 evicts it, which never happens during the timed loop.
5. The RFO can be avoided with **non-temporal stores** (`_mm256_stream_ps`): full lines go straight toward
   memory without being read first, which saves 8 MiB of reads per rep in the cold case. They also push the output
   out of the caches, which helps only if the next consumer won't read it soon. **In the warm benchmark they would
   make things worse.**

---

## Part D: what the numbers show

1. **The kernel is FMA-bound by design.** Every other resource runs below about 35% of its limit (loads 32%, everything below L0 lower).
2. **All the reuse happens in registers (×8 k, ×3 s) and L0 (×3 r).** With one thread, L1, L2 and L3 only
   carry data: each byte passes through them once per rep.
3. **The benchmark is warm.** During timing nothing comes from DRAM: 11 MiB fits in L3, and with `-t 8` each core's 1.4 MiB fits in its own L2.
   A real cold run with 8 threads would need 168 GB/s and would be DRAM-bound, at about 60% of FMA peak at best.
4. **Output traffic dominates** (8 MiB out plus 8 MiB RFO, against 3 MiB in), because K = 8 > C = 3. In a real network,
   the next step is to fuse the following op (ReLU, pooling, the next conv) before storing, use non-temporal stores
   when cold, or use a narrower output type.
5. **Use `-t 8`, not `-t 16`.** There is no SMT, so 16 threads time-share 8 cores.
6. **`WREG` makes no difference with current compilers.** gcc 13 and clang 18 already keep the 3 weight vectors in registers for each pass.

---

## Appendix: how these numbers were checked

**Compiled and run** (on this sandbox's Xeon, not on the 270K Plus, so the timing is not comparable):
`gcc -O3 -march=x86-64-v3 -ffp-contract=off -pthread conv_fast.c -lm` prints `verdict BIT-EXACT vs reference`.

**Assembly counts** for `conv_image` (gcc 13 and clang 18). The code has 9 static block bodies: top, interior
and bottom row × left, inner and tail block.

| | Expected | gcc 13 | clang 18 |
|---|---|---|---|
| `vfmadd` | 6×(35+36+23) + 9×(35+36+23) + 6×(35+36+23) = **1,974** | 1,974 | 1,974 |
| `vbroadcastss` | 6×36 + 9×36 + 6×36 = **756** | 756 | 756 |
| FMAs with a memory operand | 1,974 if every weight were folded | 28 | 0 |
| weight-vector loads | 63 passes × 3 = 189 | 196 | 189 |
| ymm spills to the stack (static stores) | 0 | 29 | 7 |

To check your own binary: `objdump -d conv_fast | grep -c vfmadd` and
`objdump -d conv_fast | grep vfmadd | grep -c '(%'`.

**Traffic simulation:** `tools/conv_traffic_sim.py` replays the kernel's exact address stream through a
write-back, write-allocate LRU hierarchy. Instruction counts printed per rep: 7,004,352 FMAs,
2,713,728 broadcasts, 605,088 weight loads, 262,144 stores. These match the formulas above.

**Check the hardware figures on the 270K Plus:**

```
lscpu -C                                                     # cache sizes and ways per level
grep . /sys/devices/system/cpu/cpu0/cache/index*/{level,type,size,ways_of_associativity}
cat /sys/devices/cpu_core/cpus                               # the P-core list read_pcores() uses
perf stat -d ./conv_fast -t 1 -g 3.7                         # cycles, instructions, L1 loads/misses, LLC
```

Depending on the kernel version, Linux may report the 48 KiB L0 as "L1d" and may not list the 192 KiB level
separately. Intel's L0/L1/L2 naming is the one used in this document.
