#!/usr/bin/env python3
"""Count instructions and simulate cache-line traffic for conv_fast.c (UW=12, nChw8c output).

Replays the exact load/store address stream of the kernel (one broadcast per input column,
3 weight-vector loads per (c, r) pass as the compilers emit it, 12 or 8 ymm stores per block)
through a write-back, write-allocate, LRU cache hierarchy and reports how many bytes cross
each boundary per repetition.

Model (assumed Lion Cove P-core sizes, verify with `lscpu -C` on the target):
    L0 48 KiB 12-way | L1 192 KiB 12-way | L2 3 MiB 12-way per core | L3 36 MiB 12-way shared
Each level only sees the misses and dirty evictions of the level above it. Hardware
prefetchers are not modelled: they change *when* a line arrives, not *how many* lines move.

usage:  python3 tools/conv_traffic_sim.py THREADS REPS        e.g.  1 2   or   8 2
        python3 tools/conv_traffic_sim.py snapshot ROW         contents of L0/L1 after output row ROW
"""
import sys
from collections import OrderedDict

N, C, H, W, K, R, S = 16, 3, 128, 128, 8, 3, 3
LINE = 64
IN_BASE = 0x0000000     # big_alloc: 2 MiB aligned, 3 MiB input rounded up to 4 MiB
OUT_BASE = 0x0400000    # next 2 MiB aligned block
WV_BASE = 0x1000000     # packed weights wv: 64 B aligned, 864 B
BLOCKS = [(0, 12, 1, 0)] + [(12 * b, 12, 0, 0) for b in range(1, 10)] + [(120, 8, 0, 1)]  # q0, NW, LEFT, RIGHT


class Cache:
    def __init__(self, name, size, ways, nxt=None):
        self.name, self.ways, self.nxt = name, ways, nxt
        self.nsets = size // LINE // ways
        self.sets = [OrderedDict() for _ in range(self.nsets)]
        self.fill = 0   # lines fetched from the next level (demand misses + RFOs)
        self.wb = 0     # dirty lines written back to the next level

    def _insert(self, line, dirty):
        st = self.sets[line % self.nsets]
        if len(st) >= self.ways:
            victim, vdirty = st.popitem(last=False)
            if vdirty:
                self.wb += 1
                if self.nxt:
                    self.nxt.writeback(victim)
        st[line] = dirty

    def access(self, line, write):
        st = self.sets[line % self.nsets]
        if line in st:
            st.move_to_end(line)
            if write:
                st[line] = True
            return
        self.fill += 1
        if self.nxt:
            self.nxt.access(line, False)    # a store miss is a read-for-ownership from below
        self._insert(line, write)

    def writeback(self, line):              # a full dirty line arriving from above: no fetch
        st = self.sets[line % self.nsets]
        if line in st:
            st.move_to_end(line)
            st[line] = True
            return
        self._insert(line, True)


def image_stream(n, cnt):
    """Yield (line, is_write) for image n in program order, dropping repeats of the same line."""
    last = None
    img = IN_BASE + n * C * H * W * 4
    out = OUT_BASE + n * K * H * W * 4

    def emit(addr, wr):
        nonlocal last
        key = (addr // LINE, wr)
        if key != last:
            last = key
            return [key]
        return []

    for p in range(H):
        rlo, rhi = (1 if p == 0 else 0), (1 if p == H - 1 else 2)
        for q0, nw, left, right in BLOCKS:
            for c in range(C):
                for r in range(rlo, rhi + 1):
                    ih = p - 1 + r
                    for s in range(S):
                        cnt['wload'] += 1
                        yield from emit(WV_BASE + ((c * R + r) * S + s) * 32, False)
                    for t in range(nw + 2):
                        if (left and t == 0) or (right and t == nw + 1):
                            continue
                        cnt['bcast'] += 1
                        cnt['fma'] += sum(1 for d in (2, 1, 0) if 0 <= t - d < nw)
                        yield from emit(img + ((c * H + ih) * W + q0 - 1 + t) * 4, False)
            for j in range(nw):
                cnt['store'] += 1
                yield from emit(out + ((p * W + q0 + j) * K) * 4, True)


def run(threads, reps):
    ncores = min(threads, 8)                # 8 P-cores, no SMT
    l3 = Cache('L3', 36 << 20, 12)
    cores = []
    for _ in range(ncores):
        l2 = Cache('L2', 3 << 20, 12, l3)
        l1 = Cache('L1', 192 << 10, 12, l2)
        cores.append((Cache('L0', 48 << 10, 12, l1), l1, l2))
    for rep in range(reps):
        before = [[(lv.fill, lv.wb) for lv in core] for core in cores] + [[(l3.fill, l3.wb)]]
        cnt = {'wload': 0, 'bcast': 0, 'fma': 0, 'store': 0}
        for t in range(ncores):
            for n in range(N * t // ncores, N * (t + 1) // ncores):
                for line, wr in image_stream(n, cnt):
                    cores[t][0].access(line, wr)
        fill = [0, 0, 0]
        wb = [0, 0, 0]
        for i, core in enumerate(cores):
            for lvl, lv in enumerate(core):
                fill[lvl] += lv.fill - before[i][lvl][0]
                wb[lvl] += lv.wb - before[i][lvl][1]
        kib = lambda x: f"{x * LINE / 1024:9.1f} KiB"
        print(f"threads={threads} rep {rep}: {cnt}")
        print(f"  L1   -> L0  {kib(fill[0])}   L0 -> L1    {kib(wb[0])}")
        print(f"  L2   -> L1  {kib(fill[1])}   L1 -> L2    {kib(wb[1])}")
        print(f"  L3   -> L2  {kib(fill[2])}   L2 -> L3    {kib(wb[2])}")
        print(f"  DRAM -> L3  {kib(l3.fill - before[-1][0][0])}   L3 -> DRAM  {kib(l3.wb - before[-1][0][1])}")


def snapshot(stop_row):
    l2 = Cache('L2', 3 << 20, 12)
    l1 = Cache('L1', 192 << 10, 12, l2)
    l0 = Cache('L0', 48 << 10, 12, l1)
    cnt = {'wload': 0, 'bcast': 0, 'fma': 0, 'store': 0}
    written = set()
    for line, wr in image_stream(0, cnt):
        l0.access(line, wr)
        if wr:
            written.add(line)
        if len(written) == (stop_row + 1) * 64:     # 64 output lines per row
            break
    for lv in (l0, l1):
        wts, ins, outs = 0, {}, {}
        for st in lv.sets:
            for line in st:
                a = line * LINE
                if WV_BASE <= a < WV_BASE + 1024:
                    wts += 1
                elif a < OUT_BASE:
                    row = (a - IN_BASE) % (H * W * 4) // (W * 4)
                    ins[row] = ins.get(row, 0) + 1
                else:
                    row = (a - OUT_BASE) % (K * H * W * 4) // (W * K * 4)
                    outs[row] = outs.get(row, 0) + 1
        total = wts + sum(ins.values()) + sum(outs.values())
        print(f"after output row {stop_row}: {lv.name} holds {total} lines = {total * LINE / 1024:.1f} KiB")
        print(f"  weight lines: {wts}")
        print(f"  input rows  (row: lines, 3 channels): {sorted(ins.items())}")
        print(f"  output rows (row: lines): {sorted(outs.items())}")


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == 'snapshot':
        snapshot(int(sys.argv[2]))
    elif len(sys.argv) == 3:
        run(int(sys.argv[1]), int(sys.argv[2]))
    else:
        sys.exit(__doc__)
