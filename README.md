# SCH_KER_OPT_PRAC

Practice in scheduling and optimising compute kernels.

| File | What it is |
|---|---|
| [`conv_fast.c`](conv_fast.c) | 3×3 conv, N=16 C=3 128×128 K=8, AVX2 register-blocked microkernel (UW=12, nChw8c output), bit-exact against the reference loop |
| [`docs/conv_dataflow_walkthrough.md`](docs/conv_dataflow_walkthrough.md) | The plan (output-first tiling down to one FMA), the hardware limits that fix each size, and every byte's path DRAM → L3 → L2 → L1 → L0 → registers → FMA → back |
| [`tools/conv_traffic_sim.py`](tools/conv_traffic_sim.py) | Replays the kernel's address stream through a cache model and prints instruction counts and bytes per level |

Build and run (on the target):

```
gcc -O3 -march=native -ffp-contract=off -pthread conv_fast.c -o conv_fast -lm
./conv_fast -t 1 -g 3.7        # single P-core
./conv_fast -t 8 -g 3.7        # one thread per P-core (no SMT on Arrow Lake)
python3 tools/conv_traffic_sim.py 1 2
```
