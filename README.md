# Substrata

A cellular automaton whose cells learn, form collectives, and evolve how they learn.

Three layers on one substrate:

1. **Cells** learn within a lifetime (a small neural network per cell, built from a genome, with a genome-encoded plasticity rule)
2. **Collectives** coordinate through bonds, sharing energy and feeding group state back down as expression context
3. **Evolution** reshapes how learning works, through reproduction, mutation and selection driven by scarcity

Scarcity comes from the real limits of the substrate: space, compute energy, memory and bandwidth.

See [docs/design-principles.md](docs/design-principles.md) for the full design and the assumption register.

## Setup (Windows, NVIDIA GPU)

```powershell
cd C:\Users\steve\dev\substrata
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The last line should print `True` and the name of the GPU.

## Benchmark

Measures ticks per second and GPU memory across grid size, per-cell network size and message width.

```powershell
python bench\bench_tick.py --quick   # smoke test, under a minute
python bench\bench_tick.py           # full sweep, writes bench\results.csv
```

## Layout

```
bench/   performance benchmarks
docs/    design principles and notes
```
