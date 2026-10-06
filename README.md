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

## Run

```powershell
python -m substrata.run --config configs/trial_a.toml
```

This opens the Rerun viewer: grid views on the left (energy, genome, age, active parameters,
messages, transfers, generation), metric charts on the right, and a timeline you can scrub back.
Empty sites are dark gray. In the genome view, similar genomes have similar colors; a clone
population is uniform gray until mutations spread.

Useful variations:

```powershell
# change any config value without editing the file
python -m substrata.run --config configs/trial_a.toml --set layers.plasticity=false --name trial_a_noplast

# Layer 3 control ladder
python -m substrata.run --config configs/trial_a.toml --set layers.layer3=off --name ladder_b
python -m substrata.run --config configs/trial_a.toml --set layers.layer3=frozen_learning --name ladder_c

# watch from another device on the network (e.g. the Surface): open http://<skytech-ip>:9090
python -m substrata.run --config configs/trial_a.toml --rerun serve

# record to a file instead of a live viewer (open later with: rerun runs\<run>\recording.rrd)
python -m substrata.run --config configs/trial_a.toml --rerun save

# continue a stopped run
python -m substrata.run --resume runs\<run>\checkpoint.pt
```

Each run writes `runs/<name>_<time>/` with `config.json`, `metrics.csv` and `checkpoint.pt`.
Ctrl-C stops cleanly and saves a checkpoint.

## Tests

```powershell
python tests\test_world.py
```

## Layout

```
substrata/   engine: config, grid, genome, world, metrics, observers, run
configs/     run configurations (trial_a, smoke)
bench/       performance benchmark and results
docs/        design principles and notes
tests/       engine sanity checks
```
