# Substrata: Design Principles (outline v0.3)

*Origin: the original Stratum concept, a cellular automaton whose cells learn and evolve their own rules.*

---

## 1. Purpose

- Explore what emerges from a network of small learning agents arranged as a cellular automaton, with layers of organization above the cells
- Core question: do higher layers produce phenomena the flatter versions cannot?
- Downstream directions, to be chosen by results: emergent physics, game dynamics, engineering applications, new learning architectures

## 2. Guiding principle

- Fix the physics of the substrate, not the structure built on it
- Every fixed choice is an assumption; name it, make it a switch, and state how to test it
- Capabilities can exist but start disabled, so we learn the basics before adding freedom

## 3. Substrate

- **Topology:** fixed 2D grid; rewiring capability built in, disabled at start
- **Resources** (all real limits of a digital substrate, nothing arbitrary):
  - Space: grid sites, one cell per site
  - Energy: compute; each cell pays per tick in proportion to its active parameters
  - Memory: hidden state and network size carry a cost
  - Bandwidth: messages between cells cost energy and/or have capped width
- **Energy inflow:** a variable field, set uniform at start; spatial or temporal variation is a change of values, not of architecture
- **Energy transfer:** cells can pass energy to neighbors
- **Abstract material resource:** not included; possible later as a disabled field for comparison

## 4. Cell (Layer 1: learns within a lifetime)

- Genome generates the cell's neural network
- Fixed maximum network per cell; genome masks which parts are active (GPU friendly; energy cost = active parameters)
- **Expression:** context dependent; local signals and group state shape what the genome builds
- **Heritable expression marks:** built in, disabled at start
- **Plasticity:** genome encodes the rule for how weights change during life

## 5. Collectives (Layer 2: coordinates across a collective)

- Scaffold: bonds between neighboring cells; the rule for forming bonds lives in the genome
- Bonded group shares an energy pool
- Group state pooled from members, fed back down as expression context (the downward channel)
- Bonded groups provide the macro variable for causal emergence measurement

## 6. Evolution (Layer 3: evolves how learning works)

- Within a world: continuous turnover; surplus energy leads to reproduction into an empty site with a mutated genome, zero energy leads to death
- Across worlds: generational sweeps for experiments only
- Selection comes from scarcity, not a written fitness function
- Functional targets are applied as environmental changes, not scores

## 7. Layer switches and comparisons

- Flat, two-layer, three-layer runs on the identical substrate
- Each layer must switch off without changing anything else
- Layer 3 controls are a ladder, climbed in order toward Layer 3 fully on:
  - **B. Single clone, no mutation:** the "off" baseline; Layers 1 and 2 only. The Layer 3 on run starts from the same single genome
  - **C. Learning genes frozen:** plasticity and expression rules fixed, everything else evolves. Tests the thesis that evolving how learning works adds something beyond evolving behavior
  - **E. Neutral drift:** mutation on, selection removed. Tests whether evolution is doing real work or random variation would produce the same result
- **Seed learning rule:** random, with a time-boxed trial; if nothing emerges within the limit, design a seed rule (e.g., Hebbian)
- **Random seed trial success criteria:**
  - Floor (required): alive and not trivial; no extinction, no frozen monoculture, turnover continues
  - Success: the floor plus either (a) learning does something, with plasticity on harvesting more energy than plasticity frozen, or (b) structure beyond chance, with bonded groups persisting longer than in a shuffled control
  - Real evolution (lineages beyond the neutral drift control) is deferred to the Layer 3 runs
- **Time limit:** counted in generations, not hours; roughly 500 to 1,000 generations, translated to wall time from the benchmark

## 8. Metrics

- **Growth:** assembly-theory-style measure (approximate assembly index weighted by copy number)
- **Emergence:** causal emergence of bonded groups vs. cells
- **Function:** adaptation speed after an inflow shift; recovery after damage; application-specific targets later
- *Open: what counts as an "object" for the assembly measure*

## 9. Platform

- Python + PyTorch, CUDA on the RTX 4070 Super (12 GB)
- Batched per-cell networks with masks
- Starting scale to be set by benchmark (expect 128 to 256 square grid)

## 10. Observation

- Live grid view, cells colored by a selectable channel (energy, lineage, bond group, active parameters, message traffic, expression state)
- Bond overlay
- Metric charts over time
- Recording with scrub-back, so a moment of emergence can be found and replayed
- **Version 1 tooling:** Rerun
- Engine emits through a single observer interface, so a browser dashboard can plug into the same stream later
- **Planned:** custom browser dashboard served from the SkyTech, viewable on the local network (e.g., from the Surface)

## 11. Assumption register

| Assumption | Default | Switch | How to test |
|---|---|---|---|
| Fixed topology | on | rewiring | compare emergence with rewiring enabled |
| Uniform inflow | uniform | inflow field values | spatial / seasonal inflow runs |
| Context-dependent expression | on | expression mode | fixed expression vs. context |
| Heritable marks | off | marks on | compare adaptation speed |
| Bond scaffold | on | Layer 2 off | flat vs. two-layer |
| Plasticity evolves | on | Layer 3 off | two-layer vs. three-layer |
| Energy transfer | on | transfer off | cooperation and collective formation rates |
| No abstract material | off | material field | material vs. none |
| Random seed learning rule | random | designed seed rule | time-boxed trial; switch to designed rule if nothing emerges |

## 12. Open questions

- Exact generation count for the random seed trial (after benchmark)
- Definition of an "object" for assembly
- Maximum network size per cell; message width (set by GPU benchmark)
- Mutation rates and what parts of the genome mutate
