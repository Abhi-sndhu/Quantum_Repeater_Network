# `Linear Chain Simulation` — Documentation

## Overview

`Linear_run.py` is a driver script for the **discrete-event quantum repeater chain simulation** implemented in `sim_linear.py` (and its supporting module `network.py`), part of the `qrnetwork` package. It builds a repeater chain out of a configurable number of nodes and memories, runs the event-driven simulation until a target number of end-to-end entangled pairs is produced, and prints the resulting timing and fidelity statistics.


## Script Walkthrough

```python
config = SimulationConfig(
    n_nodes=5,
    memories_per_link=1,
    link_length_km=22.0,
    source_efficiency=0.01,
    source_state_fidelity=0.99,
    dark_count_rate_hz=100,
    detection_window_s=1e-10,
    dephasing_time_s=0.001,
    trial_period_s=1e-9,
    bsm_fidelity=0.99,
    n_target_pairs=1,
    seed=None,
)

sim = RepeaterSimulation(config)
result = sim.run()
```

1. A **`SimulationConfig`** object is created, describing the physical chain (number of nodes, memories per link, link length), the hardware imperfections (source efficiency/fidelity, dark counts, dephasing, imperfect Bell-state measurements), and the stopping criterion (`n_target_pairs`).
2. A **`RepeaterSimulation`** is constructed from that config and `run()` is called, which executes the discrete-event simulation until the requested number of end-to-end entangled pairs has been generated.

---

## `SimulationConfig`

*Defined in `sim_linear.py` — a frozen dataclass holding every physical/statistical parameter of the simulation.*

| Field | Meaning |
|---|---|
| `n_nodes` | Total number of nodes in the chain (2 end nodes + repeaters in between). Must be ≥ 3. |
| `memories_per_link` | Number of quantum memory slots allocated per link. Must be ≥ 1. |
| `link_length_km` | Physical length of each link, in kilometres. |
| `attenuation_length_km` | Fibre attenuation length used to compute photon survival probability. |
| `fibre_speed_km_per_s` | Signal propagation speed in the fibre (used for classical signalling delay). |
| `source_efficiency` | Probability that a source gives an entangled pair. |
| `source_state_fidelity` | Fidelity of the raw entangled state produced by the source (1.0 = perfect Bell pair). |
| `dark_count_rate_hz` | Detector dark-count rate, used to compute false-click probability. |
| `detection_window_s` | Detection time window, combined with the dark-count rate to get a per-trial dark-count probability. |
| `dephasing_time_s` | Characteristic memory dephasing time — how quickly a stored qubit decoheres while waiting. |
| `trial_period_s` | Duration of one entanglement-generation attempt/trial. |
| `bsm_fidelity` | Fidelity of the imperfect Bell-state measurement. |
| `n_target_pairs` | Number of end-to-end entangled pairs the simulation should generate before stopping. |
| `seed` | Random seed for the run's `numpy` random generator (`None` → nondeterministic). |

It validates its inputs in `__post_init__` (e.g. rejecting `n_nodes < 3`, fidelities outside `[0, 1]`, etc.) and exposes derived, read-only properties used internally by the simulation:

- **`n_repeaters`** — `n_nodes - 2`.
- **`n_links`** — `n_nodes - 1`.
- **`dark_count_probability`** — `dark_count_rate_hz * detection_window_s`.
- **`midpoint_arrival_probability`** — photon survival probability over half a link, `exp(-link_length_km / (2 * attenuation_length_km))`.
- **`classical_signalling_time_s`** — one-way classical signalling delay across a link, `link_length_km / fibre_speed_km_per_s`, used as the resolve delay for entanglement-swap heralding.


## `RepeaterSimulation`

*Defined in `sim_linear.py` — the main simulation engine that `Linear_run.py` drives.*

### Construction (`__init__`)

Given a `SimulationConfig`, the constructor wires up all the pieces needed to run the discrete-event simulation:

- **`self.rng`** — a seeded `numpy.random.Generator` (via `config.seed`) shared by every stochastic component so the whole run is reproducible when a seed is given.
- **`self.chain`** — a **`RepeaterChain`** (from `network.py`) modelling the physical topology: a list of `Node`s connected by `MemorySlot`s, built from `config.n_nodes` and `config.memories_per_link`.
- **Heralding statistics** — two `HeraldingStatistics` objects computed via `heralding_statistics(...)`, one for source-link generation and one for the repeater-to-repeater swap stage (see above).
- **`bsm`** — an **`ImperfectBSM`** object (fidelity = `config.bsm_fidelity`) modelling a non-ideal Bell-state measurement.
- **`mem`** — a **`DephasingMemory`** object (dephasing time = `config.dephasing_time_s`) modelling decoherence of qubits while they wait in memory.
- **`self.source`** — an **`EntanglementSource`**: combines the BSM model, source-side heralding statistics, trial period, and source-state fidelity to model how an elementary link pair is generated (including the random wait time until a heralded success).
- **`self.swapper`** — an **`EntanglementSwapper`**: combines the BSM model, swap-side heralding statistics, and the memory-noise model to model an entanglement-swap attempt at a repeater node (including dephasing accumulated by each half-pair while it waited).
- **`self.scheduler`** — a **`Scheduler`** object that ties the chain, source, and swapper together and knows how to *start* a source attempt or a swap attempt at a given simulated time, pushing the corresponding completion event.
- **`self.queue`** — an **`EventQueue`**, a priority queue (min-heap) of future `Event`s (`SOURCE_COMPLETE` or `SWAP_RESOLVE`), ordered by simulated time.
- **`self.now`** — the current simulated time, starting at `0.0`.

### `run()`

Executes the discrete-event loop until either the event queue empties or `config.n_target_pairs` end-to-end pairs have been completed:

1. **Trigger initial link attempts** — `_trigger_source_attempts()` starts an entanglement-generation attempt on every link whose memory slots are free, scheduling a `SOURCE_COMPLETE` event for each.
2. **Main loop** — repeatedly pops the earliest event off the queue and advances `self.now` to its time:
   - **`SOURCE_COMPLETE`**: the elementary link pair produced by the source is marked `READY` in its two memory slots.
   - **`SWAP_RESOLVE`**: the outcome of a swap attempt (started earlier by the scheduler) is resolved. On success, if the new pair now spans the *entire* chain (`span_hops == n_nodes - 1`), its fidelity against the ideal Φ⁺ Bell state is computed (via `primaryfn.fidelity_to_state`) and it is recorded as a **`CompletedPair`**; otherwise the (still partial) pair's slots are marked `READY` so it can be used in a further swap. On failure, the involved slots are freed.
   - After handling each event, the loop again calls `_trigger_source_attempts()` (to refill any newly-freed link slots) and `_check_and_trigger_swaps()` (to start a swap wherever a repeater now has ready pairs on both sides).
3. Once the loop ends, it computes:
   - **`average_fidelity`** — mean fidelity over all completed pairs (0.0 if none).
   - **`entanglement_rate`** — `len(completed_pairs) / self.now` (pairs per second).
4. Returns a **`SimulationResult`** bundling `completed_pairs`, `total_sim_time` (`self.now`), `average_fidelity`, and `entanglement_rate` — exactly the fields printed by `Linear_run.py`.

### Supporting internal components (used by `RepeaterSimulation`, all from `network.py`/`sim_linear.py`)

| Component | Role |
|---|---|
| **`RepeaterChain`** | Builds and holds the chain topology: a list of `Node`s, each with left/right `MemorySlot`s connecting it to its neighbours. Provides helpers such as `repeater_indices()`, `link_left_indices()`, and `new_pair_id()`. |
| **`Node`** | A single chain node (end node or repeater) holding its left/right memory slots. |
| **`MemorySlot`** | One quantum-memory slot on a node, tracking its `SlotStatus` (`EMPTY`/`PENDING`/`READY`) and which `EntangledPair` (if any) currently occupies it. |
| **`SlotStatus`** | Enum of a memory slot's state: `EMPTY`, `PENDING` (an attempt is in flight), `READY` (holds a usable entangled pair). |
| **`EntangledPair`** | Represents one entangled pair (elementary or swapped), storing its density matrix `rho`, `ready_time`, the two memory slots it occupies, and (for swapped pairs) the list of intermediate slots it consumed. Exposes `span_hops`, the number of chain links it currently bridges. |
| **`EntanglementSource`** | Models one entanglement-generation attempt on a link: draws a random number of trials until heralded success (`rng.geometric`), computes the resulting wait time, and produces a Werner-noised Bell pair via `perform_bell_swap`. |
| **`EntanglementSwapper`** | Models one entanglement-swap attempt at a repeater: applies memory dephasing to each half-pair based on how long it has waited, then performs the (possibly failing) Bell-state measurement to merge the two pairs into a longer one. |
| **`ImperfectBSM`** | Models a Bell-state measurement with limited fidelity, mixing the ideal projective outcome with a partially-scrambled state. |
| **`DephasingMemory`** | Models T-dephasing noise accumulated by a qubit held in memory for a given wait time. |
| **`HeraldingStatistics`** / `heralding_statistics()` | Probability model for what a heralding click pattern actually tells you (correct click / wrong pair / no information), as described above. |
| **`Scheduler`** | Thin coordination layer: given the chain/source/swapper, knows how to start a source attempt (`start_source_attempt`) or a swap attempt (`start_swap`) at a given time and package the result as an `Event` payload. |
| **`EventQueue`** / **`Event`** / **`EventType`** | Minimal discrete-event simulation primitives: a time-ordered priority queue of `Event` objects (`SOURCE_COMPLETE` or `SWAP_RESOLVE`), each carrying a payload (`EntangledPair` or `SwapAttempt`). |
| **`SwapAttempt`** | Dataclass capturing the full context and outcome of one swap attempt (which slots were involved, whether it succeeded, the resulting `EntangledPair` if any, and when it resolves). |
| **`primaryfn.fidelity_to_state(rho, target)`** | Computes the fidelity `Tr(rho · target)` of a density matrix against a target state (ket or density matrix) — used to score each completed end-to-end pair against the ideal Φ⁺ Bell state. |

## `SimulationResult`

*Defined in `sim_linear.py` — the return value of `RepeaterSimulation.run()`, consumed directly by `Linear_run.py`.*

| Field | Meaning |
|---|---|
| `completed_pairs` | List of `CompletedPair` objects — one per end-to-end entangled pair generated, each with `id`, `rho`, `fidelity`, and `created_time`. |
| `total_sim_time` | Total simulated time (seconds) elapsed until the target number of pairs was reached. |
| `average_fidelity` | Mean fidelity across all completed pairs. |
| `entanglement_rate` | `completed_pairs / total_sim_time`, i.e. pairs generated per second. |

## `CompletedPair`

*Defined in `sim_linear.py`.* A simple record of one successfully delivered end-to-end pair: its `id`, final density matrix `rho`, computed `fidelity` (against the ideal Bell state), and the simulated `created_time` at which it was completed. `Linear_run.py` iterates over `result.completed_pairs` to print each one.

---

## Summary of Data Flow

```
SimulationConfig
      │
      ▼
RepeaterSimulation.__init__
      │  builds: RepeaterChain, EntanglementSource, EntanglementSwapper,
      │          ImperfectBSM, DephasingMemory, Scheduler, EventQueue
      ▼
RepeaterSimulation.run()
      │  loop: SOURCE_COMPLETE / SWAP_RESOLVE events
      │        → triggers new source attempts & swap attempts
      │        → scores completed end-to-end pairs vs. Φ⁺ (fidelity_to_state)
      ▼
SimulationResult (completed_pairs, total_sim_time, average_fidelity, entanglement_rate)
      │
      ▼
Linear_run.py prints results.
```
