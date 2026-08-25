# `Quantum Repeater Simulation` — Documentation

## Overview

`example_QR.py` is a driver script for the **discrete-event quantum repeater chain simulation** implemented in `qsimulator.py` (and its supporting module `network.py`), part of the `qrnetwork` package. It builds a repeater chain out of a configurable number of nodes and memories, runs the event-driven simulation until a target number of end-to-end entangled pairs is produced, and prints the resulting timing and fidelity statistics.


## Script Walkthrough

```python
config = SimulationConfig(
    rep_nodes=3,
    rep_spacing_km=30.0,

    pairgen_prob_perpulse=0.01,
    prep_time_s=1e-8,
    source_state_fidelity=0.99,

    dark_count_rate_hz=100,
    detection_window_s=1e-9,
    dephasing_time_s=0.01,

    bsm_fidelity=0.99, 
)

sim = RepeaterSimulation(config)
result = sim.run()
```

1. A **`SimulationConfig`** object is created, describing the physical chain and simulation parameters.
2. A **`RepeaterSimulation`** is constructed from that config and `run()` is called, which executes the discrete-event simulation until the requested number of end-to-end entangled pairs has been generated.

---

## `SimulationConfig`

*Defined in `qsimulator.py` — a frozen dataclass holding every physical/statistical parameter of the simulation.*

| Field | Meaning |
|---|---|
| `rep_nodes` | Number of repeater nodes. The chain also contains two end nodes, so total nodes = `rep_nodes + 2`. |
| `rep_spacing_km` | Distance between neighbouring nodes, in kilometres. |
| `attenuation_length_km` | Fibre attenuation length used to compute photon survival probability. |
| `fibre_speed_km_per_s` | Signal propagation speed in the fibre, used for swapping/communication delay. |
| `pairgen_prob_perpulse` | Probability of getting one entangled pair per pulse, used for elementary-link heralding/source generation. |
| `prep_time_s` | Time associated with one source-generation attempt/trial. |
| `source_state_fidelity` | Fidelity range used when generating the initial noisy entangled state. |
| `dark_count_rate_hz` | Detector dark-count rate. |
| `detection_window_s` | Detector time window, used with the dark-count rate to compute per-trial dark-count probability. |
| `dephasing_time_s` | Characteristic memory dephasing time. |
| `bsm_fidelity` | Fidelity of the imperfect Bell-state measurement. |
| `n_parallel_slots` | Number of parallel memory slots. Used for spatial-multiplexing. |
| `n_target_pairs` | Number of end-to-end entangled pairs to generate before stopping. |
| `seed` | Random seed for the simulation's NumPy random generator (`None` → nondeterministic). |

The configuration validates its inputs in `__post_init__` and exposes derived properties used internally by the simulation:

- **`total_nodes`** — `rep_nodes + 2`.
- **`n_links`** — `rep_nodes + 1`.
- **`dark_count_probability`** — `dark_count_rate_hz * detection_window_s`.
- **`channel_efficiency`** — photon survival probability when it travels from source to repeater node:
  `exp(-rep_spacing_km / (2 * attenuation_length_km))`.
- **`swapping_time_s`** — `rep_spacing_km / fibre_speed_km_per_s`, used as the swap resolve delay.

## `RepeaterSimulation`

*Defined in `qsimulator.py` — the main simulation engine that `example_QR.py` drives.*

### Construction (`__init__`)

Given a `SimulationConfig`, the constructor wires up all the pieces needed to run the discrete-event simulation:

- **`self.rng`** — a seeded `numpy.random.Generator` (via `config.seed`) shared by every stochastic component so the whole run is reproducible when a seed is given.
- **`self.chain`** — a **`RepeaterChain`** (from `network.py`) modelling the physical topology: a list of `Node`s connected by `MemorySlot`s, built from `config.rep_nodes` and `config.n_parallel_slots`.
- **`source_heralding, swap_heralding`** — two `DarkHeralding` objects computed via `DarkHeralding.calculate(...)`, one for source-link generation and one for the repeater-to-repeater swap stage.
- **`bsm`** — an **`ImperfectBSM`** object (fidelity = `config.bsm_fidelity`) modelling a non-ideal Bell-state measurement.
- **`mem`** — a **`DephasingMemory`** object (dephasing time = `config.dephasing_time_s`) modelling decoherence of qubits while they wait in memory.
- **`self.source`** — an **`EntanglementSource`**: combines the BSM model, source-side heralding statistics, trial period, and source-state fidelity to model how an elementary link pair is generated (including the random wait time until a heralded success).
- **`self.swapper`** — an **`EntanglementSwapper`**: combines the BSM model, swap-side heralding statistics, and the memory-noise model to model an entanglement-swap attempt at a repeater node (including dephasing accumulated by each half-pair while it waited).
- **`self.scheduler`** — a **`Scheduler`** object that ties the chain, source, and swapper together and knows how to *start* a source attempt or a swap attempt at a given simulated time, pushing the corresponding completion event.
- **`self.queue`** — an **`EventQueue`**, a priority queue (min-heap) of future `Event`s (`SOURCE_COMPLETE` or `SWAP_RESOLVE`), ordered by simulated time.
- **`self.now`** — the current simulated time, starting at `0.0`.

### `run()`

Executes the discrete-event loop until either the event queue empties or `config.n_target_pairs` end-to-end pairs have been completed:

1. **Trigger initial link attempts**  
   `_trigger_source_attempts()` starts entanglement-generation attempts on links whose memory slots are free and schedules `SOURCE_COMPLETE` events.

2. **Process the earliest event**

   The event queue is ordered by simulated time. The simulator pops the earliest event and advances `self.now`.

   - **`SOURCE_COMPLETE`**  
     The elementary link pair is stored in its two memory slots and both slots become `READY`.

   - **`SWAP_RESOLVE`**  
     The simulator resolves a previously started swap attempt.

     - On success, a new longer-distance pair is created.
     - If `new_pair.span_hops == self.config.total_nodes - 1`, the pair spans the complete chain. Its fidelity against the ideal `Φ⁺` Bell state is calculated and the pair is stored as a `CompletedPair`.
     - Otherwise, the new pair is stored in the outer memory slots as `READY` for a later swap.
     - On failure, the involved slots are released.

3. **Schedule new work**

   After each event, the simulator:

   - calls `_trigger_source_attempts()` to refill available elementary-link slots;
   - calls `_check_and_trigger_swaps()` to start swaps whenever a repeater has ready pairs on both sides.

4. **Calculate final statistics**

   When the simulation stops, it computes:

   - **`average_fidelity`** — mean fidelity of all completed pairs, or `0.0` if none were produced.
   - **`entanglement_rate`** — `len(completed_pairs) / self.now`, in pairs per second.

5. **Return the result**

   `run()` returns a **`SimulationResult`** containing the completed pairs and the aggregate simulation statistics.

### Supporting internal components (used by `RepeaterSimulation`, all from `network.py`/`qsimulator.py`)

| Component | Role |
|---|---|
| **`RepeaterChain`** | Builds and holds the chain topology: `Node`s connected by left/right `MemorySlot`s. It provides helpers such as `repeater_indices()`, `links_indices()`, and `new_pair_id()`. |
| **`Node`** | Represents one chain node, either an end node or repeater, and stores its left and right memory slots. |
| **`MemorySlot`** | Represents one quantum-memory slot. It tracks its `SlotStatus` and the `EntangledPair` currently stored in it. |
| **`SlotStatus`** | Enum with `EMPTY`, `PENDING`, and `READY` states. |
| **`EntangledPair`** | Represents an elementary or swapped entangled pair. It stores `rho`, `ready_time`, the two memory slots, and exposes `left_node`, `right_node`, and `span_hops`. |
| **`EntanglementSource`** | Models elementary-link generation. It samples a random number of trials until heralded success and produces the resulting noisy entangled-pair state. |
| **`EntanglementSwapper`** | Models an entanglement-swap attempt. It applies memory dephasing based on waiting times and then performs the Bell-state measurement. |
| **`ImperfectBSM`** | Models a non-ideal Bell-state measurement using the configured BSM fidelity. |
| **`DephasingMemory`** | Models dephasing accumulated while qubits remain in memory. |
| **`DarkHeralding`** | Calculates and stores heralding probabilities and applies the corresponding dark-count/heralding mixture to the density matrix. |
| **`BellSwapResult`** | Stores the density matrix produced by the Bell-state swapping operation. |
| **`Scheduler`** | Coordinates source and swap attempts and creates the corresponding future events. |
| **`EventQueue` / `Event` / `EventType`** | Provide the discrete-event simulation mechanism: a time-ordered queue containing `SOURCE_COMPLETE` and `SWAP_RESOLVE` events. |
| **`SwapAttempt`** | Stores the information required to resolve a swap later, including the involved memory slots, success status, resolve time, and resulting pair. |
| **`primaryfn.Qp(...).fid()`** | Computes the fidelity used to evaluate the final end-to-end state against the target `Φ⁺` Bell state. |

## `SimulationResult`

*Defined in `qsimulator.py` — the return value of `RepeaterSimulation.run()`, consumed directly by `example_QR.py`.*

| Field | Meaning |
|---|---|
| `completed_pairs` | List of `CompletedPair` objects — one per end-to-end entangled pair generated, each with `id`, `rho`, `fidelity`, and `created_time`. |
| `total_sim_time` | Total simulated time (seconds) elapsed until the target number of pairs was reached. |
| `average_fidelity` | Mean fidelity across all completed pairs. |
| `entanglement_rate` | `completed_pairs / total_sim_time`, i.e. pairs generated per second. |

## `CompletedPair`

*Defined in `qsimulator.py`.* A simple record of one successfully delivered end-to-end pair: its `id`, final density matrix `rho`, computed `fidelity` (against the ideal Bell state), and the simulated `created_time` at which it was completed. `example_QR.py` iterates over `result.completed_pairs` to print each one.

---

## Summary of Data Flow

```text
SimulationConfig
      │
      ▼
RepeaterSimulation.__init__
      │
      ├── RepeaterChain
      ├── DarkHeralding
      ├── ImperfectBSM
      ├── DephasingMemory
      ├── EntanglementSource
      ├── EntanglementSwapper
      ├── Scheduler
      └── EventQueue
      │
      ▼
RepeaterSimulation.run()
      │
      ├── start elementary-link attempts
      │
      ├── SOURCE_COMPLETE events
      │       └── store READY elementary pairs
      │
      ├── start swaps when both sides are READY
      │
      ├── SWAP_RESOLVE events
      │       ├── success and end to end pair → Memories emptied and pair stored
      │       ├── partial pair → Memories ready for another swap
      │       └── failure → Memories emptied
      |      
      │
      └── repeat until n_target_pairs is reached
      │
      ▼
SimulationResult
      │
      ├── completed_pairs
      ├── total_sim_time
      ├── average_fidelity
      └── entanglement_rate
      │
      ▼
example_QR.py prints results
```
