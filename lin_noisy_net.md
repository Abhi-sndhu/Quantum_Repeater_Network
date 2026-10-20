# `Quantum Repeater Simulation` — Documentation

## Overview

In the package `qrnetwork`, there are three core modules `ent_manager.py`(Entanglement Management), `network.py` (Network Management) and `linear_noisy_sim.py`(Main Simulator Kernel).
1. `ent_manager.py` manages Entanglement Generation `(EntGenSource)` and Entanglement Swapping `(RepSwap)`, which takes care of Memory dephasing noise, Dark counts and Imperfect BSM `(Noise)`.
2. `network.py` manages linear chain topology `(RepeaterChain)`.
3. `linear_noisy_sim.py` manages scheduling `(Scheduler)` and resolving events `(RepeaterSimulation)`.
4. EntGenSource, RepSwap, RepeaterChain talks with Scheduler, which schedules source and swap events.
5. `RepeaterSimulation(config).run` starts the loop of resolving events and scheduling events till we get our desired link.
6. `example_QR.py` is an example script for the **discrete-event, noisy quantum repeater chain simulation**.


## Script Walkthrough
## `example_QR.py`
1. **`Configuration:`** SimulationConfig object (config) is created, describing the simulation parameters.
2. **`Initialization:`** RepeaterSimulation is constructed from config and `run()` is called, which executes the discrete-event simulation until the requested number of end-to-end entangled pairs has been generated.
```python
config = SimulationConfig(
    # Nodes
    rep_nodes=3,                   # 3 repeaters + 2end nodes -> total qubits = 3*2 + 2 = 8
    rep_spacing_km=30.0,
    # Source
    conversion_efficiency=1e-8,
    source_state_fidelity=0.99,
    # Laser
    laser_power_avg_mW=0.04,
    laser_rep_rate_MHz=100,
    # Detector
    detector_rep_rate_MHz=25,     # laser rate / N for synchronization
    dark_count_rate_hz=100,
    detection_window_s=8e-9,      # time for which detector is on,   # takes care of uncertaintity in arrival time
    # Memory
    dephasing_param_s=0.01,
    # Imperfect BSM
    bsm_fidelity=0.99,
    # Entanglement Swapping Policy
    swap_policy="asap",
    # Seed for reproducibility
    seed=102,
    )

result = RepeaterSimulation(config).run()
print(vars(result))
```
---

## `SimulationConfig`

*Defined in `linear_noisy_sim.py` — a frozen dataclass holding every physical/statistical parameter of the simulation.*\
We can change any of the below attributes while configuration.

| Category | Attribute | Meaning |
|---|---| --- |
|`Nodes` | rep_nodes  | Number of repeater nodes. |
| | rep_spacing_km | Distance between neighbouring nodes (km) |
|`Fiber` |  attenuation_length_km | Fibre attenuation length |
| | fiber_speed_km_per_s | Signal propagation speed in the fibre |
| `Source` | source_efficiency | Probability of getting one entangled pair per photon pump. |
| | source_state_fidelity | Fidelity with which source produces werner state. |
| `Laser` | laser_pulsed | True for pulsed laser and False for continuous laser |
| | laser_power_avg_mW | Average power of laser used to pump source in mW. |
| | laser_rep_rate_MHz | Repitition rate of pulsed laser. |
| | laser_wavelength_nm | Wavelength of laser. |
| `Detector` | detector_rep_rate_MHz | Reptition rate of gated detector. Synchronized with repition rate of pulsed laser. |
| | dark_count_rate_hz | Detector dark-count rate. |
| | detection_window_s | Time for which detector stays ON. |
| `Memory` | dephasing_param_s | Characteristic memory dephasing parameter (T_dp). |
| `Imperfect BSM` | bsm_fidelity | May consider imperfect beam splitters or low HOM visibility in this. |
| `Swapping Policy` | swap_policy | Three swapping policies: Swap at last ("last"), Nesting ("nesting"), Swap asap ("asap") |
| `Multiplexing` | n_parallel_slots | Number of parallel memory slots. |
|  | n_target_links | Required number of end-to-end entangled links. |
| `Seed` | seed | Seed for the random generator (None → nondeterministic). |

The configuration validates its inputs in `__post_init__` and exposes derived properties used internally by the simulation:

- **`pairgen_prob_pertrial`** — Probability of getting one entangled pair per pulse/trial.
- **`trial_time_s`** — Depends on the clock rate of detector.
- **`dark_count_probability`** — dark_count_rate_hz * detection_window_s.
- **`channel_efficiency`** — photon survival probability when it travels from source to repeater node:
  `exp(-rep_spacing_km / (2 * attenuation_length_km))`.
- **`swapping_time_s`** — Time taken by photon to travel and classical communication: `rep_spacing_km / fibre_speed_km_per_s`.

## `RepeaterSimulation`

*Defined in `linear_noisy_sim.py` — the main simulation engine that `example_QR.py` drives.*

### Initialization (`__init__`)

Given a `SimulationConfig`, the constructor wires up all the pieces needed to run the discrete-event simulation:

- **`self.chain`** — a **`RepeaterChain`** (from `network.py`) modelling the physical topology: a list of `Node`s connected by `MemorySlot`s, built from `config.rep_nodes` and `config.n_parallel_slots`.
- **`ent_gen_noise, swap_noise`** — two `Noise` objects computed via `Noise.calculate_stats(...)`, one for source-link generation and one for the swapping at repeater.
- **`self.source`** — an **`EntGenSource`**: Models generation of elementary link pair (including the random wait time until a heralded success).
- **`self.swapper`** — an **`RepSwap`**: Models an entanglement-swap attempt at a repeater node (including dephasing accumulated by each half-pair while it waited).
- **`self.scheduler`** — a **`Scheduler`** object that ties the chain, source, and swapper together and decides when to *start* a source attempt or a swap attempt.
- **`self.queue`** — an **`EventQueue`**, a priority queue (min-heap) of future `Event`s (`SOURCE_COMPLETE` or `SWAP_RESOLVE`), ordered by simulated time.
- **`self.now`** — the current simulation time, starting at `0.0`.

### `run()`

Executes the discrete-event loop until either the event queue empties or `n_target_pairs` end-to-end pairs have been completed:

1. **Trigger initial source events**  
   `schedule_source_attempts` schedules entanglement-generation attempts on links whose memory slots are free and schedules `SOURCE_COMPLETE` events.

2. **Resolve the earliest event**

   The event queue is ordered by simulated time. The simulator pops the earliest event and advances `self.now`.

   - **`SOURCE_COMPLETE`**  
     The elementary link pair is stored in its two memory slots and both slots become `READY`.

   - **`SWAP_RESOLVE`**  
     The simulator resolves a previously started swap attempt.
     - Inner memories are emptied.
     - If Swap fails, outer memories are also emptied.
     - On success, a new longer-distance pair is created.
     - If new pair spans the complete chain, fidelity is computed against the ideal `Φ⁺` Bell state.
     - Otherwise, the new pair is stored in the outer memory slots, which are marked `READY` for next swaps.

3. **Swapping Policies**

   After resolving each event, the simulator calls the Scheduler which schedules more events:
   - `Swap at last:` Swap events are scheduled only when all the links are available.
   - `Nesting:` If there are 2^k links, swapping can be done such that elementary link of length L goes to 2L first, then two 2L goes to 4L and so on.
   - `Swap Asap:` Swap event is scheduled immediately when both neighbouring links are available.

4. **Calculate final statistics**

   When the simulation stops, it computes:

   - **`average_fidelity`** — mean fidelity of all completed links (if n_target_links > 1).
   - **`entanglement_rate`** — end to end links generated per second.

5. **Return the result**

   `run()` returns a **`SimulationResult`** containing all the required results of the simulation.