from __future__ import annotations
import heapq
import itertools
from dataclasses import dataclass, field
from enum import Enum, auto
import numpy as np

from . import Q,Qp
from .network import (Link,MemorySlot,RepeaterChain,SlotStatus)
from .ent_manager import(EntGenSource,RepSwap,Noise)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PhysicsParams:
    h=6.62607015 * 1e-34 #js
    c=299_792_458

@dataclass(frozen=True)
class SimulationConfig:
    # Nodes
    rep_nodes: int = 3
    rep_spacing_km: float = 50.0
    # Fiber
    attenuation_length_km: float = 22.0
    fiber_speed_km_per_s: float = 200_000.0
    # Source
    source_efficiency: float = 1e-8    #conversion efficiency per photon
    source_state_fidelity: float = 1.0
    # Laser
    laser_pulsed: bool = True              # False for continuous laser
    laser_power_avg_mW: float = 0.04       # 40microWatt
    laser_rep_rate_MHz: float = 100        # 10ns
    laser_wavelength_nm: float = 405
    # Detector
    detector_rep_rate_MHz: float = 25      # laser rate / N for synchronization -> clock_rate of source -> 40ns trial time 
    dark_count_rate_hz: float = 100.0
    detection_window_s: float = 8e-9       #8ns       # time for which detector is on,   # more than jitter time
    # Memory
    dephasing_param_s: float = 0.01
    # Imperfect BSM
    bsm_fidelity: float = 1.0
    # Entanglement Swapping Policy
    swap_policy: str = "asap"                  # asap/last/nesting
    # Spatial-multiplexing
    n_target_links: int = 1
    n_parallel_slots: int = 1              # Increase n_parallel_slots for spatial-multiplexing (multi-memory slots model)
    # Seed for reproducibility
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.rep_nodes < 1:
            raise ValueError("rep_nodes must be >= 1 (need at least one repeater)")
        if self.n_parallel_slots < 1:
            raise ValueError("n_parallel_slots must be >= 1")
        if not (0.0 <= self.source_state_fidelity <= 1.0):
            raise ValueError("source_state_fidelity must be in [0, 1]")
        if not (0.0 <= self.bsm_fidelity <= 1.0):
            raise ValueError("bsm_fidelity must be in [0, 1]")
        if self.n_target_links < 1:
            raise ValueError("n_target_links must be >= 1")
        if self.pairgen_prob_pertrial > 0.01:
            raise ValueError("Multi-Photons noise is too much. Reduce laser power.")
        if self.detection_window_s > 1/(self.laser_rep_rate_MHz * 1e6):
            raise ValueError("More than one pulse per detector window. Reduce detection_window_s or reduce laser_rep_rate_MHz")
        if self.swap_policy in (SwapPolicy.SWAP_AT_LAST, SwapPolicy.NESTING) and self.n_parallel_slots > 1:
            raise ValueError("This scenario is not yet defined.")
        if self.swap_policy == SwapPolicy.NESTING and ((self.rep_nodes + 1) & self.rep_nodes) != 0:
            raise ValueError("Nesting requires rep_nodes = 2^k - 1 (e.g., 1, 3, 7, 15).")

    @property
    def pairgen_prob_pertrial(self) -> float:   #0.01 # P(n)=mu^n/(1+mu)^(n+1), where n is number of pairs and
                                                # mu represents the mean number of photon pairs generated per pump pulse
                                                # P(1)= eta_source * number of photons per pulse
        Energy_photon = PhysicsParams.h * PhysicsParams.c / (self.laser_wavelength_nm * 1e-9)
        if self.laser_pulsed:
            Energy_pulse = self.laser_power_avg_mW * 1e-3 / (self.laser_rep_rate_MHz * 1e6)
            return self.source_efficiency * Energy_pulse / Energy_photon
        else:                                   # In CW laser, temporal mismatch reduces visibility -> fidelity
                                                # For good fidelity, detection window < coherence time(500fs=0.0005ns) -> reduces pairgen_prob_pertrial
                                                # Trade-off between Rate and Fidelity
            Energy_per_window = self.laser_power_avg_mW * 1e-3 * self.detection_window_s
            return self.source_efficiency * Energy_per_window / Energy_photon
    
    @property
    def trial_time_s(self) -> float:            #25ns             # trial time for source event  
                                                # depends on clock rate(At what rate EGS makes a new attempt)
        return 1/(self.detector_rep_rate_MHz*10**6)

    @property
    def dark_count_probability(self) -> float:
        return self.dark_count_rate_hz * self.detection_window_s

    @property
    def channel_efficiency(self) -> float:  # e^(-L/Latt) # In our protocol, photon travels from source to repeater, so L -> L/2
        return float(np.exp(-self.rep_spacing_km / (2 * self.attenuation_length_km)))

    @property
    def swapping_time_s(self) -> float:   #L/c: While swapping, It takes L/2c for photon to reach Rep + L/2c for communication
        return self.rep_spacing_km / self.fiber_speed_km_per_s


# ---------------------------------------------------------------------------
# Event Queue
# ---------------------------------------------------------------------------
@dataclass(order=True)
class Event:
    time: float
    sequence: int
    type: EventType = field(compare=False)
    payload: Link | SwapAttempt = field(compare=False)

class EventType(Enum):
    SOURCE_COMPLETE = auto()
    SWAP_RESOLVE = auto()

class EventQueue:
    def __init__(self) -> None:
        self._heap: list[Event] = []
        self._counter = itertools.count()

    def push(self, time: float, type: EventType, payload: Link | SwapAttempt) -> None:
        heapq.heappush(self._heap, Event(time=time, sequence=next(self._counter), type=type, payload=payload))

    def pop(self) -> Event:
        return heapq.heappop(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------
@dataclass
class SwapAttempt:
    repeater: int
    inner_left: MemorySlot
    inner_right: MemorySlot
    outer_left: MemorySlot
    outer_right: MemorySlot
    succeeded: bool
    new_link: Link | None

class Scheduler:
    def __init__(self, chain: RepeaterChain, source: EntGenSource, swapper: RepSwap, 
                 swapping_time_s: float, rng: np.random.Generator,):
        self.chain = chain
        self.source = source
        self.swapper = swapper
        self.t_swap = swapping_time_s
        self.rng = rng

    def start_source_attempt(self, left_slot: MemorySlot, right_slot: MemorySlot, now: float) -> tuple[float, Link]:
        left_slot.status = right_slot.status = SlotStatus.PENDING
        wait_s, result = self.source.attempt(self.rng)
        complete_time = now + wait_s
        link = Link(rho=result, ready_time=complete_time, left_slot=left_slot, 
                             right_slot=right_slot,)
        return complete_time, link

    def schedule_source_attempts(self, now: float, queue: EventQueue) -> None:
        for i in self.chain.links_indices(): #for i in range(n_links)
            left_node = self.chain.nodes[i]
            right_node = self.chain.nodes[i + 1]
            for l_slot, r_slot in zip(left_node.right_slots, right_node.left_slots):
                if l_slot.status == SlotStatus.EMPTY and r_slot.status == SlotStatus.EMPTY :
                    finish_t, link = self.start_source_attempt(l_slot, r_slot, now)
                    queue.push(finish_t, EventType.SOURCE_COMPLETE, link)

    def start_swap(self, repeater: int, inner_left_slot: MemorySlot, inner_right_slot: MemorySlot, now: float, det_swap: bool=False) -> tuple[float, SwapAttempt]:
        left_link, right_link = inner_left_slot.link, inner_right_slot.link
        assert left_link is not None and right_link is not None
        outer_left_slot, outer_right_slot = left_link.left_slot, right_link.right_slot

        left_wait = now - left_link.ready_time
        right_wait = now - right_link.ready_time
        for slot in (inner_left_slot, inner_right_slot, outer_left_slot, outer_right_slot):
            slot.status = SlotStatus.PENDING

        if det_swap: # For simultaneous deterministic swapping ("last")
            # This keeps deterministic swapping both hierarchical(for correct density matrix calculation) and simulataneous(no extra extra wait time)
            if repeater == self.chain.rep_nodes:
                resolve_time = now + self.t_swap  # Total time taken for all swaps = t_swap
            else:
                resolve_time = now
            result = self.swapper.det_swap(left_link.rho, right_link.rho, left_wait, right_wait, self.rng)
            new_link = Link(rho=result, ready_time=resolve_time, left_slot=outer_left_slot, 
                                                    right_slot=outer_right_slot,)
            succeeded = True
        else:
            succeeded, result = self.swapper.attempt(left_link.rho, right_link.rho, left_wait, right_wait, self.rng)
            new_link = None
            resolve_time = now + self.t_swap
            if succeeded:
                new_link = Link(rho=result, ready_time=resolve_time, left_slot=outer_left_slot, 
                                        right_slot=outer_right_slot,)

        return resolve_time, SwapAttempt(repeater=repeater, inner_left=inner_left_slot, inner_right=inner_right_slot, outer_left=outer_left_slot, 
                           outer_right=outer_right_slot, succeeded=succeeded, new_link=new_link,)

    def schedule_swap_asap(self, now: float, queue: EventQueue) -> None:
        for r_idx in self.chain.repeater_indices(): #for all repeaters
            node = self.chain.nodes[r_idx]
            for l_slot in node.left_slots:
                if l_slot.status == SlotStatus.READY and l_slot.link is not None:
                    for r_slot in node.right_slots:
                        if r_slot.status == SlotStatus.READY and r_slot.link is not None:
                            finish_t, swap_attempt = self.start_swap(r_idx, l_slot, r_slot, now)
                            queue.push(finish_t, EventType.SWAP_RESOLVE, swap_attempt)
                            break

    def schedule_swap_last(self, now: float, queue: EventQueue) -> None:
        for r_idx in self.chain.repeater_indices():
            l_slot = self.chain.nodes[r_idx].left_slots[0]   # Only for n_parallel_slots == 1
            r_slot = self.chain.nodes[r_idx].right_slots[0]
            if l_slot.status == SlotStatus.READY and l_slot.link is not None and r_slot.status == SlotStatus.READY and r_slot.link is not None:
                finish_t, swap_attempt = self.start_swap(r_idx, l_slot, r_slot, now, det_swap = True)
                queue.push(finish_t, EventType.SWAP_RESOLVE, swap_attempt)
                break                

    def schedule_swap_nesting(self, now: float, queue: EventQueue) -> None:
        for r_idx in self.chain.repeater_indices():
            node = self.chain.nodes[r_idx]
            l_slot = node.left_slots[0]   # Only for n_parallel_slots == 1
            r_slot = node.right_slots[0]
            if l_slot.status == SlotStatus.READY and l_slot.link is not None and r_slot.status == SlotStatus.READY and r_slot.link is not None:
                outer_node_left = l_slot.link.left_node
                outer_node_right = r_slot.link.right_node
                nesting_level_max = int(np.log2(self.chain.rep_nodes + 1))
                nesting_level = np.zeros((1,self.chain.rep_nodes))
                old_rep = [self.chain.rep_nodes+1]
                for n in reversed(range(nesting_level_max)):
                    new_rep = []
                    for i in old_rep:
                        new_rep.append(i-min(old_rep)/2)
                        new_rep.append(i+min(old_rep)/2)
                    for r in new_rep:
                        if r < self.chain.rep_nodes+1:
                            r= int(r)
                            nesting_level[0][r-1] = n + 1
                    old_rep = new_rep

                if outer_node_right.index - outer_node_left.index == 2**nesting_level[0][r_idx-1]:
                    finish_t, swap_attempt = self.start_swap(r_idx, l_slot, r_slot, now)
                    queue.push(finish_t, EventType.SWAP_RESOLVE, swap_attempt)


# ---------------------------------------------------------------------------
# Simulation Engine & Results
# ---------------------------------------------------------------------------
class SwapPolicy(str,Enum):
    SWAP_AT_LAST = "last"
    NESTING = "nesting"
    SWAP_ASAP = "asap"

@dataclass
class CompletedLink:
    rho: Q
    fidelity: float
    created_time: float


@dataclass
class SimulationResult:
    completed_links: list[CompletedLink]
    total_sim_time: float
    average_fidelity: float
    entanglement_rate: float


class RepeaterSimulation:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed) #defining rng for the entire run with seed managed
        self.chain = RepeaterChain(config.rep_nodes, config.n_parallel_slots)

        ent_gen_noise = Noise.calculate_stats(config.dephasing_param_s, config.bsm_fidelity,
                                             config.pairgen_prob_pertrial, config.pairgen_prob_pertrial, 
                                             config.dark_count_probability)
        swap_noise = Noise.calculate_stats(config.dephasing_param_s, config.bsm_fidelity,
                                           config.channel_efficiency, config.channel_efficiency, 
                                           config.dark_count_probability)
        self.source = EntGenSource(noise=ent_gen_noise, trial_time_s=config.trial_time_s,
                                   source_state_fidelity=config.source_state_fidelity,)
        self.swapper = RepSwap(swap_noise)
        self.swap_policy = SwapPolicy(config.swap_policy)
        self.scheduler = Scheduler(chain=self.chain, source=self.source, swapper=self.swapper, 
                                   swapping_time_s=config.swapping_time_s, rng=self.rng,)
        self.queue = EventQueue()
        self.now = 0.0

    def last_policy_failed_attempts(self) -> None:
        prob_success_swaps = (self.swapper.noise.eta_eff)**self.config.rep_nodes
        n_attempts = self.rng.geometric(prob_success_swaps)
        attempt_time = 0.0
        for _ in range(n_attempts-1):
            n_trials = max(self.rng.geometric(self.source.noise.eta_eff) for _ in self.chain.links_indices())
            ready_time = n_trials * self.config.trial_time_s
            attempt_time += ready_time + self.config.swapping_time_s
        self.now += attempt_time

    def run(self) -> SimulationResult:  
        if self.swap_policy == SwapPolicy.SWAP_AT_LAST:
            self.last_policy_failed_attempts()
        
        completed_links: list[CompletedLink] = []
        self.scheduler.schedule_source_attempts(self.now, self.queue)

        target_bell = Q.Bell.phi_plus

        while self.queue and len(completed_links) < self.config.n_target_links:
            event = self.queue.pop()
            self.now = event.time

            if event.type == EventType.SOURCE_COMPLETE:
                new_link: Link = event.payload
                new_link.left_slot.status = SlotStatus.READY
                new_link.right_slot.status = SlotStatus.READY
                new_link.left_slot.link = new_link
                new_link.right_slot.link = new_link

            elif event.type == EventType.SWAP_RESOLVE:
                swap: SwapAttempt = event.payload
                swap.inner_left.status = SlotStatus.EMPTY
                swap.inner_right.status = SlotStatus.EMPTY
                swap.inner_left.link = None
                swap.inner_right.link = None

                if swap.succeeded and swap.new_link is not None:
                    new_link = swap.new_link
                    if new_link.left_node.is_end_node and new_link.right_node.is_end_node:
                        fid=Qp(new_link.rho,target_bell).fid()
                        completed_links.append(
                            CompletedLink(rho=new_link.rho, fidelity=fid, created_time=self.now,))
                        swap.outer_left.status = SlotStatus.EMPTY
                        swap.outer_right.status = SlotStatus.EMPTY
                        swap.outer_left.link = None
                        swap.outer_right.link = None
                    else:
                        swap.outer_left.status = SlotStatus.READY
                        swap.outer_right.status = SlotStatus.READY
                        swap.outer_left.link = new_link
                        swap.outer_right.link = new_link
                else:
                    swap.outer_left.status = SlotStatus.EMPTY
                    swap.outer_right.status = SlotStatus.EMPTY
                    swap.outer_left.link = None
                    swap.outer_right.link = None

            if self.swap_policy == SwapPolicy.SWAP_AT_LAST:
                if not self.queue and len(completed_links) < self.config.n_target_links: # empty queue: When Ready to Swap
                    self.scheduler.schedule_swap_last(self.now, self.queue) # Deterministic swaps
                    if not self.queue:  # still queue empty (Full link obtained)
                        self.last_policy_failed_attempts()
                        self.scheduler.schedule_source_attempts(self.now, self.queue)
            elif self.swap_policy == SwapPolicy.NESTING:
                self.scheduler.schedule_source_attempts(self.now, self.queue)
                self.scheduler.schedule_swap_nesting(self.now, self.queue)
            elif self.swap_policy == SwapPolicy.SWAP_ASAP:
                self.scheduler.schedule_source_attempts(self.now, self.queue)
                self.scheduler.schedule_swap_asap(self.now, self.queue)

        avg_fid = float(np.mean([p.fidelity for p in completed_links])) if completed_links else 0.0
        rate = len(completed_links) / self.now if self.now > 0 else 0.0

        return SimulationResult(completed_links=completed_links, total_sim_time=self.now, average_fidelity=avg_fid,
                                entanglement_rate=rate,)