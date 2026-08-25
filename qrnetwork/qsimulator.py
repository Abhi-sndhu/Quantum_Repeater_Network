from __future__ import annotations
import heapq
import itertools
from dataclasses import dataclass, field
from enum import Enum, auto
import numpy as np

from . import primaryfn
from .network import (EntangledPair,MemorySlot,RepeaterChain,SlotStatus,EntanglementSource,EntanglementSwapper,ImperfectBSM,
                      DephasingMemory,DarkHeralding,)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SimulationConfig:
    rep_nodes: int = 3
    rep_spacing_km: float = 50.0
    attenuation_length_km: float = 22.0
    fibre_speed_km_per_s: float = 200_000.0
 
    pairgen_prob_perpulse: float = 0.01    # P(n)=mu^n/(1+mu)^(n+1), where n is number of pairs and
                                           # mu represents the mean number of photon pairs generated per pump pulse
    prep_time_s: float = 1e-8              # preparation time for source event  
                                           # depends on clock rate(At what rate source can make a new attempt)
    source_state_fidelity: float = 1.0

    dark_count_rate_hz: float = 100.0
    detection_window_s: float = 1e-9       # time for which detector is on,   # more than jitter time,   # least possible available: 20ps*2-3 
    dephasing_time_s: float = 0.01

    bsm_fidelity: float = 1.0

    n_target_pairs: int = 1
    n_parallel_slots: int = 1              # Increase n_parallel_slots for spatial-multiplexing (multi-memory slots model)
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.rep_nodes < 1:
            raise ValueError("rep_nodes must be >= 1 (need at least one repeater)")
        if self.n_parallel_slots < 1:
            raise ValueError("n_parallel_links must be >= 1")
        if not (0.0 <= self.source_state_fidelity <= 1.0):
            raise ValueError("source_state_fidelity must be in [0, 1]")
        if not (0.0 <= self.bsm_fidelity <= 1.0):
            raise ValueError("bsm_fidelity must be in [0, 1]")
        if self.n_target_pairs < 1:
            raise ValueError("n_target_pairs must be >= 1")

    @property
    def total_nodes(self) -> int:
        return self.rep_nodes + 2

    @property
    def n_links(self) -> int:
        return self.rep_nodes + 1

    @property
    def dark_count_probability(self) -> float:
        return self.dark_count_rate_hz * self.detection_window_s

    @property
    def channel_efficiency(self) -> float:  # e^(-L/Latt) # In our protocol, photon travels from source to repeater, so L -> L/2
        return float(np.exp(-self.rep_spacing_km / (2 * self.attenuation_length_km)))

    @property
    def swapping_time_s(self) -> float:   #L/c: While swapping, It takes L/2c for photon to reach Rep + L/2c for communication
        return self.rep_spacing_km / self.fibre_speed_km_per_s


# ---------------------------------------------------------------------------
# Event Queue
# ---------------------------------------------------------------------------

class EventType(Enum):
    SOURCE_COMPLETE = auto()
    SWAP_RESOLVE = auto()


@dataclass(order=True)
class Event:
    time: float
    sequence: int
    type: EventType = field(compare=False)
    payload: EntangledPair | SwapAttempt = field(compare=False)


class EventQueue:
    def __init__(self) -> None:
        self._heap: list[Event] = []
        self._counter = itertools.count()

    def push(self, time: float, type: EventType, payload: EntangledPair | SwapAttempt) -> None:
        heapq.heappush(self._heap, Event(time=time, sequence=next(self._counter), type=type, payload=payload))

    def pop(self) -> Event:
        return heapq.heappop(self._heap)

    def __len__(self) -> int:
        return len(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------

@dataclass
class SwapAttempt:
    repeater: int
    inner_left: MemorySlot
    inner_right: MemorySlot
    outer_left: MemorySlot
    outer_right: MemorySlot
    resolve_time: float
    succeeded: bool
    new_pair: EntangledPair | None


class Scheduler:
    def __init__(self, chain: RepeaterChain, source: EntanglementSource, swapper: EntanglementSwapper, 
                 swapping_time_s: float, rng: np.random.Generator,):
        self.chain = chain
        self.source = source
        self.swapper = swapper
        self.t_swap = swapping_time_s
        self.rng = rng

    def start_source_attempt(self, left_slot: MemorySlot, right_slot: MemorySlot, now: float) -> tuple[float, EntangledPair]:
        left_slot.status = right_slot.status = SlotStatus.PENDING
        wait_s, result = self.source.attempt(self.rng)
        complete_time = now + wait_s
        pair = EntangledPair(id=self.chain.new_pair_id(), rho=result.rho, ready_time=complete_time, left_slot=left_slot, 
                             right_slot=right_slot,)
        return complete_time, pair

    def start_swap(self, repeater: int, left_slot: MemorySlot, right_slot: MemorySlot, now: float) -> SwapAttempt:
        left_pair, right_pair = left_slot.pair, right_slot.pair
        assert left_pair is not None and right_pair is not None
        outer_left, outer_right = left_pair.left_slot, right_pair.right_slot

        left_wait = now - left_pair.ready_time
        right_wait = now - right_pair.ready_time
        succeeded, result = self.swapper.attempt(left_pair.rho, right_pair.rho, left_wait, right_wait, self.rng)

        for slot in (left_slot, right_slot, outer_left, outer_right):
            slot.status = SlotStatus.PENDING

        resolve_time = now + self.t_swap
        new_pair = None
        if succeeded:
            new_pair = EntangledPair(id=self.chain.new_pair_id(), rho=result.rho, ready_time=resolve_time, left_slot=outer_left, 
                                     right_slot=outer_right,)

        return SwapAttempt(repeater=repeater, inner_left=left_slot, inner_right=right_slot, outer_left=outer_left, 
                           outer_right=outer_right, resolve_time=resolve_time,
                           succeeded=succeeded, new_pair=new_pair,)


# ---------------------------------------------------------------------------
# Simulation Engine & Results
# ---------------------------------------------------------------------------

@dataclass
class CompletedPair:
    id: int
    rho: np.ndarray
    fidelity: float
    created_time: float


@dataclass
class SimulationResult:
    completed_pairs: list[CompletedPair]
    total_sim_time: float
    average_fidelity: float
    entanglement_rate: float


class RepeaterSimulation:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed) #defining rng for the entire run with seed managed
        self.chain = RepeaterChain(config.rep_nodes, config.n_parallel_slots)

        source_heralding = DarkHeralding.calculate(config.pairgen_prob_perpulse, config.pairgen_prob_perpulse, 
                                                config.dark_count_probability)
        swap_heralding = DarkHeralding.calculate(config.channel_efficiency, config.channel_efficiency, 
                                              config.dark_count_probability)
        bsm = ImperfectBSM(fidelity=config.bsm_fidelity)
        mem = DephasingMemory(dephasing_time_s=config.dephasing_time_s)

        self.source = EntanglementSource(imperfect_bsm=bsm, heralding_probs=source_heralding, prep_time_s=config.prep_time_s,
            source_state_fidelity=config.source_state_fidelity,)
        self.swapper = EntanglementSwapper(imperfect_bsm=bsm, heralding_probs=swap_heralding, memory_noise=mem)
        self.scheduler = Scheduler(chain=self.chain, source=self.source, swapper=self.swapper, 
                                   swapping_time_s=config.swapping_time_s, rng=self.rng,)
        self.queue = EventQueue()
        self.now = 0.0

    def _trigger_source_attempts(self) -> None:
        for i in self.chain.links_indices(): #for i in range(n_links)
            left_node = self.chain.nodes[i]
            right_node = self.chain.nodes[i + 1]
            for l_slot, r_slot in zip(left_node.right_slots, right_node.left_slots):
                if l_slot.is_available_for_new_attempt() and r_slot.is_available_for_new_attempt():
                    finish_t, pair = self.scheduler.start_source_attempt(l_slot, r_slot, self.now)
                    self.queue.push(finish_t, EventType.SOURCE_COMPLETE, pair)

    def _check_and_trigger_swaps(self) -> None:
        for r_idx in self.chain.repeater_indices(): #for all repeaters
            node = self.chain.nodes[r_idx]
            for l_slot in node.left_slots:
                if l_slot.status == SlotStatus.READY and l_slot.pair is not None:
                    for r_slot in node.right_slots:
                        if r_slot.status == SlotStatus.READY and r_slot.pair is not None:
                            swap_attempt = self.scheduler.start_swap(r_idx, l_slot, r_slot, self.now)
                            self.queue.push(swap_attempt.resolve_time, EventType.SWAP_RESOLVE, swap_attempt)
                            break

    def run(self) -> SimulationResult:
        completed_pairs: list[CompletedPair] = []
        self._trigger_source_attempts()

        target_bell = primaryfn.Q.Bell.phi_plus

        while self.queue and len(completed_pairs) < self.config.n_target_pairs:
            event = self.queue.pop()
            self.now = event.time

            if event.type == EventType.SOURCE_COMPLETE:
                pair: EntangledPair = event.payload
                pair.left_slot.status = SlotStatus.READY
                pair.right_slot.status = SlotStatus.READY
                pair.left_slot.pair = pair
                pair.right_slot.pair = pair

            elif event.type == EventType.SWAP_RESOLVE:
                attempt: SwapAttempt = event.payload
                attempt.inner_left.status = SlotStatus.EMPTY
                attempt.inner_right.status = SlotStatus.EMPTY
                attempt.inner_left.pair = None
                attempt.inner_right.pair = None

                if attempt.succeeded and attempt.new_pair is not None:
                    new_pair = attempt.new_pair
                    if new_pair.span_hops == self.config.total_nodes - 1:
                        fid=primaryfn.Qp(new_pair.rho,target_bell).fid()
                        completed_pairs.append(
                            CompletedPair(id=new_pair.id, rho=new_pair.rho, fidelity=fid, created_time=self.now,))
                        attempt.outer_left.status = SlotStatus.EMPTY
                        attempt.outer_right.status = SlotStatus.EMPTY
                        attempt.outer_left.pair = None
                        attempt.outer_right.pair = None
                    else:
                        attempt.outer_left.status = SlotStatus.READY
                        attempt.outer_right.status = SlotStatus.READY
                        attempt.outer_left.pair = new_pair
                        attempt.outer_right.pair = new_pair
                else:
                    attempt.outer_left.status = SlotStatus.EMPTY
                    attempt.outer_right.status = SlotStatus.EMPTY
                    attempt.outer_left.pair = None
                    attempt.outer_right.pair = None

            self._trigger_source_attempts()
            self._check_and_trigger_swaps()

        avg_fid = float(np.mean([p.fidelity for p in completed_pairs])) if completed_pairs else 0.0
        rate = len(completed_pairs) / self.now if self.now > 0 else 0.0

        return SimulationResult(completed_pairs=completed_pairs, total_sim_time=self.now, average_fidelity=avg_fid,
                                entanglement_rate=rate,)