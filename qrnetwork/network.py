from __future__ import annotations
import numpy as np
from . import qteleportation
from . import primaryfn
from sympy import Matrix
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
# Change the default warning format
def custom_warning_format(message, category, filename, lineno, line=None):
    return f"⚠️ {message}\n" # Just return warn message not category, filename, etc.
warnings.formatwarning = custom_warning_format
# Bell bases
bell_states = [primaryfn.Q.Bell.phi_plus, primaryfn.Q.Bell.phi_minus, 
               primaryfn.Q.Bell.psi_plus, primaryfn.Q.Bell.psi_minus]
# Possible Bell pairs for swap
swap_pairs = [np.kron(primaryfn.Q.Bell.phi_plus, primaryfn.Q.Bell.phi_plus),
              np.kron(primaryfn.Q.Bell.phi_plus, primaryfn.Q.Bell.phi_minus),
              np.kron(primaryfn.Q.Bell.phi_plus, primaryfn.Q.Bell.psi_plus),
              np.kron(primaryfn.Q.Bell.phi_plus, primaryfn.Q.Bell.psi_minus),
              np.kron(primaryfn.Q.Bell.phi_minus, primaryfn.Q.Bell.phi_plus),
              np.kron(primaryfn.Q.Bell.phi_minus, primaryfn.Q.Bell.phi_minus),
              np.kron(primaryfn.Q.Bell.phi_minus, primaryfn.Q.Bell.psi_plus),
              np.kron(primaryfn.Q.Bell.phi_minus, primaryfn.Q.Bell.psi_minus),
              np.kron(primaryfn.Q.Bell.psi_plus, primaryfn.Q.Bell.phi_plus),
              np.kron(primaryfn.Q.Bell.psi_plus, primaryfn.Q.Bell.phi_minus),
              np.kron(primaryfn.Q.Bell.psi_plus, primaryfn.Q.Bell.psi_plus),
              np.kron(primaryfn.Q.Bell.psi_plus, primaryfn.Q.Bell.psi_minus),
              np.kron(primaryfn.Q.Bell.psi_minus, primaryfn.Q.Bell.phi_plus),
              np.kron(primaryfn.Q.Bell.psi_minus, primaryfn.Q.Bell.phi_minus),
              np.kron(primaryfn.Q.Bell.psi_minus, primaryfn.Q.Bell.psi_plus),
              np.kron(primaryfn.Q.Bell.psi_minus, primaryfn.Q.Bell.psi_minus)]
swap_pairs_matrix = [primaryfn.Qp(b).op() for b in swap_pairs]
# We want to obtain the desired state (phi_plus) after each swap.
desired_state = primaryfn.Q(primaryfn.Q.Bell.phi_plus).dm()
# desired_state = np.outer(qteleportation.Bell.phi_plus,qteleportation.Bell.phi_plus.conj())

# Pauli corrections tables to get phi_plus after swap.
pauli_corrections = [
    [primaryfn.Q.Pauli.I, primaryfn.Q.Pauli.Z, primaryfn.Q.Pauli.X, 1j*primaryfn.Q.Pauli.Y],
    [primaryfn.Q.Pauli.Z, primaryfn.Q.Pauli.I, 1j*primaryfn.Q.Pauli.Y, primaryfn.Q.Pauli.X],
    [primaryfn.Q.Pauli.X, 1j*primaryfn.Q.Pauli.Y, primaryfn.Q.Pauli.I, primaryfn.Q.Pauli.Z],
    [1j*primaryfn.Q.Pauli.Y, primaryfn.Q.Pauli.X, primaryfn.Q.Pauli.Z, primaryfn.Q.Pauli.I],
    [primaryfn.Q.Pauli.Z, primaryfn.Q.Pauli.I, 1j*primaryfn.Q.Pauli.Y, primaryfn.Q.Pauli.X],
    [primaryfn.Q.Pauli.I, primaryfn.Q.Pauli.Z, primaryfn.Q.Pauli.X, 1j*primaryfn.Q.Pauli.Y],
    [1j*primaryfn.Q.Pauli.Y, primaryfn.Q.Pauli.X, primaryfn.Q.Pauli.Z, primaryfn.Q.Pauli.I],
    [primaryfn.Q.Pauli.X, 1j*primaryfn.Q.Pauli.Y, primaryfn.Q.Pauli.I, primaryfn.Q.Pauli.Z],
    [primaryfn.Q.Pauli.X, 1j*primaryfn.Q.Pauli.Y, primaryfn.Q.Pauli.I, primaryfn.Q.Pauli.Z],
    [1j*primaryfn.Q.Pauli.Y, primaryfn.Q.Pauli.X, primaryfn.Q.Pauli.Z, primaryfn.Q.Pauli.I],
    [primaryfn.Q.Pauli.I, primaryfn.Q.Pauli.Z, primaryfn.Q.Pauli.X, 1j*primaryfn.Q.Pauli.Y],
    [primaryfn.Q.Pauli.Z, primaryfn.Q.Pauli.I, 1j*primaryfn.Q.Pauli.Y, primaryfn.Q.Pauli.X],
    [1j*primaryfn.Q.Pauli.Y, primaryfn.Q.Pauli.X, primaryfn.Q.Pauli.Z, primaryfn.Q.Pauli.I],
    [primaryfn.Q.Pauli.X, 1j*primaryfn.Q.Pauli.Y, primaryfn.Q.Pauli.I, primaryfn.Q.Pauli.Z],
    [primaryfn.Q.Pauli.Z, primaryfn.Q.Pauli.I, 1j*primaryfn.Q.Pauli.Y, primaryfn.Q.Pauli.X],
    [primaryfn.Q.Pauli.I, primaryfn.Q.Pauli.Z, primaryfn.Q.Pauli.X, 1j*primaryfn.Q.Pauli.Y]
]
def channel_efficiency(L, L_att=22, loss=None):
    if loss is not None:
        L_att = 10 * np.log10(np.e) / loss   # compute from loss in dB/km
    return np.exp(-L / L_att)

def dephasing_coeff(T_p, T_dp, L):
    c = 2 * 10**8  # speed of light in optical fiber (m/s)
    t = T_p + 2*L/c
    return (1 - np.exp(-t / T_dp)) / 2

def clicking_coeff(eta, p_d):
    eta_eff = 1 - (1 - eta) * (1 - p_d)**2
    return (eta * (1 - p_d)) / eta_eff

def amp_damp_kraus(gamma):
    K0 = np.array([[1, 0],
                   [0, np.sqrt(1 - gamma)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(gamma)],
                   [0, 0]], dtype=complex)
    return [K0,K1]


class eswap:
    def __init__(self, node1, node2, tol=1e-10):
        self.qubits = [node1, node2]
        self.tol = tol

    def ket(self):
        full_state = np.kron(self.qubits[0], self.qubits[1])
        fidelities = [primaryfn.Qp(full_state, c).fid() for c in swap_pairs]
        j = int(np.argmax(fidelities))
        results = []
        for i, M in enumerate(bell_states):
            correction = pauli_corrections[j][i]
            opr = np.kron(np.kron(np.eye(2), primaryfn.Q(M).d()),correction) # Note first and last qubit give the desired state.
            post_meas = opr @ full_state
            prob = round(np.real((primaryfn.Qp(post_meas).ip()).item()),2)
            norm = np.sqrt(prob)
            if norm == 0:
                continue
            obtained_state = post_meas / norm
            fidelity = primaryfn.Qp(primaryfn.Q.Bell.phi_plus, obtained_state).fid()
            results.append({"measurement_outcome": f"Bell-{i+1}",
                            "probability": prob,
                            "fidelity": fidelity,
                            "obtained_entangled_state": obtained_state
                                })
        return results
    def density(self):
        full_state = np.kron(self.qubits[0], self.qubits[1])
        fidelities = [primaryfn.Qp(full_state, c).fid() for c in swap_pairs_matrix]
        j = int(np.argmax(fidelities))
        results = []
        for i, M in enumerate(bell_states):
            correction = pauli_corrections[j][i]
            opr = np.kron(np.kron(np.eye(2), primaryfn.Q(M).d()),correction)
            post_meas = opr @ full_state @ primaryfn.Q(opr).d() # @ is not allowed in our Qstate
            norm = np.trace(post_meas)
            prob = round(float(np.real(norm)), 2)
            if norm == 0:
                continue
            obtained_state = post_meas / norm
            # enforce Hermiticity & normalization
            obtained_state = (obtained_state + primaryfn.Q(obtained_state).d()) / 2
            obtained_state = obtained_state / np.trace(obtained_state)
            fidelity = primaryfn.Qp(desired_state, obtained_state).fid()
            results.append({"measurement_outcome": f"Bell-{i+1}",
                            "probability": prob,
                            "fidelity": fidelity,
                            "obtained_entangled_state": obtained_state
                                })
        return results
    def noise(self, L, T_p, T_dp, eta, p_d, loss=None): #L: Fiber length, T_p: Entanglement generation time, 
        # T_dp: Depahsing time, eta: Channel efficiency, p_d: Dark count probability
        rhoin = np.kron(self.qubits[0], self.qubits[1])
        # --- Amplitude damping ---
        if loss is not None:
            gamma = 1 - channel_efficiency(L, loss=loss)
        else:
            gamma = 1 - channel_efficiency(L)
        K = amp_damp_kraus(gamma)
        kraus_op_2q = [np.kron(np.kron(np.kron(np.eye(2), k1), k2), np.eye(2))
                       for k1 in K for k2 in K]
        rho1 = sum(K @ rhoin @ primaryfn.Q(K).d() for K in kraus_op_2q) 
        rho1 = rho1 / np.trace(rho1)
        # --- Dephasing ---
        lam_t = dephasing_coeff(T_p, T_dp, L)
        # lam_t_coeff1 = (1 - lam_t) * np.eye(16)
        lam_t_coeff2 = (np.kron(np.kron(np.eye(2), 
                                                np.kron(primaryfn.Q.Pauli.Z, primaryfn.Q.Pauli.Z)), 
                                                np.eye(2)))
        rho2 = (1 - lam_t) * rho1 + \
               lam_t * (lam_t_coeff2 @ rho1 @ primaryfn.Q(lam_t_coeff2).d())
        rho2 = rho2 / np.trace(rho2)
        # --- Detector noise (clicking) ---
        alpha = clicking_coeff(eta, p_d)
        # alpha_coeff1 = alpha * np.eye(16)
        # alpha_coeff2 = ((1 - alpha) / 2) * np.eye(16)
        rho3 = alpha * rho2 + \
               ((1 - alpha) / 2)* rho2
        rho3 = rho3 / np.trace(rho3)
        # --- Fidelity with expected channels ---
        fidelities = [primaryfn.Qp(rho3, c).fid() for c in swap_pairs_matrix]
        j = int(np.argmax(fidelities))
        # --- Measurement and correction ---
        results = []
        for i, M in enumerate(bell_states):
            correction = pauli_corrections[j][i]
            opr = np.kron(np.kron(np.eye(2), primaryfn.Q(M).d()),correction)
            post_meas = opr @ rho3 @ primaryfn.Q(opr).d()
            norm = np.trace(post_meas)
            prob = round(float(np.real(norm)), 2)
            if norm == 0:
                continue
            obtained_state = post_meas / norm
            # enforce Hermiticity & normalization
            obtained_state = (obtained_state + primaryfn.Q(obtained_state).d()) / 2
            obtained_state = obtained_state / np.trace(obtained_state)
            fidelity = primaryfn.Qp(desired_state, obtained_state).fid()
            results.append({"measurement_outcome": f"Bell-{i+1}",
                            "probability": prob,
                            "fidelity": fidelity,
                            "obtained_entangled_state": obtained_state
                                })
        return results

class QRep:
    def __init__(self,state_shared,tol=1e-10):
        self.state_shared = state_shared
        self.tol = tol
        self.noise_params = None
    def noise(self, L, T_p, T_dp, eta, p_d, loss=None):
        """Enable noisy swapping with given parameters"""
        self.noise_params = {"L": L, "T_p": T_p, "T_dp": T_dp, "eta": eta, "p_d": p_d, "loss": loss}
        return self
    def linear(self):
        #---noisy case---
        if self.noise_params is not None:
            L, T_p, T_dp, eta, p_d, loss = (
                self.noise_params["L"],
                self.noise_params["T_p"],
                self.noise_params["T_dp"],
                self.noise_params["eta"],
                self.noise_params["p_d"],
                self.noise_params["loss"],
            )

            if all([primaryfn.Is(state).ket() for state in self.state_shared]):
                raise ValueError("Noise model requires density matrices, not kets.")

            if len(self.state_shared) == 1:
                raise ValueError("The shared entangled state should be greater than two.")
            elif len(self.state_shared) == 2:
                if loss is not None:
                    return eswap(self.state_shared[0], self.state_shared[1]).noise(L, T_p, T_dp, eta, p_d, loss)
                else:
                    # Raise a warning
                    # print("Warning: By default loss in fiber is taken as 0.2 dB/km") 
                    warnings.warn("By default loss in fiber is taken as 0.2 dB/km", UserWarning)
                    return eswap(self.state_shared[0], self.state_shared[1]).noise(L, T_p, T_dp, eta, p_d)
            else:
                current_state = self.state_shared[0]
                for i in range(1, len(self.state_shared)):
                    if loss is not None:
                        results = eswap(current_state, self.state_shared[i]).noise(L, T_p, T_dp, eta, p_d, loss)
                    else:
                        # Raise a warning
                        # print(" Warning: By default loss in fiber is taken as 0.2 dB/km.")
                        # With print command warning prints multiple times depending on loops size.
                        warnings.warn("By default loss in fiber is taken as 0.2 dB/km", UserWarning)
                        results = eswap(current_state, self.state_shared[i]).noise(L, T_p, T_dp, eta, p_d)
                    current_state = results[0]["obtained_entangled_state"]
                return results
        # --- noiseless case ---
        if all([primaryfn.Is(state).ket() for state in self.state_shared]): #Check dimensions
            purity = [primaryfn.Q(c).u() for c in self.state_shared]
            ent_val = [primaryfn.Ebit(d) for d in self.state_shared]
            if all(v >= self.tol for v in ent_val):
                if len(self.state_shared) == 1:
                    raise ValueError("The shared entangled state should be greater than two.")
                elif len(self.state_shared) == 2:
                    output = eswap(self.state_shared[0], self.state_shared[1]).ket()
                    return output
                elif len(self.state_shared) > 2:
                    #measurements = []
                    current_state = self.state_shared[0]
                    for i in range(1, len(self.state_shared)):
                        results = eswap(current_state, self.state_shared[i]).ket()
                        #measurements.append(results)
                        current_state = results[0]["obtained_entangled_state"]
                    return results #measurements
            else:
                raise ValueError("The shared states are not not entangled.")
        elif all([primaryfn.Is(state).square() for state in self.state_shared]):
            purity = [primaryfn.Q(c).u() for c in self.state_shared]
            ent_val = [primaryfn.Ebit(d) for d in self.state_shared]
            if all(np.abs(c) > self.tol for c in ent_val):
                if len(self.state_shared) == 1:
                    raise ValueError("The shared entangled state should be greater than two.")
                elif len(self.state_shared) == 2:
                    output = eswap(self.state_shared[0], self.state_shared[1]).density()
                    return output
                elif len(self.state_shared) > 2:
                    current_state = self.state_shared[0]
                    for i in range(1, len(self.state_shared)):
                        results = eswap(current_state, self.state_shared[i]).density()
                        #measurements.append(results)
                        current_state = results[0]["obtained_entangled_state"]
                    return results #measurements
            else:
                raise ValueError("The shared states are not entangled.")
        else:
            raise ValueError("Incorrect dimensions of the shared state.")

# ---------------------------------------------------------------------------
# Projectors & Operators for Hardware Models & BSM
# ---------------------------------------------------------------------------
_I2 = primaryfn.Q.Pauli.I
_PAULI_X = primaryfn.Q.Pauli.X
_PAULI_Y = primaryfn.Q.Pauli.Y
_PAULI_Z = primaryfn.Q.Pauli.Z

_RHO_00 = primaryfn.Q([[1],[0],[0],[0]]).dm().state
_RHO_11 = primaryfn.Q([[0],[0],[0],[1]]).dm().state
_RHO_PHI_PLUS = primaryfn.Q(primaryfn.Q.Bell.phi_plus).dm().state
_RHO_PHI_MINUS = primaryfn.Q(primaryfn.Q.Bell.phi_minus).dm().state
_RHO_PSI_PLUS = primaryfn.Q(primaryfn.Q.Bell.psi_plus).dm().state
_RHO_PSI_MINUS = primaryfn.Q(primaryfn.Q.Bell.psi_minus).dm().state
_MAX_MIXED_2Q = np.eye(4, dtype=complex) / 4
_OTHER_BELL_STATES_SUM = _RHO_PHI_PLUS + _RHO_PHI_MINUS + _RHO_PSI_PLUS + _RHO_PSI_MINUS

_PSI_PROJECTORS = (_RHO_PSI_PLUS, _RHO_PSI_MINUS)
_MIDDLE_PROJECTORS = tuple(np.kron(np.kron(_I2, P), _I2) for P in _PSI_PROJECTORS)
_CORRECTION_OPS = (np.kron(_I2, _PAULI_X), np.kron(_I2, _PAULI_Y))


# ---------------------------------------------------------------------------
# Noise Models & Bell State Measurement
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HeraldingStatistics:
    p_success: float
    p_correct_click: float
    p_wrong_bell_pair: float
    p_no_information: float

    def __post_init__(self) -> None:
        total = self.p_correct_click + self.p_wrong_bell_pair + self.p_no_information
        if not np.isclose(total, 1.0, atol=1e-9):
            raise ValueError(f"heralding outcome probabilities must sum to 1, got {total}")


def heralding_statistics(eta_left: float, eta_right: float, p_dark: float) -> HeraldingStatistics:
    both_arrive = eta_left * eta_right
    only_left = eta_left * (1 - eta_right)
    only_right = (1 - eta_left) * eta_right
    neither_arrives = (1 - eta_left) * (1 - eta_right)

    p_success = (0.5 * both_arrive * (1 - p_dark) ** 2
        + 0.5 * both_arrive * 2 * p_dark * (1 - p_dark) ** 2
        + neither_arrives * 4 * p_dark**2 * (1 - p_dark) ** 2
        + only_left * 2 * p_dark * (1 - p_dark) ** 2
        + only_right * 2 * p_dark * (1 - p_dark) ** 2)
    
    p_correct_click = (0.5 * both_arrive * (1 - p_dark) ** 2) / p_success
    p_wrong_bell_pair = (0.5 * both_arrive * 2 * p_dark * (1 - p_dark) ** 2) / p_success
    p_no_information = 1.0 - p_correct_click - p_wrong_bell_pair
    return HeraldingStatistics(p_success, p_correct_click, p_wrong_bell_pair, p_no_information)


def apply_dark_count_mixing(rho: np.ndarray, stats: HeraldingStatistics) -> np.ndarray:
    return (stats.p_correct_click * rho
        + stats.p_wrong_bell_pair * (_RHO_00 + _RHO_11) / 2
        + stats.p_no_information * _MAX_MIXED_2Q)


@dataclass(frozen=True)
class DephasingMemory:
    dephasing_time_s: float

    def dephasing_probability(self, wait_time_s: float) -> float:
        if wait_time_s < 0:
            raise ValueError(f"negative memory wait time ({wait_time_s:.3e}s)")
        exponent = -wait_time_s / self.dephasing_time_s
        return (1 - np.exp(exponent)) / 2

    def apply(self, rho: np.ndarray, wait_time_s: float, qubit: int) -> np.ndarray:
        p = self.dephasing_probability(wait_time_s)
        if p == 0.0:
            return rho
        z_op = np.kron(_PAULI_Z, _I2) if qubit == 0 else np.kron(_I2, _PAULI_Z)
        return (1 - p) * rho + p * (z_op @ rho @ z_op)


@dataclass(frozen=True)
class ImperfectBSM:
    fidelity: float = 1.0

    def apply(self, rho_4q: np.ndarray) -> np.ndarray:
        if self.fidelity >= 1.0:
            return rho_4q
        reduced_outer = primaryfn.partial_trace_multi(rho_4q, 4, [1, 2])
        scrambled = primaryfn.expand_with_maximally_mixed(reduced_outer, kept_positions=[0, 3], total_qubits=4)
        return self.fidelity * rho_4q + (1 - self.fidelity) * scrambled


@dataclass(frozen=True)
class BellSwapResult:
    rho: np.ndarray
    outcome: int  # 0 -> psi+ heralded, 1 -> psi- heralded


def perform_bell_swap(rho_left_pair: np.ndarray,rho_right_pair: np.ndarray,imperfect_bsm: ImperfectBSM,
                      heralding: HeraldingStatistics,rng: np.random.Generator,) -> BellSwapResult:
    rho_4q = np.kron(rho_left_pair, rho_right_pair)
    rho_4q = imperfect_bsm.apply(rho_4q)
    probs = np.clip(np.array([np.trace(P @ rho_4q).real for P in _MIDDLE_PROJECTORS]), 0.0, None)
    probs_normalised = probs / probs.sum()
    outcome = int(rng.choice(2, p=probs_normalised))

    projector = _MIDDLE_PROJECTORS[outcome]
    projected = projector @ rho_4q @ projector / probs[outcome]
    rho_outer = primaryfn.partial_trace_multi(projected, 4, [1, 2])
    rho_outer = apply_dark_count_mixing(rho_outer, heralding)

    correction = _CORRECTION_OPS[outcome]
    rho_corrected = correction @ rho_outer @ correction.conj().T
    return BellSwapResult(rho=rho_corrected, outcome=outcome)


# ---------------------------------------------------------------------------
# Link Generation and Swappers
# ---------------------------------------------------------------------------

def werner_state(rho_target: np.ndarray, fidelity: float) -> np.ndarray:
    if fidelity >= 1.0:
        return rho_target
    others = _OTHER_BELL_STATES_SUM - rho_target
    return fidelity * rho_target + (1 - fidelity) / 3 * others


@dataclass(frozen=True)
class EntanglementSource:
    imperfect_bsm: ImperfectBSM
    heralding: HeraldingStatistics
    trial_period_s: float
    source_state_fidelity: float = 1.0

    def attempt(self, rng: np.random.Generator) -> tuple[float, BellSwapResult]:
        n_trials = rng.geometric(self.heralding.p_success)
        wait_time_s = n_trials * self.trial_period_s

        rho_left = werner_state(_RHO_PHI_PLUS, self.source_state_fidelity)
        rho_right = werner_state(_RHO_PHI_PLUS, self.source_state_fidelity)
        result = perform_bell_swap(rho_left, rho_right, self.imperfect_bsm, self.heralding, rng)
        return wait_time_s, result


@dataclass(frozen=True)
class EntanglementSwapper:
    imperfect_bsm: ImperfectBSM
    heralding: HeraldingStatistics
    memory_noise: DephasingMemory

    def attempt(self, rho_left_pair: np.ndarray, rho_right_pair: np.ndarray, left_wait_time_s: float, 
                right_wait_time_s: float, rng: np.random.Generator,) -> tuple[bool, BellSwapResult | None]:
        herald_fires = rng.random() < self.heralding.p_success
        if not herald_fires:
            return False, None

        rho_left = self.memory_noise.apply(rho_left_pair, left_wait_time_s, qubit=0)
        rho_left = self.memory_noise.apply(rho_left, left_wait_time_s, qubit=1)
        rho_right = self.memory_noise.apply(rho_right_pair, right_wait_time_s, qubit=0)
        rho_right = self.memory_noise.apply(rho_right, right_wait_time_s, qubit=1)

        result = perform_bell_swap(rho_left, rho_right, self.imperfect_bsm, self.heralding, rng)
        return True, result


# ---------------------------------------------------------------------------
# Chain Topology & Memory Slots
# ---------------------------------------------------------------------------

class SlotStatus(Enum):
    EMPTY = auto()
    PENDING = auto()
    READY = auto()


@dataclass
class MemorySlot:
    owner_node: int
    faces_node: int
    status: SlotStatus = SlotStatus.EMPTY
    pair: Optional["EntangledPair"] = None
    retired: bool = False

    def is_available_for_new_attempt(self) -> bool:
        return self.status is SlotStatus.EMPTY and not self.retired


@dataclass
class EntangledPair:
    id: int
    rho: np.ndarray
    ready_time: float
    left_slot: MemorySlot
    right_slot: MemorySlot
    consumed_inner_slots: list[MemorySlot] = field(default_factory=list)

    @property
    def left_node(self) -> int:
        return self.left_slot.owner_node

    @property
    def right_node(self) -> int:
        return self.right_slot.owner_node

    @property
    def span_hops(self) -> int:
        return self.right_node - self.left_node


@dataclass
class Node:
    index: int
    is_end_node: bool
    left_slots: list[MemorySlot] = field(default_factory=list)
    right_slots: list[MemorySlot] = field(default_factory=list)

    @property
    def all_slots(self) -> list[MemorySlot]:
        return self.left_slots + self.right_slots


class RepeaterChain:
    def __init__(self, n_nodes: int, memories_per_link: int):
        if n_nodes < 3:
            raise ValueError("n_nodes must be >= 3 (need at least one repeater)")
        if memories_per_link < 1:
            raise ValueError("memories_per_link must be >= 1")

        self.n_nodes = n_nodes
        self.memories_per_link = memories_per_link
        self.nodes: list[Node] = [Node(index=i, is_end_node=(i == 0 or i == n_nodes - 1)) for i in range(n_nodes)]
        for i in range(n_nodes - 1):
            j = i + 1
            for _ in range(memories_per_link):
                left_side_slot = MemorySlot(owner_node=i, faces_node=j)
                right_side_slot = MemorySlot(owner_node=j, faces_node=i)
                self.nodes[i].right_slots.append(left_side_slot)
                self.nodes[j].left_slots.append(right_side_slot)

        self._next_pair_id = 0

    def new_pair_id(self) -> int:
        self._next_pair_id += 1
        return self._next_pair_id

    def repeater_indices(self) -> range:
        return range(1, self.n_nodes - 1)

    def link_left_indices(self) -> range:
        return range(self.n_nodes - 1)

    def total_slots(self) -> int:
        return sum(len(n.all_slots) for n in self.nodes)

    def occupancy_summary(self) -> dict:
        counts = {status: 0 for status in SlotStatus}
        retired = 0
        for node in self.nodes:
            for slot in node.all_slots:
                counts[slot.status] += 1
                retired += int(slot.retired)
        return {"by_status": {s.name: c for s, c in counts.items()}, "retired": retired}