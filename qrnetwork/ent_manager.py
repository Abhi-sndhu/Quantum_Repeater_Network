from __future__ import annotations
from dataclasses import dataclass
from . import Q, Qp
import numpy as np


# ---------------------------------------------------------------------------
# Projectors & Operators for Hardware Models & BSM
# ---------------------------------------------------------------------------
_I2 = Q.Pauli.I
_PAULI_Z = Q.Pauli.Z
_PAULI_X = Q.Pauli.X
_PAULI_Y = Q.Pauli.Y

_RHO_00 = Q([[1],[0],[0],[0]]).dm()
_RHO_11 = Q([[0],[0],[0],[1]]).dm()
_RHO_PSI_PLUS = Q(Q.Bell.psi_plus).dm()
_RHO_PSI_MINUS = Q(Q.Bell.psi_minus).dm()
_MAX_MIXED_2Q = Q(np.eye(4, dtype=complex) / 4)

_PSI_PROJECTORS = (_RHO_PSI_PLUS, _RHO_PSI_MINUS)
_MIDDLE_PROJECTORS = tuple( Qp(Qp(_I2,P).tp(),_I2).tp() for P in _PSI_PROJECTORS)
_CORRECTION_OPS = (Qp(_I2, _PAULI_X).tp(), Qp(_I2, _PAULI_Y).tp())


# ---------------------------------------------------------------------------
# Noise Models & Bell State Measurement
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Noise:
    dephasing_param_s: float  # Memory noise
    bsm_fidelity: float       # Imperfect BSM
    #Linear BSM Stats
    eta_eff: float            # effective probability of success
    p_correct_click: float    # real event | success
    p_wrong_bell_pair: float  # wrong bell pair | success
    p_photon_loss: float      # photon lost | success


    @classmethod
    def calculate_stats(cls, dephasing_time_s: float, bsm_fidelity: float, eta_left: float, eta_right: float, p_dark: float):
        both_arrive = eta_left * eta_right
        only_left = eta_left * (1 - eta_right)
        only_right = (1 - eta_left) * eta_right
        neither_arrives = (1 - eta_left) * (1 - eta_right)

        eta_eff = (0.5 * both_arrive * (1 - p_dark) ** 2
            + 0.5 * both_arrive * 2 * p_dark * (1 - p_dark) ** 2
            + neither_arrives * 4 * p_dark**2 * (1 - p_dark) ** 2
            + only_left * 2 * p_dark * (1 - p_dark) ** 2
            + only_right * 2 * p_dark * (1 - p_dark) ** 2)
        
        p_correct_click = (0.5 * both_arrive * (1 - p_dark) ** 2) / eta_eff
        p_wrong_bell_pair = (0.5 * both_arrive * 2 * p_dark * (1 - p_dark) ** 2) / eta_eff
        p_photon_loss = 1.0 - p_correct_click - p_wrong_bell_pair
        return cls(dephasing_time_s, bsm_fidelity, eta_eff, p_correct_click, p_wrong_bell_pair, p_photon_loss)    


    """Dark_counts noise on successfull attempt (Entanglement Swapping)
       alpha1 rho + alpha2 (|00><00| + |11><11|)/2 + alpha3 I/4"""
    def dark_counts(self, rho: Q) -> Q:
        return (self.p_correct_click * rho
            + self.p_wrong_bell_pair * (_RHO_00 + _RHO_11) / 2
            + self.p_photon_loss * _MAX_MIXED_2Q)

    """Memory dephasing as dephasing noise channel on ith qubit.
       epsilon_i(t)rho = (1-lambda(t))rho + lambda(t) Z_i rho Z_i"""
    def mem_dephasing(self, rho: Q, wait_time_s: float, qubit: int) -> Q:
        exponent = -wait_time_s / self.dephasing_param_s
        p = (1 - np.exp(exponent)) / 2
        if p == 0.0:
            return rho
        z_op = Qp(_PAULI_Z, _I2).tp() if qubit == 0 else Qp(_I2, _PAULI_Z).tp()
        return (1 - p) * rho + p * (z_op @ rho @ z_op)


    """Imperfect BSM as two-qubit depolarizing noise on ith and jth qubit, which will be followed by BSM.
       epsilon_ij(lambda_bsm)rho + (1-lambda_bsm)/4 * tr_ij(rho) X I_ij"""
    def imperfect_bsm(self, rho_4q: Q) -> Q:
        if self.bsm_fidelity == 1.0:
            return rho_4q
        reduced_outer = rho_4q.partial_trace([1, 2])
        scrambled = reduced_outer.expand_mixed(kept_positions=[0, 3], total_qubits=4)
        return self.bsm_fidelity * rho_4q + (1 - self.bsm_fidelity) * scrambled

@dataclass(frozen=True)
class LinearEswap:
    noise: Noise
    @classmethod
    def perform(cls, rho_left_pair: Q, rho_right_pair: Q, noise: Noise, rng: np.random.Generator,) -> Q:
        rho_4q = Qp(rho_left_pair, rho_right_pair).tp()
        rho_4q = noise.imperfect_bsm(rho_4q)
        probs = np.array( [(P @ rho_4q).trace() for P in _MIDDLE_PROJECTORS] )
        probs_normalised = probs / probs.sum()
        outcome = int(rng.choice(2, p=probs_normalised))  # 0 -> psi+ heralded, 1 -> psi- heralded

        projector = _MIDDLE_PROJECTORS[outcome]
        projected = projector @ rho_4q @ projector / probs[outcome]
        rho_outer = projected.partial_trace([1, 2])
        correction = _CORRECTION_OPS[outcome]
        rho_corrected = correction @ rho_outer @ correction.d()

        rho_dc = noise.dark_counts(rho_corrected)
        return rho_dc


# ---------------------------------------------------------------------------
# Entanglement Generation and Entanglement Swapping
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EntGenSource:
    noise: Noise
    trial_time_s: float
    source_state_fidelity: float

    def attempt(self, rng: np.random.Generator) -> tuple[float, Q]:
        n_trials = rng.geometric(self.noise.eta_eff)
        wait_time_s = n_trials * self.trial_time_s
        rho_left = Q.Werner(p=(4*self.source_state_fidelity-1)/3)
        rho_right = Q.Werner(p=(4*self.source_state_fidelity-1)/3)
        state = LinearEswap.perform(rho_left, rho_right, self.noise, rng)
        return wait_time_s, state


@dataclass(frozen=True)
class RepSwap:
    noise: Noise

    def attempt(self, rho_left_pair: Q, rho_right_pair: Q, left_wait_time_s: float, 
                right_wait_time_s: float, rng: np.random.Generator,) -> tuple[bool, Q | None]:
        herald_fires = rng.random() < self.noise.eta_eff
        if not herald_fires:
            return False, None

        rho_left = self.noise.mem_dephasing(rho_left_pair, left_wait_time_s, qubit=0)
        rho_left = self.noise.mem_dephasing(rho_left, left_wait_time_s, qubit=1)
        rho_right = self.noise.mem_dephasing(rho_right_pair, right_wait_time_s, qubit=0)
        rho_right = self.noise.mem_dephasing(rho_right, right_wait_time_s, qubit=1)

        result = LinearEswap.perform(rho_left, rho_right, self.noise, rng)
        return True, result