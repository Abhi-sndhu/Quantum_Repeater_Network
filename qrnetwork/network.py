import numpy as np
from . import qteleportation
from . import primaryfn
from sympy import Matrix

# Bell bases
bell_states = [qteleportation.Bell.phi_plus, qteleportation.Bell.phi_minus, 
               qteleportation.Bell.psi_plus, qteleportation.Bell.psi_minus]
# Possible Bell pairs for swap
swap_pairs = [np.kron(qteleportation.Bell.phi_plus,qteleportation.Bell.phi_plus),
              np.kron(qteleportation.Bell.phi_plus,qteleportation.Bell.phi_minus),
              np.kron(qteleportation.Bell.phi_plus,qteleportation.Bell.psi_plus),
              np.kron(qteleportation.Bell.phi_plus,qteleportation.Bell.psi_minus),
              np.kron(qteleportation.Bell.phi_minus,qteleportation.Bell.phi_plus),
              np.kron(qteleportation.Bell.phi_minus,qteleportation.Bell.phi_minus),
              np.kron(qteleportation.Bell.phi_minus,qteleportation.Bell.psi_plus),
              np.kron(qteleportation.Bell.phi_minus,qteleportation.Bell.psi_minus),
              np.kron(qteleportation.Bell.psi_plus,qteleportation.Bell.phi_plus),
              np.kron(qteleportation.Bell.psi_plus,qteleportation.Bell.phi_minus),
              np.kron(qteleportation.Bell.psi_plus,qteleportation.Bell.psi_plus),
              np.kron(qteleportation.Bell.psi_plus,qteleportation.Bell.psi_minus),
              np.kron(qteleportation.Bell.psi_minus,qteleportation.Bell.phi_plus),
              np.kron(qteleportation.Bell.psi_minus,qteleportation.Bell.phi_minus),
              np.kron(qteleportation.Bell.psi_minus,qteleportation.Bell.psi_plus),
              np.kron(qteleportation.Bell.psi_minus,qteleportation.Bell.psi_minus)]
swap_pairs_matrix = [primaryfn.Qprod.outer(b) for b in swap_pairs]
# We want to obtain the desired state (phi_plus) after each swap.
desired_state = primaryfn.Belld.pure(qteleportation.Bell.phi_plus)
# desired_state = np.outer(qteleportation.Bell.phi_plus,qteleportation.Bell.phi_plus.conj())

# Pauli corrections tables to get phi_plus after swap.
pauli_corrections = [
    [primaryfn.Pauli.I, primaryfn.Pauli.Z, primaryfn.Pauli.X, 1j*primaryfn.Pauli.Y],
    [primaryfn.Pauli.Z, primaryfn.Pauli.I, 1j*primaryfn.Pauli.Y, primaryfn.Pauli.X],
    [primaryfn.Pauli.X, 1j*primaryfn.Pauli.Y, primaryfn.Pauli.I, primaryfn.Pauli.Z],
    [1j*primaryfn.Pauli.Y, primaryfn.Pauli.X, primaryfn.Pauli.Z, primaryfn.Pauli.I],
    [primaryfn.Pauli.Z, primaryfn.Pauli.I, 1j*primaryfn.Pauli.Y, primaryfn.Pauli.X],
    [primaryfn.Pauli.I, primaryfn.Pauli.Z, primaryfn.Pauli.X, 1j*primaryfn.Pauli.Y],
    [1j*primaryfn.Pauli.Y, primaryfn.Pauli.X, primaryfn.Pauli.Z, primaryfn.Pauli.I],
    [primaryfn.Pauli.X, 1j*primaryfn.Pauli.Y, primaryfn.Pauli.I, primaryfn.Pauli.Z],
    [primaryfn.Pauli.X, 1j*primaryfn.Pauli.Y, primaryfn.Pauli.I, primaryfn.Pauli.Z],
    [1j*primaryfn.Pauli.Y, primaryfn.Pauli.X, primaryfn.Pauli.Z, primaryfn.Pauli.I],
    [primaryfn.Pauli.I, primaryfn.Pauli.Z, primaryfn.Pauli.X, 1j*primaryfn.Pauli.Y],
    [primaryfn.Pauli.Z, primaryfn.Pauli.I, 1j*primaryfn.Pauli.Y, primaryfn.Pauli.X],
    [1j*primaryfn.Pauli.Y, primaryfn.Pauli.X, primaryfn.Pauli.Z, primaryfn.Pauli.I],
    [primaryfn.Pauli.X, 1j*primaryfn.Pauli.Y, primaryfn.Pauli.I, primaryfn.Pauli.Z],
    [primaryfn.Pauli.Z, primaryfn.Pauli.I, 1j*primaryfn.Pauli.Y, primaryfn.Pauli.X],
    [primaryfn.Pauli.I, primaryfn.Pauli.Z, primaryfn.Pauli.X, 1j*primaryfn.Pauli.Y]
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
        fidelities = [primaryfn.QFidelity(full_state, c) for c in swap_pairs]
        j = int(np.argmax(fidelities))
        results = []
        for i, M in enumerate(bell_states):
            correction = pauli_corrections[j][i]
            opr = np.kron(np.kron(np.eye(2), primaryfn.Qstate(M).dag()),correction) # Note first and last qubit give the desired state.
            post_meas = opr @ full_state
            prob = round(np.real(primaryfn.Qprod.inner(post_meas, post_meas).item()),2)
            norm = np.sqrt(prob)
            if norm == 0:
                continue
            obtained_state = post_meas / norm
            fidelity = primaryfn.QFidelity(qteleportation.Bell.phi_plus, obtained_state)
            results.append({"measurement_outcome": f"Bell-{i+1}",
                            "probability": prob,
                            "fidelity": fidelity,
                            "obtained_entangled_state": obtained_state
                                })
        return results
    def density(self):
        full_state = np.kron(self.qubits[0], self.qubits[1])
        fidelities = [primaryfn.QFidelity(full_state, c) for c in swap_pairs_matrix]
        j = int(np.argmax(fidelities))
        results = []
        for i, M in enumerate(bell_states):
            correction = pauli_corrections[j][i]
            opr = np.kron(np.kron(np.eye(2), primaryfn.Qstate(M).dag()),correction)
            post_meas = opr @ full_state @ opr.conj().T # @ is not allowed in our Qstate
            norm = np.trace(post_meas)
            prob = round(float(np.real(norm)), 2)
            if norm == 0:
                continue
            obtained_state = post_meas / norm
            # enforce Hermiticity & normalization
            obtained_state = (obtained_state + primaryfn.Qstate(obtained_state).dag()) / 2
            obtained_state = obtained_state / np.trace(obtained_state)
            fidelity = primaryfn.QFidelity(desired_state, obtained_state)
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
            gamma = channel_efficiency(L, loss=loss)
        else:
            gamma = channel_efficiency(L)
        K = amp_damp_kraus(gamma)
        kraus_op_2q = [np.kron(np.kron(np.kron(np.eye(2), k1), k2), np.eye(2))
                       for k1 in K for k2 in K]
        rho1 = sum(K @ rhoin @ K.conj().T for K in kraus_op_2q) 
        rho1 = rho1 / np.trace(rho1)
        # --- Dephasing ---
        lam_t = dephasing_coeff(T_p, T_dp, L)
        lam_t_coeff1 = (1 - lam_t) * np.eye(16)
        lam_t_coeff2 = lam_t * (np.kron(np.kron(np.eye(2), np.kron(primaryfn.Pauli.Z, primaryfn.Pauli.Z)), np.eye(2)))
        rho2 = lam_t_coeff1 @ rho1 @ lam_t_coeff1.conj().T + \
               lam_t_coeff2 @ rho1 @ lam_t_coeff2.conj().T
        rho2 = rho2 / np.trace(rho2)
        # --- Detector noise (clicking) ---
        alpha = clicking_coeff(eta, p_d)
        alpha_coeff1 = alpha * np.eye(16)
        alpha_coeff2 = ((1 - alpha) / 2) * np.eye(16)
        rho3 = alpha_coeff1 @ rho2 @ alpha_coeff1.conj().T + \
               alpha_coeff2 @ rho2 @ alpha_coeff2.conj().T
        rho3 = rho3 / np.trace(rho3)
        # --- Fidelity with expected channels ---
        fidelities = [primaryfn.QFidelity(rho3, c) for c in swap_pairs_matrix]
        j = int(np.argmax(fidelities))
        # --- Measurement and correction ---
        results = []
        for i, M in enumerate(bell_states):
            correction = pauli_corrections[j][i]
            opr = np.kron(np.kron(np.eye(2), primaryfn.Qstate(M).dag()),correction)
            post_meas = opr @ rho3 @ opr.conj().T
            norm = np.trace(post_meas)
            prob = round(float(np.real(norm)), 2)
            if norm == 0:
                continue
            obtained_state = post_meas / norm
            # enforce Hermiticity & normalization
            obtained_state = (obtained_state + primaryfn.Qstate(obtained_state).dag()) / 2
            obtained_state = obtained_state / np.trace(obtained_state)
            fidelity = primaryfn.QFidelity(desired_state, obtained_state)
            results.append({"measurement_outcome": f"Bell-{i+1}",
                            "probability": prob,
                            "fidelity": fidelity,
                            "obtained_entangled_state": obtained_state
                                })
        return results

class SWAPN:
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

            if all([primaryfn.MatrixDim(state).is_ket() for state in self.state_shared]):
                raise ValueError("Noise model requires density matrices, not kets.")

            if len(self.state_shared) == 1:
                raise ValueError("The shared entangled state should be greater than two.")
            elif len(self.state_shared) == 2:
                if loss is not None:
                    return eswap(self.state_shared[0], self.state_shared[1]).noise(L, T_p, T_dp, eta, p_d, loss)
                else:
                    return eswap(self.state_shared[0], self.state_shared[1]).noise(L, T_p, T_dp, eta, p_d)
            else:
                current_state = self.state_shared[0]
                for i in range(1, len(self.state_shared)):
                    if loss is not None:
                        results = eswap(current_state, self.state_shared[i]).noise(L, T_p, T_dp, eta, p_d, loss)
                    else:
                        results = eswap(current_state, self.state_shared[i]).noise(L, T_p, T_dp, eta, p_d)
                    current_state = results[0]["obtained_entangled_state"]
                return results
        # --- noiseless case ---
        if all([primaryfn.MatrixDim(state).is_ket() for state in self.state_shared]): #Check dimensions
            purity = [primaryfn.QStateAnalyzer(c) for c in self.state_shared]
            ent_val = [primaryfn.QEntangle2(d) for d in self.state_shared]
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
        elif all([primaryfn.MatrixDim(state).is_square() for state in self.state_shared]):
            purity = [primaryfn.QStateAnalyzer(c) for c in self.state_shared]
            ent_val = [primaryfn.QEntangle2(d) for d in self.state_shared]
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