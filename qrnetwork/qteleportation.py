from . import primaryfn #in a package, it should be relative import
#---import primaryfn--- #Direct import
import numpy as np
from sympy import Matrix
import pprint
# import time
# Basis kets
ket0 = np.array([[1.0], [0.0]], dtype=complex)  # |0>
ket1 = np.array([[0.0], [1.0]], dtype=complex)  # |1>
# Computational basis for two qubits
basis_comp = [np.eye(4, dtype=complex)[:, i].reshape(-1, 1) for i in range(4)]
# (reshape(-1,1) ensures column vector shape (4,1))
# Bell state
class Bell:
    phi_plus  = np.array([[1.0], [0.0], [0.0], [1.0]], dtype=complex) / np.sqrt(2)
    phi_minus = np.array([[1.0], [0.0], [0.0], [-1.0]], dtype=complex) / np.sqrt(2)
    psi_plus  = np.array([[0.0], [1.0], [1.0], [0.0]], dtype=complex) / np.sqrt(2)
    psi_minus = np.array([[0.0], [1.0], [-1.0], [0.0]], dtype=complex) / np.sqrt(2)

bell_states= [Bell.phi_plus, Bell.phi_minus, Bell.psi_plus, Bell.psi_minus]

# Density matrix form of the Bell States
qBellChannel = [(b @ primaryfn.Qstate(b).dag()) for b in bell_states]

# Construct unitary matrix that maps Bell basis to computational basis
local_unitary = np.zeros((4, 4), dtype=complex)
for i in range(4):
    local_unitary += primaryfn.Qprod.outer(basis_comp[i], bell_states[i])

# Pauli correction table for quantum teleportation protocol
pauli_corrections = [
    [primaryfn.Pauli.I, primaryfn.Pauli.Z, primaryfn.Pauli.X, 1j*primaryfn.Pauli.Y],
    [primaryfn.Pauli.Z, primaryfn.Pauli.I, 1j*primaryfn.Pauli.Y, primaryfn.Pauli.X],
    [primaryfn.Pauli.X, 1j*primaryfn.Pauli.Y, primaryfn.Pauli.I, primaryfn.Pauli.Z],
    [1j*primaryfn.Pauli.Y, primaryfn.Pauli.X, primaryfn.Pauli.Z, primaryfn.Pauli.I]
]
# Function define to teleport a pure state
def teleport_ket_state(state,qchannel):
    input_state = np.kron(state, qchannel)
    fidelities = [primaryfn.QFidelity(qchannel, c) for c in bell_states]
    j = int(np.argmax(fidelities))
    results = []  # List to store results for each measurement outcome
    for i, bell_M in enumerate(bell_states):
        correction = pauli_corrections[j][i]
        post_meas = np.kron(primaryfn.Qstate(bell_M).dag(), correction) @ input_state
        prob = np.real(primaryfn.Qprod.inner(post_meas, post_meas).item())
        norm = np.sqrt(prob)
        if norm == 0:
            continue
        obtained_state = post_meas / norm
        fidelity = primaryfn.QFidelity(state, obtained_state)
        results.append({
            "state_obtained": Matrix(obtained_state),
            "measurement_outcome": f"Bell-{i+1}",
            "probability": round(prob,2),
            "fidelity": round(fidelity,4)
            })
    return results

# Function define to teleport a density matrix state
def teleport_density_state(state,qchannel):
    input_state = np.kron(state, qchannel)
    fidelities = [primaryfn.QFidelity(qchannel, c) for c in qBellChannel]
    j = int(np.argmax(fidelities))
    results = []  # List to store results for each measurement outcome
    for i, M in enumerate(bell_states):
        correction = pauli_corrections[j][i]
        post_meas = np.kron(primaryfn.Qstate(M).dag(), correction) @ input_state @ np.kron(M, correction)
        norm = np.trace(post_meas)
        prob = round(np.real(norm), 2)
        if norm == 0:
            continue       
        obtained_state = post_meas / norm
        fidelity = primaryfn.QFidelity(obtained_state, state)
        results.append({
            "state_obtained": Matrix(obtained_state),
            "measurement_outcome": f"Bell-{i+1}",
            "probability": prob,
            "fidelity": round(fidelity, 2)
            })
    return results
# Teleportation function    
class QTeleportation:
    def __init__(self, qS, qC, tol=1e-10):
        self.tol = tol
        self.results = self.run(qS, qC)
    # ---------- Helpers ----------
    def __repr__(self):
        return "QTeleportation Results:\n" + pprint.pformat(self.results, indent=4)
    def run(self, qS, qC):
        # start = time.perf_counter()
        if primaryfn.MatrixDim(qS).is_ket() and primaryfn.MatrixDim(qC).is_ket():
            state = primaryfn.QStateAnalyzer(qS)
            channel = primaryfn.QStateAnalyzer(qC)
            entangled = primaryfn.QEntangle2(channel)
            if entangled > self.tol: 
                output = teleport_ket_state(state, channel)
                # end = time.perf_counter()
                # elapsed = end - start
                # print(f"Execution time: {elapsed:.6f} seconds")
                return {
                    "results": output,
                    "Local unitary matrix": Matrix(local_unitary)
                }
            else:
                raise ValueError("Your quantum channel is not entangled.")
        elif primaryfn.MatrixDim(qS).is_square() and primaryfn.MatrixDim(qC).is_square():
            state = primaryfn.QStateAnalyzer(qS)
            channel = primaryfn.QStateAnalyzer(qC)
            conc = primaryfn.QEntangle2(channel)
            if np.abs(conc)!=self.tol: #np.isclose(conc, 1.0)
                output = teleport_density_state(state, channel)
                # end = time.perf_counter()
                # elapsed = end - start
                # print(f"Execution time: {elapsed:.6f} seconds")
                return {
                    "results": output,
                    "Local unitary matrix": Matrix(local_unitary)
                }
            else:
                raise ValueError("Your quantum channel is not entangled.")
        else:
            raise ValueError("Incorrect dimensions for quantum state or quantum channel.")