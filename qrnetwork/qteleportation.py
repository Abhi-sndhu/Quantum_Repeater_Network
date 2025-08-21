from . import primaryfn #in a package, it should be relative import
#---import primaryfn--- #Direct import
import numpy as np
from sympy import Matrix
import pprint
# Basis kets
ket0 = np.array([[1.0], [0.0]], dtype=complex)  # |0>
ket1 = np.array([[0.0], [1.0]], dtype=complex)  # |1>
# Computational basis for two qubits
basis_comp = [np.eye(4, dtype=complex)[:, i].reshape(-1, 1) for i in range(4)]
# (reshape(-1,1) ensures column vector shape (4,1))
# Bell state
phi_plus  = np.array([[1.0], [0.0], [0.0], [1.0]], dtype=complex) / np.sqrt(2)
phi_minus = np.array([[1.0], [0.0], [0.0], [-1.0]], dtype=complex) / np.sqrt(2)
psi_plus  = np.array([[0.0], [1.0], [1.0], [0.0]], dtype=complex) / np.sqrt(2)
psi_minus = np.array([[0.0], [1.0], [-1.0], [0.0]], dtype=complex) / np.sqrt(2)
bell_states= [phi_plus, phi_minus, psi_plus, psi_minus]
              
# Density matrix form of the Bell States
qBellChannel = [(b @ b.conj().T) for b in bell_states]

# Construct unitary matrix that maps Bell basis to computational basis
local_unitary = np.zeros((4, 4), dtype=complex)
for i in range(4):
    local_unitary += np.outer(basis_comp[i], bell_states[i])

# Pauli matrices
I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])

# Pauli correction table for quantum teleportation protocol
pauli_corrections = [
    [I, Z, X, 1j*Y],
    [Z, I, 1j*Y, X],
    [X, 1j*Y, I, Z],
    [1j*Y, X, Z, I]
]
# Class to check given matrix is column or square matrix.
class MatrixDim:
    def __init__(self, matrix):
        self.matrix = matrix
        self.rows, self.cols = self.matrix.shape

    def is_ket(self):
        """Check if matrix is a column vector (n×1)."""
        return self.cols == 1

    def is_square(self):
        """Check if matrix is square (n×n)."""
        return self.rows == self.cols
# Function define to teleport a pure state
def teleport_ket_state(state,qchannel):
    input_state = np.kron(state, qchannel)
    fidelities = [primaryfn.QFidelity.pure(qchannel, c) for c in bell_states]
    j = int(np.argmax(fidelities))
    results = []  # List to store results for each measurement outcome
    for i, bell_M in enumerate(bell_states):
        correction = pauli_corrections[j][i]
        post_meas = np.kron(bell_M.conj().T,correction) @ input_state
        prob = round(np.real(np.vdot(post_meas, post_meas)),2)
        norm = np.sqrt(prob)
        if norm == 0:
            continue
        obtained_state = post_meas / norm
        fidelity = primaryfn.QFidelity.pure(state, obtained_state)
        results.append({
            "state_obtained": Matrix(obtained_state),
            "measurement_outcome": f"Bell-{i+1}",
            "probability": prob,
            "fidelity": round(fidelity, 2)
            })
    return results

# Function define to teleport a density matrix state
def teleport_density_state(state,qchannel):
    input_state = np.kron(state, qchannel)
    fidelities = [primaryfn.QFidelity.density(qchannel, c) for c in qBellChannel]
    j = int(np.argmax(fidelities))
    results = []  # List to store results for each measurement outcome
    for i, M in enumerate(bell_states):
        correction = pauli_corrections[j][i]
        post_meas = np.kron(M.conj().T, correction) @ input_state @ np.kron(M, correction)
        norm = np.trace(post_meas)
        prob = round(np.real(norm), 2)
        if norm == 0:
            continue       
        obtained_state = post_meas / norm
        fidelity = primaryfn.QFidelity.density(state, obtained_state)
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
        if MatrixDim(qS).is_ket() and MatrixDim(qC).is_ket():
            state = primaryfn.QStateAnalyzer().ket(qS)
            channel = primaryfn.QStateAnalyzer().ket(qC)
            entangled = primaryfn.EntanglementCriteria2qubit.ket(channel)
            if entangled >= self.tol: 
                output = teleport_ket_state(qS,qC)
                return {
                    "results": output,
                    "Local unitary matrix": Matrix(local_unitary)
                }
            else:
                raise ValueError("Your quantum channel is not entangled.")
        elif MatrixDim(qS).is_square() and MatrixDim(qC).is_square():
            purity_qS = primaryfn.QStateAnalyzer().density(qS)
            purity_qC = primaryfn.QStateAnalyzer().density(qC)
            conc = primaryfn.EntanglementCriteria2qubit.density(qC)
            if np.abs(conc)!=self.tol: #np.isclose(conc, 1.0)
                output = teleport_density_state(qS,qC)
                return {
                    "results": output,
                    "Local unitary matrix": Matrix(local_unitary)}
            else:
                raise ValueError("Your quantum channel is not entangled.")
        else:
            raise ValueError("Incorrect dimensions for quantum state or quantum channel.")