from . import primaryfn # in a package, it should be relative import
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

bell_states= [primaryfn.Q.Bell.phi_plus, primaryfn.Q.Bell.phi_minus, 
              primaryfn.Q.Bell.psi_plus, primaryfn.Q.Bell.psi_minus]

# Density matrix form of the Bell States
qBellChannel = [(b @ primaryfn.Q(b).d()) for b in bell_states]

# Construct unitary matrix that maps Bell basis to computational basis
local_unitary = np.zeros((4, 4), dtype=complex)
for i in range(4):
    local_unitary += primaryfn.Qp(basis_comp[i], bell_states[i]).op()

# Pauli correction table for quantum teleportation protocol
pauli_corrections = [
    [primaryfn.Q.Pauli.I, primaryfn.Q.Pauli.Z, primaryfn.Q.Pauli.X, 1j*primaryfn.Q.Pauli.Y],
    [primaryfn.Q.Pauli.Z, primaryfn.Q.Pauli.I, 1j*primaryfn.Q.Pauli.Y, primaryfn.Q.Pauli.X],
    [primaryfn.Q.Pauli.X, 1j*primaryfn.Q.Pauli.Y, primaryfn.Q.Pauli.I, primaryfn.Q.Pauli.Z],
    [1j*primaryfn.Q.Pauli.Y, primaryfn.Q.Pauli.X, primaryfn.Q.Pauli.Z, primaryfn.Q.Pauli.I]
]
# Function define to teleport a pure state
def teleport_ket_state(state,qchannel):
    input_state = np.kron(state, qchannel)
    fidelities = [primaryfn.Qp(qchannel, c).fid() for c in bell_states]
    j = int(np.argmax(fidelities))
    results = []  # List to store results for each measurement outcome
    for i, bell_M in enumerate(bell_states):
        correction = pauli_corrections[j][i]
        post_meas = np.kron(primaryfn.Q(bell_M).d(), correction) @ input_state
        prob = np.real((primaryfn.Qp(post_meas, post_meas).ip()).item())
        norm = np.sqrt(prob)
        if norm == 0:
            continue
        obtained_state = post_meas / norm
        fidelity = primaryfn.Qp(state, obtained_state).fid()
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
    fidelities = [primaryfn.Qp(qchannel, c).fid() for c in qBellChannel]
    j = int(np.argmax(fidelities))
    results = []  # List to store results for each measurement outcome
    for i, M in enumerate(bell_states):
        correction = pauli_corrections[j][i]
        post_meas = np.kron(primaryfn.Q(M).d(), correction) @ input_state @ np.kron(M, correction)
        norm = np.trace(post_meas)
        prob = round(np.abs(np.real(norm)), 2) # abs because sometimes we get neagtive probability
        if norm == 0:
            continue       
        obtained_state = post_meas / norm
        fidelity = primaryfn.Qp(obtained_state, state).fid()
        results.append({
            "state_obtained": Matrix(obtained_state),
            "measurement_outcome": f"Bell-{i+1}",
            "probability": prob,
            "fidelity": round(fidelity, 2)
            })
    return results
# Teleportation function    
class Teleport:

    def __init__(self, qS, qC, tol=1e-10):

        self.qS = qS if isinstance(qS, primaryfn.Q) else primaryfn.Q(qS)
        # It checks whether qS is already an instance of the Q class. If it is, it stores it directly. 
        # Otherwise, it converts qS into a Q object by calling primaryfn.Q(qS). 
        # This allows your class to accept either a Q object or a NumPy array transparently.
        self.qC = qC if isinstance(qC, primaryfn.Q) else primaryfn.Q(qC)
        self.tol = tol
        self.results = None

    def run(self):

        state = self.qS.u()
        channel = self.qC.u()

        ent = primaryfn.Ebit(channel)

        if np.abs(ent) <= self.tol:
            raise ValueError("Quantum channel is not entangled.")

        if primaryfn.Is(state).ket():

            output = teleport_ket_state(state, channel)

        elif primaryfn.Is(state).square():

            output = teleport_density_state(state, channel)

        else:

            raise ValueError("Invalid quantum state.")

        self.results = {
            "results": output,
            "Local unitary matrix": Matrix(local_unitary)
        }

        return self.results