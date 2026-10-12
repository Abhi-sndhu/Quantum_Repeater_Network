import numpy as np
from functools import reduce
from .primaryfn import Q

def factory_protocol(
    link, 
    dephasing=False, 
    Lambda=None, 
    rounds=None, 
    verbose=False
):
    """
    Factory protocol for entanglement swapping in a quantum star network.
    Supports both ideal execution and pure dephasing noise parameterized by link generation rounds.
    """
    # ----------------------------------------------------
    # 1. Operators and Initial State Setup
    # ----------------------------------------------------
    N = 2 * link  # total number of qubits
    I=Q.Pauli.I
    X=Q.Pauli.X
    Z=Q.Pauli.Z
    zero = np.array([[1], [0]], dtype=complex)
    one = np.array([[0], [1]], dtype=complex)

    zero_zero = np.kron(zero, zero)
    one_one = np.kron(one, one)
    
    eta= Q.Bell.phi_plus
    rho_0 = np.outer(eta, eta.conj())
    
    rho_link = [rho_0.copy() for _ in range(link)]

    # ----------------------------------------------------
    # 2. Dephasing Noise (Optional)
    # ----------------------------------------------------
    if dephasing:
        if Lambda is None:
            raise ValueError("Parameter 'Lambda' is required when dephasing=True.")
        if rounds is None:
            rounds = [0] * link
        elif len(rounds) != link:
            raise ValueError(f"Length of 'rounds' ({len(rounds)}) must match 'link' ({link}).")

        t_max = max(rounds)
        w = [t_max - r for r in rounds]

        if verbose:
            print("rounds in which links are generated :", rounds)
            print("waiting time for each link :", w)

        K_0 = np.sqrt((1 + Lambda) / 2) * I
        K_1 = np.sqrt((1 - Lambda) / 2) * Z
        K0_I = np.kron(K_0, I)
        K1_I = np.kron(K_1, I)

        for i in range(link):
            for _ in range(w[i]):
                rho_link[i] = (
                    K0_I @ rho_link[i] @ K0_I.conj().T + 
                    K1_I @ rho_link[i] @ K1_I.conj().T
                )

    # Composite state before gates
    rho = reduce(np.kron, rho_link)

    # ----------------------------------------------------
    # 3. Entangling CNOT Operations
    # ----------------------------------------------------
    P_00 = np.outer(zero, zero.conj())
    P_11 = np.outer(one, one.conj())

    opr1 = [I] * N
    opr1[1] = P_00
    A = reduce(np.kron, opr1)

    for i in range(2, N, 2):
        if i + 1 < N:
            opr2 = [I] * N
            opr2[1] = P_11
            opr2[i + 1] = X
            cnot_gate = A + reduce(np.kron, opr2)
            rho = cnot_gate @ rho @ cnot_gate.conj().T

    # ----------------------------------------------------
    # 4. Spoke Ancilla Measurements & Feedforward X Corrections
    # ----------------------------------------------------
    t = [j + 1 for j in range(2, N, 2) if j + 1 < N]

    for step_idx, qubit_idx in enumerate(t):
        m1 = [I] * N
        m2 = [I] * N
        m1[qubit_idx] = P_00
        m2[qubit_idx] = P_11
        M_0 = reduce(np.kron, m1)
        M_1 = reduce(np.kron, m2)

        p_0 = np.real(np.trace(M_0 @ rho).item())
        p_1 = np.real(np.trace(M_1 @ rho).item())
        p_sum = p_0 + p_1
        prob = [p_0 / p_sum, p_1 / p_sum]

        outcome = np.random.choice(["0", "1"], p=prob)
        if verbose:
            print(f"Measurement outcome of q{qubit_idx} qubit : {outcome}")

        if outcome == "0":
            rho = (M_0 @ rho @ M_0.conj().T) / prob[0]
        else:
            rho = (M_1 @ rho @ M_1.conj().T) / prob[1]

        if int(outcome) == 1:
            x_gate_ops = [I] * N
            x_gate_ops[qubit_idx - 1] = X
            X_m = reduce(np.kron, x_gate_ops)
            rho = X_m @ rho @ X_m.conj().T

    # ----------------------------------------------------
    # 5. Hub Hadamard, Measurement & Feedforward Z Correction
    # ----------------------------------------------------
    h = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
    H_ops = [I] * N
    H_ops[1] = h
    H = reduce(np.kron, H_ops)
    rho = H @ rho @ H.conj().T

    z1_ops = [I] * N
    z2_ops = [I] * N
    z1_ops[1] = P_00
    z2_ops[1] = P_11
    z1 = reduce(np.kron, z1_ops)
    z2 = reduce(np.kron, z2_ops)

    p0_q1 = np.real(np.trace(z1 @ rho).item())
    p1_q1 = np.real(np.trace(z2 @ rho).item())
    prob_q1 = np.array([p0_q1, p1_q1])
    prob_q1 = prob_q1 / np.sum(prob_q1)

    outcome_q1 = np.random.choice(["0", "1"], p=prob_q1)
    if verbose:
        print("Measurement outcome of q1 qubit :", outcome_q1)

    if outcome_q1 == "0":
        rho = (z1 @ rho @ z1.conj().T) / prob_q1[0]
    else:
        rho = (z2 @ rho @ z2.conj().T) / prob_q1[1]

    if int(outcome_q1) == 1:
        z_ops = [I] * N
        z_ops[0] = Z
        Z_gate = reduce(np.kron, z_ops)
        rho = Z_gate @ rho @ Z_gate.conj().T

    # ----------------------------------------------------
    # 6. Partial Trace
    # ----------------------------------------------------
    dims = [2] * N
    trace_qubits = set([1] + t)
    keep_qubits = [k for k in range(N) if k not in trace_qubits]

    rho_tensor = rho.reshape(dims + dims)
    row_idx = list(range(N))
    col_idx = list(range(N, 2 * N))

    for q in trace_qubits:
        col_idx[q] = row_idx[q]

    input_indices = row_idx + col_idx
    output_indices = [row_idx[k] for k in keep_qubits] + [col_idx[k] for k in keep_qubits]

    rho_final_tensor = np.einsum(rho_tensor, input_indices, output_indices)
    target_dim = 2 ** len(keep_qubits)
    rho_final = rho_final_tensor.reshape((target_dim, target_dim))

    return rho_final
