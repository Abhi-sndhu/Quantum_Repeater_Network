import numpy as np
from functools import reduce
from . import primaryfn
from qrnetwork import primaryfn

class PieceMaker:
    def __init__(self, state_shared, links, tol=1e-10):
        """
        Parameters
        ----------
        state_shared : A shared two-qubit density matrices.
        links : Number of elementary links.
        tol : float
            Numerical tolerance.
        """
        self.state_shared = state_shared
        self.links = links
        self.tol = tol
        self.dephasing_params = None

    def dephasing(self, Lambda, rounds=None):
        """
        Enable dephasing noise.

        Parameters
        ----------
        Lambda : float
            Dephasing parameter.

        rounds : list, optional
            Generation round associated with each elementary link.
            If None, all links are assumed to be generated
            in the same round.
        """
        self.dephasing_params = {
            "Lambda": Lambda,
            "rounds": rounds
        }
        return self

    def run(self, verbose=False):
        """
        Perform entanglement swapping using the PieceMaker protocol.

        Returns
        -------
        numpy.ndarray
            Final two-qubit density matrix.
        """

        # ----------------------------------------------------
        # 1. Input validation
        # ----------------------------------------------------

    
        if len(self.state_shared) > 2:
            raise ValueError(
                "The shared entangled state should be a single two-qubit entangled state."
            )

        # Check whether all states are density matrices
        if not all(
            primaryfn.Is(state).square()
            for state in self.state_shared
        ):
            raise ValueError(
                "PieceMaker protocol requires density matrices."
            )

        # Check entanglement
        ent_val = [
            primaryfn.Ebit(state)
            for state in self.state_shared
        ]

        if not all(np.abs(val) > self.tol for val in ent_val):
            raise ValueError(
                "The shared states are not entangled."
            )

        # ----------------------------------------------------
        # 2. Operators
        # ----------------------------------------------------

        zero = np.array([[1], [0]], dtype=complex)
        one = np.array([[0], [1]], dtype=complex)

        I = Q.Pauli.I
        X = Q.Pauli.X
        Y = Q.Pauli.Y
        Z = Q.Pauli.Z

        P_00 = np.outer(zero, zero.conj())
        P_11 = np.outer(one, one.conj())

        base = [zero, one]

        # CNOT gate
        CNOT = (
            np.kron(P_00, I)
            + np.kron(P_11, X)
        )

        # SWAP gate
        swap = 0.5 * (
            np.kron(I, I)
            + np.kron(Z, Z)
            + np.kron(Y, Y)
            + np.kron(X, X)
        )

        # ----------------------------------------------------
        # 3. Copy input states
        # ----------------------------------------------------

        link = len(self.state_shared)

        rho_link = [
            state.copy()
            for state in self.state_shared
        ]

        # ----------------------------------------------------
        # 4. Dephasing Noise
        # ----------------------------------------------------

        if self.dephasing_params is not None:

            Lambda = self.dephasing_params["Lambda"]
            rounds = self.dephasing_params["rounds"]

            if rounds is None:

                rounds = [0] * link

            elif len(rounds) != link:

                raise ValueError(
                    f"Length of 'rounds' ({len(rounds)}) "
                    f"must match number of links ({link})."
                )

            # Waiting time
            t_max = max(rounds)

            waiting_time = [
                t_max - r
                for r in rounds
            ]

            if verbose:

                print(
                    "Rounds in which links are generated:",
                    rounds
                )

                print(
                    "Waiting time for each link:",
                    waiting_time
                )

            # Dephasing Kraus operators
            K_0 = np.sqrt(
                (1 + Lambda) / 2
            ) * I

            K_1 = np.sqrt(
                (1 - Lambda) / 2
            ) * Z

            # Noise acts on the first qubit
            K0_I = np.kron(K_0, I)

            K1_I = np.kron(K_1, I)

            for i in range(link):

                for _ in range(waiting_time[i]):

                    rho_link[i] = (
                        K0_I
                        @ rho_link[i]
                        @ K0_I.conj().T
                        +
                        K1_I
                        @ rho_link[i]
                        @ K1_I.conj().T
                    )

        # ----------------------------------------------------
        # 5. Sequential PieceMaker Swapping
        # ----------------------------------------------------

        N = 2 * link

        rho = rho_link[0]

        remaining_links = [
            rho_link[k]
            for k in range(1, link)
        ]

        d = 0
        e = 2

        measurement_results = []

        for i in range(N - 1):

            dim = 2 * (i + 2) - d

            if dim <= N - d:

                # Add next elementary link
                rho = np.kron(
                    rho,
                    remaining_links[i]
                )

                # --------------------------------------------
                # SWAP operation
                # --------------------------------------------

                b = [I] * (dim - 1)

                b[-2] = swap

                P = reduce(
                    np.kron,
                    b
                )

                rho = (
                    P
                    @ rho
                    @ P.conj().T
                )

                # --------------------------------------------
                # CNOT operation
                # --------------------------------------------

                c = [I] * (dim - 1)

                c[-1] = CNOT

                Cx = reduce(
                    np.kron,
                    c
                )

                rho = (
                    Cx
                    @ rho
                    @ Cx.conj().T
                )

                # --------------------------------------------
                # Measurement operators
                # --------------------------------------------

                m_0 = [I] * dim

                m_0[-1] = P_00

                M_0 = reduce(
                    np.kron,
                    m_0
                )

                m_1 = [I] * dim

                m_1[-1] = P_11

                M_1 = reduce(
                    np.kron,
                    m_1
                )

                p_0 = np.real(
                    np.trace(
                        M_0 @ rho
                    ).item()
                )

                p_1 = np.real(
                    np.trace(
                        M_1 @ rho
                    ).item()
                )

                p_sum = p_0 + p_1

                if p_sum <= self.tol:

                    raise ValueError(
                        "Invalid measurement probabilities "
                        "encountered during PieceMaker protocol."
                    )

                probabilities = [
                    p_0 / p_sum,
                    p_1 / p_sum
                ]

                # --------------------------------------------
                # Measurement
                # --------------------------------------------

                outcome = np.random.choice(
                    ["0", "1"],
                    p=probabilities
                )

                measurement_results.append(
                    int(outcome)
                )

                if verbose:

                    print(
                        f"Measurement outcome "
                        f"(step {i + 1}): {outcome}"
                    )

                if outcome == "0":

                    rho = (
                        M_0
                        @ rho
                        @ M_0.conj().T
                    ) / p_0

                else:

                    rho = (
                        M_1
                        @ rho
                        @ M_1.conj().T
                    ) / p_1

                # --------------------------------------------
                # Feedforward X operation
                # --------------------------------------------

                b_val = int(outcome)

                if b_val == 1:

                    X_m = [I] * dim

                    X_m[-3] = X

                    X_m = reduce(
                        np.kron,
                        X_m
                    )

                    rho = (
                        X_m
                        @ rho
                        @ X_m.conj().T
                    )

                # --------------------------------------------
                # Remove measured qubit
                # --------------------------------------------

                M = [I] * dim

                M[-1] = base[b_val].conj().T

                M = reduce(
                    np.kron,
                    M
                )

                rho = (
                    M
                    @ rho
                    @ M.conj().T
                )

                d = i + 1

                e = dim - 1

        # ----------------------------------------------------
        # 6. Central Hub Hadamard
        # ----------------------------------------------------

        h = (
            1 / np.sqrt(2)
        ) * np.array(
            [
                [1, 1],
                [1, -1]
            ],
            dtype=complex
        )

        H = [I] * e

        H[-1] = h

        H_gate = reduce(
            np.kron,
            H
        )

        rho = (
            H_gate
            @ rho
            @ H_gate.conj().T
        )

        # ----------------------------------------------------
        # 7. Central Measurement
        # ----------------------------------------------------

        m1 = [I] * e

        m1[-1] = P_00

        M_0_q2 = reduce(
            np.kron,
            m1
        )

        m2 = [I] * e

        m2[-1] = P_11

        M_1_q2 = reduce(
            np.kron,
            m2
        )

        p_0 = np.real(
            np.trace(
                M_0_q2 @ rho
            ).item()
        )

        p_1 = np.real(
            np.trace(
                M_1_q2 @ rho
            ).item()
        )

        p_sum = p_0 + p_1

        if p_sum <= self.tol:

            raise ValueError(
                "Invalid central measurement probabilities."
            )

        prob_q2 = [
            p_0 / p_sum,
            p_1 / p_sum
        ]

        outcome_q2 = np.random.choice(
            ["0", "1"],
            p=prob_q2
        )

        if verbose:

            print(
                "Measurement outcome of central qubit:",
                outcome_q2
            )

        if outcome_q2 == "0":

            rho = (
                M_0_q2
                @ rho
                @ M_0_q2.conj().T
            ) / p_0

        else:

            rho = (
                M_1_q2
                @ rho
                @ M_1_q2.conj().T
            ) / p_1

        # ----------------------------------------------------
        # 8. Feedforward Z correction
        # ----------------------------------------------------

        b2 = int(outcome_q2)

        if b2 == 1:

            Z1_m = [I] * e

            Z1_m[0] = Z

            Z1 = reduce(
                np.kron,
                Z1_m
            )

            rho = (
                Z1
                @ rho
                @ Z1.conj().T
            )

        # ----------------------------------------------------
        # 9. Remove measured central qubit
        # ----------------------------------------------------

        M_final = [I] * e

        M_final[-1] = base[b2].conj().T

        M_final = reduce(
            np.kron,
            M_final
        )

        rho = (
            M_final
            @ rho
            @ M_final.conj().T
        )

        # ----------------------------------------------------
        # 10. Return results in QRep-style format
        # ----------------------------------------------------

        return [
            {
                "obtained_entangled_state": rho,
                "measurement_results": measurement_results,
                "central_measurement": b2
            }
        ]