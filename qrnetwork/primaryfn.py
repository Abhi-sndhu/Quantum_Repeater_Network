import numpy as np
from scipy.linalg import sqrtm, eigvals
from sympy import Matrix
from functools import reduce

# Pauli matrices
I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])

# Generate random n-qubit pure and mixed state
class QStateGenerator:
    def __init__(self, n):
        """
        Initialize the generator for n qubits.
        """
        self.n = n
        self.dim = 2 ** n

    def pure(self):
        """
        Generate a random n-qubit pure state.
        """
        real_part = np.random.randn(self.dim)
        imag_part = np.random.randn(self.dim)
        state = (real_part + 1j * imag_part) 
        # Normalize (so that <ψ|ψ> = 1)       
        state /= np.linalg.norm(state)
        # Reshape into a column vector
        return state.reshape(-1, 1)

    def mixed(self):
        """
        Generate a random n-qubit mixed state (density matrix).
        """
        A = np.random.randn(self.dim, self.dim) + 1j * np.random.randn(self.dim, self.dim)
        rho = A @ A.conj().T
        rho /= np.trace(rho)  # normalize trace
        return rho
    
class Bell:
    @staticmethod
    def impure(bell_state):
        bell_pure = bell_state @ bell_state.conj().T #np.outer does the same
        I4 = np.eye(4) / 4
        # Keep generating v until v > 1/3
        while True:
            v = round(np.random.random(), 2)
            if v > 1/3:
                break
        return v * bell_pure + (1 - v) * I4

# Define a class to compute quantum fidelity
class QFidelity:
    @staticmethod
    def pure(psi, phi):  # For pure state: F(|psi>, |phi>) = |<psi|phi>|
        value = np.vdot(psi, phi) #vdot does both conjugate and transpose
        fidelity = np.abs(value)**2
        return round(fidelity, 2)
    @staticmethod
    def density(rho, sigma):
        sqrt_rho = sqrtm(rho)
        inner = sqrt_rho @ sigma @ sqrt_rho
        sqrt_inner = sqrtm(inner)
        fidelity = np.real(np.trace(sqrt_inner))
        return round(fidelity, 2)


# A class for to check normalization and purity of a quantum states
class QStateAnalyzer:
    def __init__(self, tol=1e-10, auto_normalize=True):
        self.tol = tol
        self.auto_normalize = auto_normalize
    # ket_vector state is pure. No need to check purity.
    def ket(self, psi):
        norm = np.real(psi.conj().T @ psi) 
        if np.abs(norm - 1.0) >= self.tol:
            if self.auto_normalize:
                psi = psi / np.sqrt(norm)  # normalize
            else:
                raise ValueError("State vector not normalized.")
        #rho_psi = np.outer(psi, psi.conj())
        #purity = np.real(np.trace(rho_psi @ rho_psi))
        state = psi
        return state

    def density(self, rho):
        trace_val = np.real(np.trace(rho))
        if np.abs(trace_val - 1.0) >= self.tol:
            if self.auto_normalize:
                rho = rho / trace_val
            else:
                raise ValueError("Density matrix not normalized.")
        purity = np.real(np.trace(rho @ rho))
        return purity

# A class to check entanglement criteria for 2-qubit states
class EntanglementCriteria2qubit:
    @staticmethod
    def ket(psi):
        """
        Check entanglement for a 2-qubit pure state |ψ>.
        psi must be a 4x1 column vector.
        Uses determinant method: ad - bc (Schmidt rank test).
        """
        if psi.shape[0] != 4:
            raise ValueError("State vector must be of size 4 (2 qubits).")
        # Extract coefficients directly
        a, b, c, d = psi.flatten()
        # Entanglement witness (ad - bc)
        val = np.abs(np.real(a * d - b * c))
        return round(val, 4)  # 0 ⇒ separable, ≠0 ⇒ entangled
    @staticmethod
    def density(rho):
        """
        Compute concurrence for a 2-qubit mixed state ρ.
        """
        if rho.shape != (4, 4):
            raise ValueError("Density matrix must be 4x4 (2 qubits).")
        sy_sy = np.kron(Y, Y)
        rho_star = np.conj(rho)
        rho_tilde = sy_sy @ rho_star @ sy_sy
        R = sqrtm(sqrtm(rho) @ rho_tilde @ sqrtm(rho))
        eigenvals = np.sort(np.real(eigvals(R)))[::-1]
        concurrence_val = max(0, eigenvals[0] - np.sum(eigenvals[1:]))
        return round(concurrence_val, 4)