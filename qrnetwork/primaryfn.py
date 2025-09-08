import numpy as np
from scipy.linalg import sqrtm, eigvals
from sympy import Matrix
from functools import reduce
# Pauli matrices
class Pauli:
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
# A class to estimate conjugate transpose (hermitian adjoint or dagger), normalization
import numpy as np

class Qstate:
    def __init__(self, state):
        self.state = np.array(state, dtype=complex)

        # Determine type
        if self.state.ndim == 1 or (self.state.ndim == 2 and self.state.shape[1] == 1):
            self.type = "ket"
            if self.state.ndim == 1:
                self.state = self.state.reshape(-1, 1)  # enforce column vector
        elif self.state.ndim == 2 and self.state.shape[0] == self.state.shape[1]:
            self.type = "density"
        else:
            raise ValueError("State must be either a ket (vector) or a square density matrix.")

    def dag(self):
        """Return Hermitian conjugate (NumPy array)."""
        return np.conjugate(self.state).T
    
    def norm(self):
        """Return normalized state (NumPy array)."""
        if self.type == "ket":
            norm = np.linalg.norm(self.state)
            return norm

        elif self.type == "density":
            tr = np.trace(self.state)
            return tr

    def unit(self):
        """Return normalized state (NumPy array)."""
        if self.type == "ket":
            norm = np.linalg.norm(self.state)
            if norm == 0:
                raise ValueError("Cannot normalize zero ket.")
            return self.state / norm

        elif self.type == "density":
            tr = np.trace(self.state)
            if tr == 0:
                raise ValueError("Cannot normalize zero density matrix.")
            return self.state / tr

    def __repr__(self):
        return f"Qstate(type={self.type}, shape={self.state.shape})"

class Qprod:
    @staticmethod
    def inner(state1, state2=None):
        if state2 is None:
            state2 = state1
        else:
            state2 = state2
        return (np.conjugate(state1).T @ state2)

    @staticmethod
    def outer(state1, state2=None):
        if state2 is None:
            state2 = state1
        else:
            state2 = state2
        return state1 @ np.conjugate(state2).T

         
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
        state = Qstate(state).unit()
        # Reshape into a column vector
        return state.reshape(-1, 1)

    def mixed(self):
        """
        Generate a random n-qubit mixed state (density matrix).
        """
        A = np.random.randn(self.dim, self.dim) + 1j * np.random.randn(self.dim, self.dim)
        rho = A @ Qstate(A).dag()
        rho = Qstate(rho).unit()  # normalize trace
        return rho
# random impure bell state
class Belld:
    @staticmethod
    def pure(bell_state):
        bell_pure = bell_state @ Qstate(bell_state).dag() #np.outer does the same
        return bell_pure
    @staticmethod
    def impure(bell_state):
        bell_pure = Belld.pure(bell_state)
        I4 = np.eye(4) / 4
        # Keep generating v until v > 1/3
        while True:
            v = round(np.random.random(), 2)
            if v > 1/3:
                break
        return v * bell_pure + (1 - v) * I4

# Define a class to compute quantum fidelity
class QFidelity:
    def __new__(cls, state1, state2, tol=1e-10):
        dim1, dim2 = MatrixDim(state1), MatrixDim(state2)

        # Case 1: both are pure states (kets)
        if dim1.is_ket() and dim2.is_ket():
            value = Qprod.inner(state1, state2)  # <psi|phi>
            fidelity = np.abs(value) ** 2
            return fidelity.item()

        # Case 2: both are density matrices
        elif dim1.is_square() and dim2.is_square():
            sqrt_rho = sqrtm(state1)
            inner = sqrt_rho @ state2 @ sqrt_rho
            sqrt_inner = sqrtm(inner)
            fidelity = np.real(np.trace(sqrt_inner))
            return round(fidelity, 4)

        else:
            raise ValueError("Fidelity requires both inputs as kets or both as density matrices.")


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


# A class for to check normalization and purity of a quantum states
class QStateAnalyzer:
    def __new__(cls, state, tol=1e-10, auto_normalize=True):
        dim = MatrixDim(state)

        if dim.is_ket():
            norm = np.real(Qstate(state).dag() @ state)
            if np.abs(norm - 1.0) >= tol:
                if auto_normalize:
                    state = state / np.sqrt(norm)
                else:
                    raise ValueError("State vector not normalized.")
            return state  # return normalized ket vector

        elif dim.is_square():
            trace_val = np.real(np.trace(state))
            if np.abs(trace_val - 1.0) >= tol:
                if auto_normalize:
                    state = state / trace_val
                else:
                    raise ValueError("Density matrix not trace-normalized.")
            return state  # return normalized density matrix

        else:
            raise ValueError("Input must be a ket (n×1) or square density matrix (n×n).")
        
# Check purity of a density matrix as ket state is already pure
class QPurity:
    def __new__(cls, state, tol=1e-10):
        if MatrixDim(state).is_ket():
            return 1.0

        elif MatrixDim(state).is_square():
            # Compute purity directly, no normalization
            purity = np.real(np.trace(state @ state))
            return purity

        else:
            raise ValueError("State must be a ket or a density matrix.")

# A class to check entanglement criteria for 2-qubit states
class QEntangle2:
    def __new__(cls, state, tol=1e-10):
        dim = MatrixDim(state)
        # Case 1: pure 2-qubit state (ket)
        if dim.is_ket():
            if state.shape[0] != 4:
                raise ValueError("Ket must be a 4x1 vector (2 qubits).")
            a, b, c, d = state.flatten()
            val = np.abs(np.real(a * d - b * c))
            return round(val, 4)  # 0 ⇒ separable, ≠0 ⇒ entangled

        # Case 2: mixed 2-qubit state (density matrix)
        elif dim.is_square():
            if state.shape != (4, 4):
                raise ValueError("Density matrix must be 4x4 (2 qubits).")
            sy_sy = np.kron(Pauli.Y, Pauli.Y)
            rho_star = np.conj(state)
            rho_tilde = sy_sy @ rho_star @ sy_sy
            R = sqrtm(sqrtm(state) @ rho_tilde @ sqrtm(state))
            eigenvals = np.sort(np.real(eigvals(R)))[::-1]
            concurrence_val = max(0, eigenvals[0] - np.sum(eigenvals[1:]))
            return round(concurrence_val, 4)

        else:
            raise ValueError("Input must be a 2-qubit ket (4x1) or 2-qubit density matrix (4x4).")
