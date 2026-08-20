import numpy as np
from scipy.linalg import sqrtm, eigvals

class Q: # creating objects for quantum states and operators
    def __init__(self, data): 
        # __init__ method initializes the Q object with a given state or operator.
        #  It converts the input data into a NumPy array of complex numbers and stores it in the `state` attribute. 
        self.state = np.array(data, dtype=complex)

    def __array__(self, dtype=None, copy=None):
        """
        Let Q instances be consumed anywhere a NumPy array is expected
        (np.kron, `@`, np.array(), etc.) without callers needing to
        unwrap `.state` manually. This is what keeps the rest of the
        package working even though .d()/.u()/.dm() now return Q objects
        instead of raw arrays (see note below).
        """
        arr = self.state
        return arr if dtype is None else arr.astype(dtype)

    def __repr__(self):
        return f"Q(\n{self.state}\n)"

    # ----- Properties / operations -----
    # NOTE ON RETURN TYPES:
    # Previously these methods had inconsistent return types: some
    # returned a plain NumPy array/scalar while the class was clearly
    # meant to be used as a fluent API (e.g. Q(qS).u().n()). Because
    # .d(), .u() and .dm() returned a bare `np.ndarray` instead of `Q`,
    # any attempt to chain a second Q-method after them failed with
    # `'numpy.ndarray' object has no attribute '...'`.
    # Fix: these three now wrap their result in `Q(...)` so chaining
    # works. `.n()` and `.purity()` are terminal/reducing operations
    # (they return a number, not a state) so they intentionally keep
    # returning a plain float - that was correct before and is unchanged.

    def __add__(self, other): # Addition (+)
        if isinstance(other, Q):
            return Q(self.state + other.state)
        return Q(self.state + other)
        
    def __sub__(self, other): #Subtraction (-)
        if isinstance(other, Q):
            return Q(self.state - other.state)
        return Q(self.state - other)

    def __mul__(self, other): # Scalar Multiplication (*)
        if np.isscalar(other):
            return Q(self.state * other)
        raise TypeError(
            "Use '@' for matrix multiplication. '*' is reserved for scalar multiplication."
        )

    def __rmul__(self, other): # Scalar Multiplication from the Left
        return self.__mul__(other)

    def __truediv__(self, other): # Scalar Division (/)
        if np.isscalar(other):
            if other == 0:
                raise ZeroDivisionError("Division by zero.")
            return Q(self.state / other)
        raise TypeError("Division is only defined by a scalar.")

    def __matmul__(self, other): # Matrix Multiplication (@)
        if isinstance(other, Q):
            return Q(self.state @ other.state)
        return Q(self.state @ other)

    def item(self):
        """
        Return the scalar value if the quantum object contains a single element.
        """
        return self.state.item()

    def d(self): # dagger
        """Return Hermitian conjugate, wrapped as a Q instance."""
        return Q(self.state.conj().T)

    def n(self): # norm: ∥x∥=\sqrt{x^† x} ; \sqrt{Tr(A†A)}	
        norm = np.linalg.norm(self.state)
        return round(norm, 2)

    def u(self): # unit |ψ⟩/∥ψ∥ 
        """Return normalized state, wrapped as a Q instance."""
        norm = np.linalg.norm(self.state)
        if norm == 0:
            raise ValueError("Cannot normalize zero state/operator.")
        return Q(self.state / norm)

    def purity(self):
        """Compute the purity of the state (terminal op -> float)."""
        rows, cols = self.state.shape
        if cols == 1:
            return 1.0
        else:
            # Purity: Tr(rho^2)
            result = np.real(np.trace(self.state @ self.state))
            return round(result, 4)

    def dm(self): # density matrix
        """Return density matrix for a pure state (ket), wrapped as a Q instance."""
        if self.state.ndim != 2 or self.state.shape[1] != 1:
            raise ValueError("Density matrix can only be computed for pure states (column vectors).")
        return Q(self.state @ self.state.conj().T)

    # ----- Random state generators -----
    class rand:
        @staticmethod
        def haar(d):
            """
            Generate a Haar-random unitary matrix of dimension d using QR method.
            """
            A = np.random.normal(size=(d, d))
            B = np.random.normal(size=(d, d))
            Z = A + 1j * B
            Q, R = np.linalg.qr(Z) # QR decomposition of Z

            # Make Q Haar by adjusting phases
            Lambda = np.diag([R[i, i] / np.abs(R[i, i]) for i in range(d)])
            U = Q @ Lambda
            return U

        @staticmethod
        def p(n): # pure state
            """
            Haar-random pure state for n qubits.
            Returns |psi> as a column vector.
            """
            d = 2 ** n
            A = np.random.normal(0, 1/np.sqrt(2), d)
            B = np.random.normal(0, 1/np.sqrt(2), d)
            Z = A + 1j * B                 
            psi = Z / np.linalg.norm(Z)  # normalize
            return Q(psi.reshape(-1, 1))

        @staticmethod
        def m(n): # mixed state
            """
            Haar-random mixed state for n qubits.
            Hilbert–Schmidt induced measure with ancilla dimension k.
            """
            d = 2 ** n
            A = np.random.normal(0, 1/np.sqrt(2), (d, d))
            B = np.random.normal(0, 1/np.sqrt(2), (d, d))
            Z = A + 1j * B                  
            rho = Z @ Z.conj().T
            rho /= np.trace(rho)
            return Q(rho)
    
    # ----- Bell states -----
    class Bell:
        phi_plus = np.array([[1], [0], [0], [1]], dtype=complex) / np.sqrt(2)
        phi_minus = np.array([[1], [0], [0], [-1]], dtype=complex) / np.sqrt(2)
        psi_plus = np.array([[0], [1], [1], [0]], dtype=complex) / np.sqrt(2)
        psi_minus = np.array([[0], [1], [-1], [0]], dtype=complex) / np.sqrt(2)

        @classmethod
        def all(cls):
            """
            Return all four Bell states as a dictionary.
            Usage: Q.Bell.all()
            """
            return {
                'phi_plus': cls.phi_plus,
                'phi_minus': cls.phi_minus,
                'psi_plus': cls.psi_plus,
                'psi_minus': cls.psi_minus
            }
        
    class Pauli:
        I = np.eye(2, dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        @classmethod
        def all(cls):
            return {
                'PauliI': cls.I,
                'PauliX': cls.X,
                'PauliY': cls.Y,
                'PauliZ': cls.Z
            }
        
    class Werner:
        def __new__(cls, p=None, state=None):
            if state is None:
                state = Q.Bell.phi_plus
            bell_pure = state @ state.conj().T
            I4 = np.eye(4) / 4
            if p is None:
                p = np.random.uniform(low=1/2, high=1.0)
            # If p is provided, use it directly
            return Q(p * bell_pure + (1 - p) * I4)


class Is:
    def __init__(self, state, tol=1e-10, auto_normalize=True):
        self.state = np.array(state, dtype=complex)
        self.tol = tol
        self.rows, self.cols = self.state.shape

    def ket(self):
        """Check if the state is a column vector (n×1)."""
        return self.cols == 1

    def square(self):
        """Check if the state is square (n×n)."""
        return self.rows == self.cols

    def herm(self):
        """
        Check if the state is Hermitian: ρ = ρ†
        Only meaningful for square matrices (density matrices).
        """
        if not self.square():
            raise ValueError("Hermiticity is defined only for square matrices.")
        return np.allclose(self.state, self.state.conj().T, atol=self.tol)
    
    def pure(self):
        """Compute the purity of the state."""
        if self.ket():
            return True
        elif self.square():
            if not self.herm():
                raise ValueError("Density matrix must be Hermitian to test purity.")
            purity = np.real(np.trace(self.state @ self.state))
            return abs(purity - 1.0) < self.tol
        else:
            raise ValueError("Given quantum state is neither ket vector nor density matrix.")
        
class Qp:
    def __init__(self, state1, state2=None, tol=1e-10):
        self.state1 = np.array(state1, dtype=complex)
        self.state2 = np.array(state2, dtype=complex) if state2 is not None else self.state1
        self.tol = tol
    
    def ip(self): # inner product
        if Is(self.state1).ket() and Is(self.state2).ket():
            return Q(self.state1.conj().T @ self.state2)
        elif Is(self.state1).square() and Is(self.state2).square():
            return Q(np.trace(self.state1.conj().T @ self.state2))
        else:
            raise ValueError("Inputs must both be vectors (1D) or matrices (2D).")
            
    def op(self): # Outer product
        if Is(self.state1).square() or Is(self.state2).square():
            raise ValueError("Both inputs must be 1D vectors.")
        return Q(self.state1 @ self.state2.conj().T)
    
    def tp(self): # Tensor product
        return Q(np.kron(self.state1, self.state2))
    
    def fid(self): # Fidelity
        # Case 1: both are pure states (kets)
        if Is(self.state1).ket() and Is(self.state2).ket():
            value = self.ip()  # <psi|phi>
            fidelity = np.abs(value).item()
            return round(fidelity, 4)

        # Case 2: both are density matrices
        elif Is(self.state1).square() and Is(self.state2).square():
            sqrt_rho = sqrtm(self.state1)
            inner = sqrt_rho @ self.state2 @ sqrt_rho
            sqrt_inner = sqrtm(inner)
            fidelity = np.real(np.trace(sqrt_inner))**2
            return round(fidelity, 4)
        
        # Case 3: one pure, one mixed
        elif Is(self.state1).ket() and Is(self.state2).square():
            fidelity = np.abs((Q(self.state1).d() @ self.state2 @ self.state1).item())
            # fidelity = np.sqrt(fidelity)
            return round(fidelity, 4)
        elif Is(self.state1).square() and Is(self.state2).ket():
            fidelity = np.abs((Q(self.state2).d() @ self.state1 @ self.state2).item())
            # fidelity = np.sqrt(fidelity)
            return round(fidelity, 4)
        else:
            raise ValueError("Unsupported input types for fidelity.")


def conc_pure(state):
    a, b, c, d = state.flatten()
    val = np.abs(np.real(a * d - b * c))
    return round(val, 4)  # 0 ⇒ separable, ≠0 ⇒ entangled
def conc_mixed(state):
    sy_sy = np.kron(Q.Pauli.Y, Q.Pauli.Y)
    rho_star = np.conj(state)
    rho_tilde = sy_sy @ rho_star @ sy_sy
    R = sqrtm(sqrtm(state) @ rho_tilde @ sqrtm(state))
    eigenvals = np.sort(np.real(eigvals(R)))[::-1]
    concurrence_val = max(0, eigenvals[0] - np.sum(eigenvals[1:]))
    return round(concurrence_val, 4)
    
class Ebit:
    def __new__(cls, state, method='concurrence'):
        state = np.asarray(state, dtype=complex)
        # convert whatever user typed into lowercase letters.
        method = method.lower() if isinstance(method, str) else 'concurrence'

        # distinguish ket vs density matrix
        if state.ndim == 1 or (state.ndim == 2 and state.shape[1] == 1):   # ket
            if state.size != 4:
                raise ValueError("Ket must be a 4-dim vector.")
            if method.lower() != 'concurrence':
                raise ValueError(f"Method '{method}' not available for pure state.")
            val = conc_pure(state)
            return val

        elif state.shape == (4, 4):    # density matrix
            # if not np.allclose(state, state.conj().T): # Check hermicity
            #     raise ValueError("Density matrix must be Hermitian.")
            # if not np.isclose(np.trace(state), 1, atol=1e-12): # Check normalization
            #     raise ValueError("Density matrix must have trace 1.")
            if method == 'concurrence':
                val = conc_mixed(state)
#             elif method == 'negativity':
#                 val = negativity(state)
#             elif method == 'ppt':
#                 # state is entangled if partial transpose has negative eigenvalue
#                 eigs = np.linalg.eigvals(partial_transpose(state, sys=1))
#                 val = np.min(np.real(eigs))   # negative ⇒ entangled
            else:
                raise ValueError(f"Unknown entanglement method: {method}")
            return val
        else:
            raise ValueError("Input must be 4x1 ket or 4x4 density matrix.")

# ---------------------------------------------------------------------------
# Multi-qubit utilities and density matrix validations
# ---------------------------------------------------------------------------

def partial_trace(rho: np.ndarray, n_qubits: int, qubits: list[int]) -> np.ndarray:
    out = rho
    n = n_qubits
    for q in qubits:
            if not (0 <= q < n_qubits):
                raise ValueError(f"qubit {q} out of range for n_qubits={n_qubits}")
    for q in sorted(qubits, reverse=True):
        reshaped = out.reshape([2] * (2 * n))
        traced = np.trace(reshaped, axis1=q, axis2=q + n)
        dim = 2 ** (n - 1)
        out = traced.reshape(dim, dim)
        n -= 1
    return out


def expand_with_maximally_mixed(rho_kept: np.ndarray, kept_positions: list[int], total_qubits: int) -> np.ndarray:
    missing = [q for q in range(total_qubits) if q not in kept_positions]
    n_missing = len(missing)

    identity_block = np.eye(2**n_missing, dtype=complex)
    combined = np.kron(rho_kept, identity_block) / (2**n_missing)

    order = list(kept_positions) + missing
    inverse_order = list(np.argsort(order))
    row_perm = inverse_order
    col_perm = [total_qubits + p for p in inverse_order]
    perm = row_perm + col_perm

    tensor = combined.reshape([2] * (2 * total_qubits))
    tensor = np.transpose(tensor, perm)
    dim = 2**total_qubits
    return tensor.reshape(dim, dim)