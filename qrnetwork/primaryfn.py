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

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs): 
        # Added because matrix multiplication was giving numpy array when first operand is numpy array and second is Q object. Now it will return Q object.
        # However, it was working fine when first operand is Q object and second is numpy array.
        if method != "__call__":
            return NotImplemented
        # Unwrap any Q operands to raw ndarrays
        raw_inputs = [x.state if isinstance(x, Q) else x for x in inputs]
        out = kwargs.get("out", None)
        if out is not None:
            kwargs["out"] = tuple(
                o.state if isinstance(o, Q) else o for o in out
            )
        result = getattr(ufunc, method)(*raw_inputs, **kwargs)
        if isinstance(result, np.ndarray):
            return Q(result)
        if isinstance(result, tuple):
            return tuple(Q(r) if isinstance(r, np.ndarray) else r for r in result)
        return result

    def __getitem__(self, key):
        result = self.state[key]
        # Return a Q if it's still array-like, plain scalar otherwise
        if isinstance(result, np.ndarray):
            return Q(result)
        return result

    def __setitem__(self, key, value):
        if isinstance(value, Q):
            value = value.state
        self.state[key] = value

    def __len__(self):
        return len(self.state)

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
    def __radd__(self, other):  # Right-side addition, needed for sum()
        if other == 0:          # sum() starts with the int 0
            return self
        return Q(other + self.state)
        
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

    def __matmul__(self, other):
        if isinstance(other, Q):
            other = other.state
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

    def tr(self):
        """Compute the matrix trace Tr(A) of the quantum object.

        Returns a complex scalar (or float if purely real).
        """
        r, c = self.state.shape
        if r != c:
            raise ValueError("Trace is only defined for square operators.")
        trace_val = np.trace(self.state)
        # Return clean float if imaginary part is negligible
        return (
            float(np.real(trace_val))
            if np.isclose(np.imag(trace_val), 0, atol=1e-10)
            else complex(trace_val)
        )
    def ptr(self, qubits: list[int]):  # partial trace
        """Return density matrix after qubits traced out"""
        state = self.state
        n_qubits = int( np.log2(self.state.shape[0]) )
        n = n_qubits
        for q in qubits:
                if not (0 <= q < n_qubits):
                    raise ValueError(f"qubit {q} out of range for n_qubits={n_qubits}")
        for q in sorted(qubits, reverse=True):
            reshaped = state.reshape([2] * (2 * n))
            traced = np.trace(reshaped, axis1=q, axis2=q + n)
            dim = 2 ** (n - 1)
            state = traced.reshape(dim, dim)
            n -= 1
        return Q(state)
    
    def expand_mixed(self, kept_positions: list[int], total_qubits: int): # expand with maximally mixed
        """Return density matrix expanded with maximally mixed state."""
        rho_kept = self.state
        missing = [q for q in range(total_qubits) if q not in kept_positions]
        n_missing = len(missing)

        identity_block = np.eye(2**n_missing, dtype=complex)
        combined = np.kron(rho_kept, identity_block) / (2**n_missing)   # [0,3,1,2,4,7,5,6] -> [0,1,2,3,4,5,6,7]]

        order = list(kept_positions) + missing                 # [0,3,1,2]
        inverse_order = list(np.argsort(order))                # [0,2,3,1]
        row_perm = inverse_order                               # [0,2,3,1]
        col_perm = [total_qubits + p for p in inverse_order]   # [4,6,7,5]
        perm = row_perm + col_perm                             # [0,2,3,1,4,6,7,5]

        tensor = combined.reshape([2] * (2 * total_qubits))    # [0,3,1,2,4,7,5,6] (change according to [0,2,3,1,4,6,7,5])
        tensor = np.transpose(tensor, perm)                    # [0,1,2,3,4,5,6,7]
        dim = 2**total_qubits
        return Q(tensor.reshape(dim, dim))   
    
    def copy(self):
            """Returns an independent deep copy of the quantum object."""
            return Q(self.state.copy())
    
    # ----- Basis State Generator -----
    @staticmethod
    def basis(dim, index=0):
        """Generates computational basis ket |index> of dimension dim.
    
            Usage: Q.basis(2, 0) for |0>, Q.basis(2, 1) for |1>
            """
        if not (0 <= index < dim):
            raise ValueError(
                f"Index {index} out of range for dimension {dim}."
            )
        vec = np.zeros((dim, 1), dtype=complex)
        vec[index, 0] = 1.0
        return Q(vec)

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
    # ============================================================
    # General controlled single-qubit gate
    # ============================================================
    class Cops:
        @staticmethod
        def Cg(gate, n_qubits, control, target):
            if control == target:
                raise ValueError(
                    "Control and target qubits must be different."
                )

            if not (0 <= control < n_qubits):
                raise ValueError("Control qubit index out of range.")

            if not (0 <= target < n_qubits):
                raise ValueError("Target qubit index out of range.")

            gate_arr = gate.state if isinstance(gate, Q) else gate
            dim = 2**n_qubits
            result = np.zeros((dim, dim), dtype=complex)

            for basis_idx in range(dim):
                bits = [
                    (basis_idx >> (n_qubits - 1 - i)) & 1
                    for i in range(n_qubits)
                ]
                if bits[control] == 0:
                    result[basis_idx, basis_idx] = 1
                else:
                    for t in range(2):
                        new_bits = bits.copy()
                        new_bits[target] = t
                        new_basis = 0
                        for b in new_bits:
                            new_basis = (new_basis << 1) | b
                        result[new_basis, basis_idx] = gate_arr[
                            t, bits[target]
                        ]

            return Q(result)

        @staticmethod
        def Cx(n_qubits, control, target):
            return Q.Cops.Cg(Q.Pauli.X, n_qubits, control, target)

        @staticmethod
        def Cy(n_qubits, control, target):
            return Q.Cops.Cg(Q.Pauli.Y, n_qubits, control, target)

        @staticmethod
        def Cz(n_qubits, control, target):
            return Q.Cops.Cg(Q.Pauli.Z, n_qubits, control, target)

    # ============================================================
    # General SWAP gate
    # ============================================================
    @staticmethod
    def SWAP(n_qubits, qubit1, qubit2):
        if qubit1 == qubit2:
            raise ValueError("The two qubits must be different.")

        if not (0 <= qubit1 < n_qubits):
            raise ValueError("qubit1 index out of range.")

        if not (0 <= qubit2 < n_qubits):
            raise ValueError("qubit2 index out of range.")

        dim = 2**n_qubits
        swap = np.zeros((dim, dim), dtype=complex)

        for basis_idx in range(dim):
            bits = [
                (basis_idx >> (n_qubits - 1 - i)) & 1 for i in range(n_qubits)
            ]
            swapped_bits = bits.copy()
            swapped_bits[qubit1], swapped_bits[qubit2] = (
                swapped_bits[qubit2],
                swapped_bits[qubit1],
            )
            new_basis = 0
            for bit in swapped_bits:
                new_basis = (new_basis << 1) | bit
            swap[new_basis, basis_idx] = 1

        return Q(swap) 

    
# ----- Bell states -----
class Bell:
    phi_plus = Q(np.array([[1], [0], [0], [1]], dtype=complex) / np.sqrt(2))
    phi_minus = Q(np.array([[1], [0], [0], [-1]], dtype=complex) / np.sqrt(2))
    psi_plus = Q(np.array([[0], [1], [1], [0]], dtype=complex) / np.sqrt(2))
    psi_minus = Q(np.array([[0], [1], [-1], [0]], dtype=complex) / np.sqrt(2))

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
Q.Bell = Bell  # Attach Bell class to Q for easy access

class Werner:
    def __new__(cls, p=None, state=None):
        if state is None:
            state = Q.Bell.phi_plus
        bell_pure = state @ state.d()
        I4 = np.eye(4) / 4
        if p is None:
            p = np.random.uniform(low=1/2, high=1.0)
        # If p is provided, use it directly
        return Q(p * bell_pure + (1 - p) * I4)
Q.Werner = Werner  # Attach Werner class to Q for easy access

class Pauli:
    I = Q(np.eye(2, dtype=complex))
    X = Q(np.array([[0, 1], [1, 0]], dtype=complex))
    Y = Q(np.array([[0, -1j], [1j, 0]], dtype=complex))
    Z = Q(np.array([[1, 0], [0, -1]], dtype=complex))
    @classmethod
    def all(cls):
        return {
            'PauliI': cls.I,
            'PauliX': cls.X,
            'PauliY': cls.Y,
            'PauliZ': cls.Z
        }
Q.Pauli = Pauli  # Attach Pauli class to Q for easy access  

class Gate:
        H = Q((1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex))
        S = Q(np.array([[1, 0], [0, 1j]], dtype=complex))
        T = Q(np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex))
        #CNOT gate
        CX = Q(np.kron((Pauli.I + Pauli.Z) / 2 , Pauli.I) + np.kron((Pauli.I - Pauli.Z) / 2 , Pauli.X))
Q.Gate = Gate



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
    def __init__(self, *states, tol=1e-10):
        # Handle the case where a single list or tuple of states is passed
        if len(states) == 1 and isinstance(states[0], (list, tuple)):
            states = states[0]

        if len(states) == 0:
            raise ValueError("Qp requires at least one state.")

        # Ensure all inputs are initialized as Q objects (converting if necessary)
        self.q_objs = [
            item if isinstance(item, Q) else Q(item) for item in states
        ]
        self.tol = tol

        # Backward compatibility for .op() and other legacy functions:
        # Extract the underlying data arrays from the first two Q objects
        self.state1 = self.q_objs[0].state
        self.state2 = (
            self.q_objs[1].state if len(self.q_objs) > 1 else self.state1
        )
    
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

    def tp(self): 
        """Computes the tensor product of all states stored in this Qp instance."""
        combined_matrix = fold(np.kron, [q.state.copy() for q in self.q_objs])
        return Q(combined_matrix)

    def fid(self): # Fidelity
        # Case 1: both are pure states (kets)
        if Is(self.state1).ket() and Is(self.state2).ket():
            value = self.ip()  # <psi|phi>
            fidelity = np.abs(value).item()
            return round(fidelity.real, 4)

        # Case 2: both are density matrices
        elif Is(self.state1).square() and Is(self.state2).square():
            sqrt_rho = sqrtm(self.state1)
            inner = sqrt_rho @ self.state2 @ sqrt_rho
            sqrt_inner = sqrtm(inner)
            fidelity = np.real(np.trace(sqrt_inner))
            return round(float(fidelity.real), 4)
        
        # Case 3: one pure, one mixed
        elif Is(self.state1).ket() and Is(self.state2).square():
            fidelity = np.abs((Q(self.state1).d() @ self.state2 @ self.state1).item())
            fidelity = np.sqrt(fidelity)
            return round(fidelity.real, 4)
        elif Is(self.state1).square() and Is(self.state2).ket():
            fidelity = np.abs((Q(self.state2).d() @ self.state1 @ self.state2).item())
            fidelity = np.sqrt(fidelity)
            return round(fidelity.real, 4)
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

#fold function for the repetetive operation application

def fold(operation, objects):
    """operation : callable (f(acc, item) -> acc)
    objects   : list or tuple of elements to reduce
    """
    result = objects[0]
    for obj in objects[1:]:
        result = operation(result, obj)
    return result