import numpy as np
from qrnetwork import QStateGenerator, Bell, QTeleportation
# Example: generate a random 2-qubit pure state
qS = QStateGenerator(1).mixed()

# Example: choose a quantum channel (impure Bell state)
phi_plus  = np.array([[1.0], [0.0], [0.0], [1.0]], dtype=complex) / np.sqrt(2)
qC = phi_plus
qC = Bell.impure(qC)

# Run teleportation
results = QTeleportation(qS, qC)
print(results)
