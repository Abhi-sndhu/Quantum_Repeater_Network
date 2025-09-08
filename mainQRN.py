import numpy as np
from qrnetwork import *
# import time
# from qrnetwork import Pauli, QStateGenerator, Bell, Bells, QTeleportation, QChannel, eswap, SWAPN

# # Example: generate a random 2-qubit pure state
# qS = QStateGenerator(1).pure()
# qS = QStateGenerator(1).mixed()
# print(Qstate(qS).norm())
# print(Qprod.inner(qS))
# print(qS)

## Print Pauli operations
# X = Pauli.Y
# print(X)

## Bell state
phi_plus  = Bell.phi_plus
phi_minus = Bell.phi_minus
psi_plus  = Bell.psi_plus
psi_minus = Bell.psi_minus
# bell_states= [Bell.phi_plus, Bell.phi_minus, Bell.psi_plus, Bell.psi_minus]
# print(phi_plus)

#qC = Bell.phi_plus
# print(Bell.phi_plus.shape)
# print(QEntangle2(qC))
# qC = Belld.pure(phi_plus) # Gives density matrix state
# qC = Belld.impure(phi_plus)
# print(qC)
# print(QFidelity(Bell.phi_plus,Bell.phi_minus).item())
## Check two-qubit entanglement criteria
# print(QEntangle2(phi_plus))
# print(QEntangle2(qC))

# Run teleportation
# results = QTeleportation(qS, qC)
# # Check channel efficiency
# results = QChannel(qC, iteration=10).teleport()
## Get output of linear quantum repeater network without noise
# shared_ent_state = [Qprod.outer(psi_plus),Qprod.outer(phi_minus),
#                    Qprod.outer(phi_plus),Qprod.outer(psi_plus), 
#                    Qprod.outer(phi_minus)]
shared_ent_state = [psi_plus,psi_minus,phi_plus,phi_minus,psi_plus]
results = SWAPN(shared_ent_state).linear()
## Get output of linear quantum repeater network with noise
# start_time = time.time()
# results = SWAPN(shared_ent_state).noise(L=40, T_p= 1e-6, T_dp = 1, eta = 0.3, p_d = 1e-8).linear()
# results = SWAPN(shared_ent_state).noise(L=40, T_p= 1e-6, T_dp = 1, eta = 0.3, p_d = 1e-8, loss=0.5).linear()
# end_time = time.time() 
# elapsed_time = end_time - start_time
# print(f"Results: {results}\n Execution time: {elapsed_time} seconds")
## Print output
print(results)
