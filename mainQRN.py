import numpy as np
from qrnetwork import *
# import time
# from qrnetwork import Q, Qp, Is, Ebit, Teleport, Qrep

## Example: generate a random 2-qubit pure state
# qS = Q.rand.p(1)
# qS = Q.rand.m(1)
# print(Q(qS).n())
# print(Qp(qS).ip())
# print(qS)
# psi = Q([1, 0])

## Print Pauli operations
# X = Q.Pauli.X
# print(X)

## Bell state
# phi_plus  = Q.Bell.phi_plus
# phi_minus = Q.Bell.phi_minus
# psi_plus  = Q.Bell.psi_plus
# psi_minus = Q.Bell.psi_minus
# Q.Bell.all()
# bell_states= [Q.Bell.phi_plus, Q.Bell.phi_minus, Q.Bell.psi_plus, Q.Bell.psi_minus]
# print(bell_states)

# qC = Q.Bell.phi_plus
# print(Q.Bell.phi_plus.shape)
# qC = Q(Q.Bell.phi_plus).dm() # Gives density matrix state
# qC = Q.Werner(Q.Bell.phi_minus)
# print(Ebit(qC))
# print(qC)
# print(Qp(Q.Bell.phi_plus,Q.Bell.phi_plus).fid())
# print(Qp(Q(Q.Bell.phi_plus).dm(),Q(Q.Bell.phi_plus).dm()).fid())
# print(Qp(Q.Bell.phi_plus,Q(Q.Bell.phi_plus).dm()).fid())
# print(Qp(Q(Q.Bell.phi_plus).dm(),Q.Bell.phi_plus).fid())

## Check two-qubit entanglement criteria
# print(Ebit(Q.Bell.phi_plus))
# print(Ebit(qC))

## Run teleportation
# results = Teleport(qS, qC)
# # Check channel efficiency
# results = QChannel(qC, iteration=10).teleport()
## Get output of linear quantum repeater network without noise
shared_ent_state = [Qp(Q.Bell.psi_plus).op(),Qp(Q.Bell.phi_minus).op(),
                   Qp(Q.Bell.phi_plus).op(),Qp(Q.Bell.psi_plus).op(), 
                   Qp(Q.Bell.phi_minus).op()]
# shared_ent_state = [Q.Bell.psi_plus,Q.Bell.psi_minus,Q.Bell.phi_plus,Q.Bell.phi_minus,Q.Bell.psi_plus]
# results = QRep(shared_ent_state).linear()
## Get output of linear quantum repeater network with noise
# start_time = time.time()
# results = QRep(shared_ent_state).noise(L=40, T_p= 1e-6, T_dp = 1, eta = 0.3, p_d = 1e-8).linear()
results = QRep(shared_ent_state).noise(L=40, T_p= 1e-6, T_dp = 1, eta = 0.3, p_d = 1e-8, loss=0.8).linear()
# end_time = time.time() 
# elapsed_time = end_time - start_time
# print(f"Results: {results}\n Execution time: {elapsed_time} seconds")

## Print output
print(results)
