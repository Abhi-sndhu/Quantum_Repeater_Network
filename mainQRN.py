import numpy as np
from qrnetwork import *
# import time
# from qrnetwork import Q, Qp, Is, Ebit, Teleport, Qrep

## Define quantum states
# ket0 = Q([[1], [0]])
# print(ket0)
# ket1 = Q([[0], [1]])
# print(ket1)

## Density matrix
# qC = Q(Q.Bell.phi_plus).dm()
# qC = Q([[1], [0], [0], [1]]).dm()
# print(qC)

## Dagger
# ket0 = Q([[1], [0]])
# print(ket0.d())
# print(ket0.dm().d())

## Norm
# ket0 = Q([[1], [1]])
# print(ket0.n())
# print(ket0.dm().n())

## Unit
# ket0 = Q([[1], [1]])
# print(ket0.u())
# print(ket0.dm().u())

## Purity
# ket0 = Q([[1], [1]])
# print(ket0.purity())
# print(ket0.dm().purity())

## Random State
# print(Q.rand.p(2))
# print(Q.rand.m(2))

## Bell state
# phi_plus  = Q.Bell.phi_plus
# phi_minus = Q([[1], [0], [0], [-1]]) / np.sqrt(2) 
# psi_plus  = Q.Bell.psi_plus
# psi_minus = Q.Bell.psi_minus
# Q.Bell.all()
# bell_states= [Q.Bell.phi_plus, Q.Bell.phi_minus, Q.Bell.psi_plus, Q.Bell.psi_minus]
# print(bell_states)
# print(phi_minus)

## Print Pauli operations
# X = Q.Pauli.X
# X = Q([[0, 1], [1, 0]])
# print(X)

## Werner State
# qC = Q.Werner(Q.Bell.phi_minus)
# qC = Q.Werner(p=0.8, state=Q.Bell.phi_plus)
# print(qC.u().n())

# qS = np.cos(np.pi / 8) * ket0 + np.sin(np.pi / 8) * ket1

## Quantum Products
# print(Qp(Q.Bell.phi_plus,Q.Bell.phi_plus).ip())
# print(Qp(Q([[1],[0]])).op())
# print(Qp(Q([[1],[0]])).tp())
# print(Qp(Q.Bell.phi_plus,Q.Bell.phi_plus).fid())
# print(Qp(Q(Q.Bell.phi_plus).dm(),Q(Q.Bell.phi_plus).dm()).fid())
# print(Qp(Q.Bell.phi_plus,Q(Q.Bell.phi_plus).dm()).fid())
# print(Qp(Q(Q.Bell.phi_plus).dm(),Q.Bell.phi_plus).fid())



# qC = Q.Bell.phi_plus
# qC = np.sqrt(0.8)*Q.Bell.psi_plus + np.sqrt(0.2)*Q.Bell.psi_minus
# print(Q.Bell.phi_plus.shape)

## Entanglement measure
# print(Ebit(Q.Bell.phi_plus))
# print(Ebit(Q.Werner(p=0.8, state=Q.Bell.phi_plus)))



## Check two-qubit entanglement criteria
# print(Ebit(Q.Bell.phi_plus))
# print(Ebit(qC))

## Run teleportation
# tp = Teleport(Q.rand.p(1), Q.Bell.phi_plus).run()
# tp = Teleport(Q.rand.m(1), Q.Werner(p=0.8, state=Q.Bell.phi_plus)).run()
# print(tp)

# # Check channel efficiency
# results = QChannel(Q.Bell.phi_minus, iteration=10).teleport()
# print(results)

## Get output of linear quantum repeater network without noise
# shared_ent_state = [Qp(Q.Bell.psi_plus).op(),Qp(Q.Bell.phi_minus).op(),
#                     Qp(Q.Bell.phi_plus).op(),Qp(Q.Bell.psi_plus).op(), 
#                     Qp(Q.Bell.phi_minus).op()]
# shared_ent_state = [Q.Bell.psi_plus,Q.Bell.psi_minus,Q.Bell.phi_plus,Q.Bell.phi_minus,Q.Bell.psi_plus]
# results = QRep(shared_ent_state).linear()
## Get output of linear quantum repeater network with noise
# start_time = time.time()
# results = QRep(shared_ent_state).noise(L=40, T_p= 1e-6, T_dp = 1, eta = 0.3, p_d = 1e-8).linear()
# results = QRep(shared_ent_state).noise(L=40, T_p= 1e-6, T_dp = 1, eta = 0.3, p_d = 1e-8, loss=0.8).linear()
# end_time = time.time() 
# elapsed_time = end_time - start_time
# print(f"Results: {results}\n Execution time: {elapsed_time} seconds")

## Print output
# print(results)
