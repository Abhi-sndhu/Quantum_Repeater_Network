from .primaryfn import Q, Qp, Is, Ebit, partial_trace
from .qteleportation import Teleport
from .qchannel import QChannel
from .network import eswap, QRep, RepeaterChain, MemorySlot, EntangledPair, EntanglementSource, EntanglementSwapper, SlotStatus, heralding_statistics
from .sim_linear import SimulationConfig, RepeaterSimulation, SimulationResult, CompletedPair

# Define dagger for conjugate transpose (Hermitian adjoint) of an array or matrix
# def dag(matrix):
#     return np.conj(matrix).T
# setattr(np.ndarray, "dag", lambda self: np.conj(self).T)
# Monkey-patch numpy arrays to use .dag like in QuTiP
# If you want .dag() syntax on arrays → keep the setattr trick.
# If you’re okay with dag(A) function calls → you don’t need it.






