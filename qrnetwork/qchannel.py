from . import qteleportation
from . import primaryfn
import numpy as np

class QChannel:
    def __init__(self, channel, iteration, tol=1e-10):
        self.tol = tol
        self.channel = channel
        self.iteration = iteration
    def teleport(self):
        row, col = self.channel.shape
        fidelities = []
        if primaryfn.Is(self.channel).ket():
            channel = primaryfn.Q(self.channel).u()
            entangled = primaryfn.Ebit(self.channel)
            if entangled >= self.tol:
                for i in range(self.iteration):
                    state = primaryfn.Q.rand.p(1)
                    output = qteleportation.teleport_ket_state(state, channel)
                    fidelity = [r["fidelity"] for r in output]
                    fidelities.append(fidelity)
                return {"Average fidelity": round(np.mean(fidelities), 2)}
            else:
                raise ValueError("Your quantum channel is not entangled.")
        elif primaryfn.Is(self.channel).square():
            channel = primaryfn.Q(self.channel).u()
            entangled = primaryfn.Ebit(channel)
            if np.abs(entangled) > self.tol: 
                for i in range(self.iteration):
                    state = primaryfn.Q.rand.m(1)
                    output = qteleportation.teleport_density_state(state, channel)
                    fidelity = [r["fidelity"] for r in output]
                    fidelities.append(fidelity)
                return {"Average fidelity": round(np.mean(fidelities), 2)}
            else:
                raise ValueError("Your quantum channel is not entangled.")
        else:
            raise ValueError("Incorrect dimensions of the quantum channel.")