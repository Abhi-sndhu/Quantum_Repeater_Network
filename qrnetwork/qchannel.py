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
        if primaryfn.MatrixDim(self.channel).is_ket():
            channel = primaryfn.QStateAnalyzer(self.channel)
            entangled = primaryfn.QEntangle2(self.channel)
            if entangled >= self.tol:
                for i in range(self.iteration):
                    state = primaryfn.QStateGenerator(1).pure()
                    output = qteleportation.teleport_ket_state(state, channel)
                    fidelity = [r["fidelity"] for r in output]
                    fidelities.append(fidelity)
                return {"Average fidelity": round(np.mean(fidelities), 2)}
            else:
                raise ValueError("Your quantum channel is not entangled.")
        elif primaryfn.MatrixDim(self.channel).is_square():
            channel = primaryfn.QStateAnalyzer(self.channel)
            entangled = primaryfn.QEntangle2(channel)
            if np.abs(entangled) > self.tol:  # np.isclose(entangled, 1.0)
                for i in range(self.iteration):
                    state = primaryfn.QStateGenerator(1).mixed()
                    output = qteleportation.teleport_density_state(state, channel)
                    fidelity = [r["fidelity"] for r in output]
                    fidelities.append(fidelity)
                return {"Average fidelity": round(np.mean(fidelities), 2)}
            else:
                raise ValueError("Your quantum channel is not entangled.")
        else:
            raise ValueError("Incorrect dimensions of the quantum channel.")