from qrnetwork import SimulationConfig, RepeaterSimulation, heralding_statistics
import numpy as np

# -----------------------------------------------------------
# Discrete-Event Quantum Repeater Chain Simulation
# -----------------------------------------------------------
# config = SimulationConfig()
config = SimulationConfig(
    n_nodes=5,                    # 2 end nodes + 2 repeaters
    memories_per_link=1,          # 1 memory per link
    link_length_km=22.0,

    source_efficiency=0.01,
    source_state_fidelity=0.99,

    dark_count_rate_hz=100,
    detection_window_s=1e-10,
    dephasing_time_s=0.001,       # Change dephasing_time_s

    trial_period_s=1e-9,
    bsm_fidelity=0.99,
    n_target_pairs=1,             # Generate 1 end-to-end pair
    seed=None,
    )

sim = RepeaterSimulation(config)
result = sim.run()

print(f"Total Simulation Time: {result.total_sim_time:.6f} s")
print(f"Entanglement Rate:     {result.entanglement_rate:.2f} pairs/sec")
print(f"Average Fidelity:      {result.average_fidelity:.4f}")

for pair in result.completed_pairs:
    print(f" - Pair ID #{pair.id}: Fidelity = {pair.fidelity:.4f}, Generated at t = {pair.created_time:.6f} s")

#For average fidelity over multiple runs
avgfid=[]
runs=10
for i in range(runs):
    sim = RepeaterSimulation(config)
    result = sim.run()
    avgfid.append(result.average_fidelity)
print(f"Average fidelity over {runs} runs: {sum(avgfid)/len(avgfid)}")