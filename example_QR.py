from qrnetwork import SimulationConfig, RepeaterSimulation

# -----------------------------------------------------------
# Discrete-Event Quantum Repeater Simulation
# -----------------------------------------------------------
config = SimulationConfig(
    rep_nodes=3,                   # 3 repeaters + 2end nodes -> total qubits = 3*2 + 2 = 8
    rep_spacing_km=30.0,

    pairgen_prob_perpulse=0.01,   # P(n)=mu^n/(1+mu)^(n+1), where n is number of pairs and
                                  # mu represents the mean number of photon pairs generated per pump pulse
    prep_time_s= 1e-8,            # preparation time for source event  
                                  # depends on clock rate(At what rate source can make a new attempt)
    source_state_fidelity=0.99,

    dark_count_rate_hz=100,
    detection_window_s=1e-9,      # time for which detector is on,   # more than jitter time,   # least possible available: 20ps*2-3 
    dephasing_time_s=0.01,

    bsm_fidelity=0.99,
    )

sim = RepeaterSimulation(config)
result = sim.run()

print(f"Total Simulation Time: {result.total_sim_time:.6f} s")
print(f"Entanglement Rate:     {result.entanglement_rate:.2f} pairs/sec")
print(f"Average Fidelity:      {result.average_fidelity:.4f}")

for pair in result.completed_pairs:
    print(f" - Pair ID #{pair.id}: Fidelity = {pair.fidelity:.4f}, Generated at t = {pair.created_time:.6f} s")

# #-----------------------------------------------
# # Finding average fidelity over multiple runs
# #-----------------------------------------------
# avgfid=[]
# runs=10
# for i in range(runs):
#     sim = RepeaterSimulation(config)
#     result = sim.run()
#     avgfid.append(result.average_fidelity)
# print(f"Average fidelity over {runs} runs: {sum(avgfid)/len(avgfid)}")