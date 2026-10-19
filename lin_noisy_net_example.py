from qrnetwork import SimulationConfig, RepeaterSimulation

# -----------------------------------------------------------
# Discrete-Event Quantum Repeater Simulation
# -----------------------------------------------------------
# Configuration
config = SimulationConfig(
    # Nodes
    rep_nodes=3,                   # 3 repeaters + 2end nodes -> total qubits = 3*2 + 2 = 8
    rep_spacing_km=30.0,
    # Source
    source_efficiency=1e-8,
    source_state_fidelity=0.99,
    # Laser
    laser_power_avg_mW=0.04,
    laser_rep_rate_MHz=100,
    # Detector
    detector_rep_rate_MHz=25,     # laser rate / N for synchronization
    dark_count_rate_hz=100,
    detection_window_s=8e-9,      # time for which detector is on,   # takes care of uncertaintity in arrival time
    # Memory
    dephasing_param_s=0.01,
    # Imperfect BSM
    bsm_fidelity=0.99,
    # Entanglement Swapping Policy
    swap_policy="nesting",
    # Seed for reproducibility
    seed=102,
    )
# Initialization
setup = RepeaterSimulation(config)
# Running the simulation
result = setup.run()

print(f"Total Simulation Time: {result.total_sim_time:.6f} s")
print(f"Entanglement Rate:     {result.entanglement_rate:.2f} pairs/sec")
print(f"Average Fidelity:      {result.average_fidelity:.4f}")

for link in result.completed_links:
    print(f"Fidelity = {link.fidelity:.4f}, Generated at t = {link.created_time:.6f} s")

# #-----------------------------------------------
# # Finding average fidelity over multiple runs
# #-----------------------------------------------
# avgfid=[]
# runs=10
# config_no_seed = SimulationConfig()    #By default there is no seed   #Need to define config once: Frozen
# for i in range(runs):
#     setup = RepeaterSimulation(config_no_seed)    #Resetting the setup again and again is required  #Cannot reuse setup after a run!!!!   
#                                                   #setup has chain -> memories -> links, that change as simulation runs
#     result = setup.run()
#     avgfid.append(result.average_fidelity)
#     print(f"{result.average_fidelity:.4f}")
# print(f"Average fidelity over {runs} runs: {sum(avgfid)/len(avgfid)}")