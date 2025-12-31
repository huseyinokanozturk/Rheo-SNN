import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from src.network import Network  
def run_test():
    print("🧪 RHEO Network: Detailed System Test Initiated...")

    # 1. NETWORK SETUP
    # 50 Neurons: 10 Input neurons, 40 Hidden/Output neurons
    net = Network(num_neurons=50, num_inputs=10, num_outputs=5, dt=0.5)

    # Exaggerate parameters to visualize effects clearly during testing
    net.energy_cost = 5.0        # High cost to induce fatigue quickly
    net.recovery_rate = 0.5      # Moderate recovery rate
    net.fatigue_factor = 0.5     # Significant impact of fatigue on thresholds

    # 2. SIMULATION LOOP
    steps = 1000
    
    # Data Logging
    rec_spikes = []          # Raster Plot data (Time, Neuron ID)
    rec_voltage_n20 = []     # Voltage trace for Neuron 20
    rec_threshold_n20 = []   # Adaptive Threshold trace for Neuron 20
    rec_energy_n20 = []      # Energy Level trace for Neuron 20
    rec_dopamine = []        # Environmental Dopamine Level

    print(f"⏳ Simulation running for {steps} steps...")

    for t in range(steps):
        # A. Generate Input (Only for the first 10 neurons)
        # Random current between 2 and 8
        inputs = np.random.uniform(2, 8, size=10)
        
        # B. Scenario: DOPAMINE SHOWER (Between steps 400 and 600)
        reward_signal = 0.0
        if 400 <= t < 600:
            reward_signal = 1.0  # High Dopamine! (Excitement phase)
        
        # C. Run Simulation Step
        # 'inputs' has only 10 elements; 'step' method handles mapping to input neurons
        spike_vector = net.step(external_inputs=inputs, reward=reward_signal)

        # D. Data Recording
        # 1. Identify firing neurons for Raster Plot
        fired_indices = np.where(spike_vector)[0]
        for idx in fired_indices:
            rec_spikes.append((t, idx))
            
        # 2. Inspect a single neuron (e.g., Neuron 20)
        # Choosing an internal neuron (not one of the input neurons)
        rec_voltage_n20.append(net.voltages[20])
        rec_threshold_n20.append(net.thresholds[20])
        rec_energy_n20.append(net.energies[20])
        rec_dopamine.append(net.dopamine)

    print("✅ Simulation Completed. Plotting results...")

    # 3. VISUALIZATION
    plt.figure(figsize=(12, 10))

    # Plot 1: Raster Plot (Network Activity)
    plt.subplot(3, 1, 1)
    if len(rec_spikes) > 0:
        times, neurons = zip(*rec_spikes)
        plt.scatter(times, neurons, s=2, c='black', alpha=0.6)
    plt.title('Network Activity (Raster Plot)')
    plt.ylabel('Neuron ID')
    plt.axvline(x=400, color='green', linestyle='--', label='Dopamine Start')
    plt.axvline(x=600, color='red', linestyle='--', label='Dopamine End')
    plt.legend(loc='upper right')

    # Plot 2: Selected Neuron Dynamics (Voltage & Threshold)
    plt.subplot(3, 1, 2)
    plt.plot(rec_voltage_n20, label='Voltage (V)', color='blue', alpha=0.5)
    plt.plot(rec_threshold_n20, label='Adaptive Threshold (Th)', color='red', linestyle='--')
    plt.title('Single Neuron Dynamics (Neuron #20)')
    plt.ylabel('mV')
    plt.legend()

    # Plot 3: Energy vs Dopamine
    plt.subplot(3, 1, 3)
    plt.plot(rec_energy_n20, label='Energy (ATP)', color='green')
    plt.plot(np.array(rec_dopamine)*10 + 50, label='Dopamine Signal (Scaled)', color='orange', alpha=0.7)
    plt.title('Metabolism & Neuromodulation')
    plt.xlabel('Time (Steps)')
    plt.ylabel('Level')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_test()