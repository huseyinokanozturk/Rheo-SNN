"""
Reward-Modulated STDP (R-STDP) Verification Experiment
======================================================

Objective:
----------
This script verifies the functionality of the Reward-Modulated Spike-Timing Dependent Plasticity (R-STDP) 
mechanism implemented in `src.synapse`. It demonstrates how a synapse learns not just from spike timing 
(correlation), but from the *consequences* (Reward/Punishment) of that activity.

Experiment Protocol:
--------------------
The simulation runs for 2000 time steps with a single pre-synaptic neuron connected to a post-synaptic neuron.

1. **Stimulation Pattern**:
   - The Pre-neuron fires periodically every 200 steps.
   - A "Teacher Signal" forces the Post-neuron to fire shortly after the Pre-neuron (forcing a causal pre->post relationship).
   - Under standard STDP, this would always lead to Long-Term Potentiation (LTP) and weight growth.

2. **Reward Modulation Phases**:
   - **Phase 1 (Steps 0-1000) [Conditioning]**:
     - A **Positive Reward (+10)** is delivered 5ms after the spike event.
     - **Expected Result**: The synapse should *strengthen* (Weight increases), reinforcing the behavior.
   
   - **Phase 2 (Steps 1000-2000) [Extinction/Punishment]**:
     - The same spike timing is maintained, but now a **Negative Reward (-10)** is delivered.
     - **Expected Result**: The synapse should *weaken* (Weight decreases), unlearning the behavior despite the correlated spiking.

3. **Mechanism**:
   - The synapse uses an **eligibility trace** to remember the spike coincidence until the reward arrives.
   - Weight Update = Learning Rate * Eligibility Trace * Reward.

Visualization:
--------------
The script generates a 3-panel plot showing:
1. **Weight Evolution**: Demonstrating the rise (Phase 1) and fall (Phase 2) of synaptic strength.
2. **Reward Signal**: The switch from positive to negative reinforcement.
3. **Spike Raster**: Confirming the consistent firing patterns of pre- and post-neurons.
"""
from src.synapse import Synapse
from src.neuron import LIFNeuron
import matplotlib.pyplot as plt



neuron_pre =  LIFNeuron(tau=10, dt=0.1, threshold=1, rest_potential=0, reset_potential=0, refractory_time=2)
neuron_post =  LIFNeuron(tau=10, dt=0.1, threshold=1, rest_potential=0, reset_potential=0, refractory_time=2)

synapse = Synapse(weight=20, pre_neuron=neuron_pre, post_neuron=neuron_post, w_max=150, lr=0.5,
tau_eligibility=50, dt=0.1)

weight_history = []
reward_history = []
spike_history_pre = []
spike_history_post = []


for i in range(2000):
    
    # First every 200 steps the pre-synaptic neuron emits a signal and spikes
    current_pre = 100.0 if i % 200 == 0 else 0.0 
    spike_pre = neuron_pre.update(current_pre)

    # Transmit and teacher signal
    input_from_synapse = synapse.weight if spike_pre else 0.0

    teacher_signal = 0.0
    if i% 200 == 5:
        teacher_signal = 100.0

    spike_post = neuron_post.update(input_from_synapse + teacher_signal)

    # Reward signal: After 5ms give a reward or punishment
    reward = 0.0 

    # First half (0-1000): REWARD (+10)
    if i < 1000:
        if i % 200 == 55: #after 50 steps (5ms)
            reward = 10.0
    # Second half (1000-2000): PUNISHMENT (-10)
    else:
        if i % 200 == 55: #after 50 steps (5ms)
            reward = -10.0

    # Update the synapse
    synapse.step(spike_pre, spike_post, reward)

    # Append the weight and spikes to the history for visualization
    weight_history.append(synapse.weight)
    reward_history.append(reward)
    spike_history_pre.append(1 if spike_pre else 0)
    spike_history_post.append(1 if spike_post else 0)


"""
Visualization to see the weights and spikes.
"""

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10, 10))

ax1.plot(weight_history, label='Synaptic Weight', color='blue', linewidth=2)
ax1.set_title('Synaptic Weight(+10, -10)')
ax1.set_ylabel('Weight')
ax1.axvline(x=1000, color='black', linestyle='--', label='Mode Change')
ax1.legend()

ax2.plot(reward_history, label='Reward Signal', color='purple')
ax2.set_title('Reward Signal (+ / -)')
ax2.set_ylabel('Reward')
ax2.legend()

ax3.plot(spike_history_pre, label='Pre', color='green')
ax3.plot(spike_history_post, label='Post', color='red', alpha=0.7)
ax3.set_title('Neuron Spikes')
ax3.set_xlabel('Time (Step)')
ax3.legend()

plt.tight_layout()
plt.show()    




