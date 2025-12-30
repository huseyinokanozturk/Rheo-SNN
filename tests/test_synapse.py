from src.neuron import LIFNeuron
import matplotlib.pyplot as plt

neuron_1 = LIFNeuron(tau=10, dt=0.1, threshold=1, rest_potential=0, reset_potential=0, refractory_time=2)
neuron_2 = LIFNeuron(tau=10, dt=0.1, threshold=1, rest_potential=0, reset_potential=0, refractory_time=2)

w = 80

neuron_1_spikes = []
neuron_2_spikes = []

neuron_1_voltage = []
neuron_2_voltage = []

for i in range(1000):

    spike1 = neuron_1.update(1.5)

    current_for_neuron_2 = w * spike1

    spike2 = neuron_2.update(current_for_neuron_2)

    neuron_1_voltage.append(neuron_1.current_voltage)
    neuron_2_voltage.append(neuron_2.current_voltage)
    neuron_1_spikes.append(spike1)
    neuron_2_spikes.append(spike2)


"""
Visualization to see the voltages and spikes.
"""

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

ax1.plot(neuron_1_voltage, label='N1 Voltage (Input)', color='blue')

spike_times_n1 = [i for i, x in enumerate(neuron_1_spikes) if x == 1]
ax1.vlines(spike_times_n1, 0, 1.2, color='orange', linestyle='dashed', label='N1 Spike')
ax1.set_title('Neuron 1: Pacemaker (Continuously Firing)')
ax1.set_ylabel('Voltage (mV)')
ax1.legend(loc='upper right')


ax2.plot(neuron_2_voltage, label='N2 Voltage (Output)', color='green')

spike_times_n2 = [i for i, x in enumerate(neuron_2_spikes) if x == 1]
ax2.vlines(spike_times_n2, 0, 1.2, color='orange', linestyle='dashed', label='N2 Spike')
ax2.set_title('Neuron 2: Stimulated by Neuron 1')
ax2.set_xlabel('Time (Step)')
ax2.set_ylabel('Voltage (mV)')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()