from src.neuron import LIFNeuron
import matplotlib.pyplot as plt

"""
Create a LIF Neuron and simulate it for 1000 ms with a constant input current of 1.5 mV.
"""

voltages = []
spikes = []

neuron = LIFNeuron(tau=10, dt=0.1, threshold=1, rest_potential=0, reset_potential=0, refractory_time=2)

for i in range(1000):

    is_spike = neuron.update(input_current=1.5)

    spikes.append(is_spike)
    voltages.append(neuron.current_voltage)




plt.plot(voltages, label='Voltage')
plt.plot(spikes, label='Spikes')
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
plt.legend()
plt.show()

