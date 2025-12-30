import numpy as np


class LIFNeuron:
    """
    A simple neuron model.
    """
    def __init__(self, tau: float, dt: float, threshold: float, rest_potential: float, 
    reset_potential: float, refractory_time: float):
        '''
        Initialize the neuron with the given parameters.
        Args:
            tau (float): Time constant of the neuron in ms. If tau is large, the neuron will have a long memory.
            dt (float): Time step in ms. Sensitivity of the simulation.
            threshold (float): Threshold potential of the neuron in V. If the neuron's potential exceeds this value, it will fire.
            rest_potential (float): Resting potential of the neuron in V. 
            reset_potential (float): Reset potential of the neuron in V. When the neuron fires, its potential is reset to this value.
            refractory_time (float): Refractory period of the neuron in ms. The neuron cannot fire during this period.
        '''
        self.tau = tau
        self.dt = dt
        self.threshold = threshold
        self.rest_potential = rest_potential
        self.reset_potential = reset_potential
        self.refractory_time = refractory_time
        self.decay_factor = np.exp(-self.dt/self.tau) # The number that determines how much of the voltage remains at each step
        self.ref_steps = int(self.refractory_time / self.dt) # Converts the refractory time(ms) to steps
        self.current_voltage = self.rest_potential # At the beginning the voltage is at rest potential
        self.ref_count = 0 # The number of steps the neuron has been in the refractory period

    def update(self, input_current: float):
        """
        Update the neuron's state.
        Args:
            input_current (float): Input current in V. The current that flows into the neuron.
        """
        if self.ref_count > 0: # If the neuron is in the refractory period (tired) it cannot fire 
            self.current_voltage = self.reset_potential
            self.ref_count -= 1
            return 0
        else:
            self.current_voltage = (self.current_voltage * self.decay_factor) + (input_current * (1-self.decay_factor)) # Update the voltage (1-self.decay_factor simulates the Resistance)
            if self.current_voltage >= self.threshold:
                self.ref_count = self.ref_steps
                self.current_voltage = self.reset_potential
                return 1
            else:
                return 0
        