from rhi_network.reaching_model import *

import numpy as np
from typing import Optional, Tuple


def softmax(x: np.array,
            temperature: float) -> np.ndarray:
    return np.exp(x / temperature) / np.sum(np.exp(x / temperature))


def simulate_reaching(
        s1_inputs: np.ndarray,
        m1_inputs: Optional[np.ndarray],
        training: bool,
        wait_time: float = 50.,
        reach_time: float = 350.,
):
    # build up baseline activities
    ann.simulate(wait_time)

    # simulate reaching process
    S1.baseline = s1_inputs
    if training:
        # build up baseline activities
        M1.baseline = m1_inputs
        ann.simulate(50.)
        # reward
        SNc.firing = 1
        ann.simulate(reach_time)
    else:
        ann.simulate(reach_time)

    # Readout
    m1_rates = M1.r

    # reset to start conditions
    ann.reset(monitors=False, populations=True)

    return m1_rates


def simulate_cpg(m1_rates: np.ndarray,
                 temperature: float = 1.0,
                 encodings_m1: np.ndarray = parameters["encodings_m1"]
                 ) -> Tuple[np.ndarray, float]:
    """
    Simulate the CPG output and the angle output
    :param m1_rates: Firing rates of M1
    :param temperature: Temperature of the softmax
    :param encodings_m1: Encodings of M1 (Movement based on starting angles)
    :return: CPG output and presumed starting angle of movement
    """
    decision = softmax(m1_rates, temperature)
    Brainstem.baseline = decision

    ann.step()
    ann.reset(monitors=False, populations=True)

    return CPG_output.r, np.dot(decision, encodings_m1)


def train_over_inputs(
        s1_inputs: np.ndarray | list,
        m1_inputs: np.ndarray | list,
        wait_time: float = 50.,
        reach_time: float = 350.,
):
    if isinstance(s1_inputs, list):
        s1_inputs = np.array(s1_inputs)
    if isinstance(m1_inputs, list):
        m1_inputs = np.array(m1_inputs)

    if s1_inputs.ndim == 1:
        s1_inputs = s1_inputs.reshape(1, -1)
    if m1_inputs.ndim == 1:
        m1_inputs = m1_inputs.reshape(1, -1)

    # check if
    assert s1_inputs.shape[0] == m1_inputs.shape[0], "Number of s1_inputs and m1_inputs should be equal"

    # enable learning
    ann.enable_learning()

    for i in range(s1_inputs.shape[0]):
        simulate_reaching(s1_inputs=s1_inputs[i],
                          m1_inputs=m1_inputs[i],
                          training=True,
                          wait_time=wait_time,
                          reach_time=reach_time)


def predict_over_inputs(
        s1_inputs: np.ndarray | list,
        wait_time: float = 50.,
        reach_time: float = 350.,
        temperature: float = 0.1
):
    if isinstance(s1_inputs, list):
        s1_inputs = np.array(s1_inputs)

    if s1_inputs.ndim == 1:
        s1_inputs = s1_inputs.reshape(1, -1)

    # disable learning
    ann.disable_learning()

    m1_rates = []
    cpgs = []
    angles = []

    for i in range(s1_inputs.shape[0]):
        m1_rate = simulate_reaching(s1_inputs=s1_inputs[i],
                                    m1_inputs=None,
                                    training=False,
                                    wait_time=wait_time,
                                    reach_time=reach_time)

        cpg_output, angle_output = simulate_cpg(m1_rates=m1_rate, temperature=temperature)
        m1_rates.append(m1_rate), cpgs.append(cpg_output), angles.append(angle_output)

    return m1_rates, cpgs, angles
