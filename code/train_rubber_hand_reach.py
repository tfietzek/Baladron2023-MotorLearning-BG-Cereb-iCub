from bg_network.reaching_model import *

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
    ann.simulate(wait_time)
    S1.baseline = s1_inputs
    if training:
        SNc.firing = 1
    if m1_inputs is not None:
        M1.baseline = m1_inputs

    ann.simulate(reach_time)

    m1_rates = M1.r
    ann.reset(monitors=False, populations=True)

    return m1_rates


def simulate_cpg(m1_rates: np.ndarray,
                 temperature: float = 1.0,
                 encodings_m1: np.ndarray = parameters["encodings_m1"]
                 ) -> Tuple[np.ndarray, np.ndarray]:
    decision = softmax(m1_rates, temperature)
    Brainstem.baseline = decision

    ann.step()
    ann.reset(monitors=False, populations=True)

    return CPG_output.r, decision * encodings_m1


def train_over_inputs(
        s1_inputs: np.ndarray,
        m1_inputs: np.ndarray,
        wait_time: float = 50.,
        reach_time: float = 350.,
):
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
        s1_inputs: np.ndarray,
        wait_time: float = 50.,
        reach_time: float = 350.,
):
    if s1_inputs.ndim == 1:
        s1_inputs = s1_inputs.reshape(1, -1)

    # disable learning
    ann.disable_learning()

    cpgs = []
    angles = []

    for i in range(s1_inputs.shape[0]):
        m1_rates = simulate_reaching(s1_inputs=s1_inputs[i],
                                     m1_inputs=None,
                                     training=False,
                                     wait_time=wait_time,
                                     reach_time=reach_time)

        cpg_output, angle_output = simulate_cpg(m1_rates=m1_rates)
        cpgs.append(cpg_output), angles.append(angle_output)

    return np.array(cpgs), np.array(angles)