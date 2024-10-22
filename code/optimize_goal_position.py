import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize


def calculate_equidistant_point(positions: list | np.ndarray, min_distance: float = 0.2):
    if not isinstance(positions, np.ndarray):
        positions = np.array(positions)

    def distance_variance(point):
        distances = np.linalg.norm(positions - point, axis=1)
        return np.var(distances)

    def min_distance_constraint(point):
        distances = np.linalg.norm(positions - point, axis=1)
        return np.min(distances) - min_distance

    initial_guess = np.mean(positions, axis=0)

    constraints = ({'type': 'ineq', 'fun': min_distance_constraint})

    result = minimize(distance_variance, initial_guess, method='SLSQP', constraints=constraints)

    return result.x

if __name__ == '__main__':

    data = np.load('results/RHI_j11_sigma2/network_inverse_kinematic/inverse_results_run1.npz')
    df = pd.DataFrame({
        'changed_angle': data['changed_angle'],
        'init_pos': data['initial_position'].tolist(),
    })

    point = calculate_equidistant_point(df['init_pos'].tolist())
    print(point)

    plt.plot(df['changed_angle'], np.linalg.norm(np.array(df['init_pos'].tolist()) - point, axis=1), 'o')
    plt.xlabel('Angle in [°]')
    plt.ylabel('Distance to goal in [m]')
    plt.savefig('optimized_distance_to_goal.pdf', dpi=300, bbox_inches='tight')
    plt.show()