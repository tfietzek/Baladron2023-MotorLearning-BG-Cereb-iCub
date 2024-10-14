import matplotlib.pyplot as plt
import numpy as np
from glob import glob


def plot_error_inverse_kinematics(runs: tuple = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
                                  path: str = 'results/network_inverse_kinematic/'):

    fig, axs = plt.subplots(nrows=len(runs), ncols=1, figsize=(12, 4), sharex=True)

    for i, run in enumerate(runs):
        data = np.load(path + f'inverse_results_run{run}.npz')
        axs[i].plot(data['changed_angle'], data['error'])
        axs[i].set_title(f'Run {run}, Mean Error: {data["error"].mean():.4f}', y=1.0, pad=-14)
        axs[i].set_ylabel('Error')
        axs[i].set_xticks(data['changed_angle']), axs[i].set_xlabel('Angle')
        axs[i].grid()

    plt.show()


def choose_run_with_lowest_error(path: str = 'results/network_inverse_kinematic/'):
    pattern = path + 'inverse_results_run*.npz'

    lowest_error = np.inf
    lowest_run = None

    for glob_pattern in glob(pattern):
        data = np.load(glob_pattern)
        error = data['error'].mean()

        if error < lowest_error:
            lowest_error = error
            lowest_run = glob_pattern

    return np.load(lowest_run)


if __name__ == '__main__':
    plot_error_inverse_kinematics()
