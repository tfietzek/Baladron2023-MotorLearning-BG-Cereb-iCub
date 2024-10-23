import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import re
import os


def extract_tuple(input_string):
    # Regular expression to match a tuple inside square brackets
    match = re.search(r'\[\(([^)]+)\)\]', input_string)
    if match:
        return f"({match.group(1)})"
    return None


def calculate_distance(df: pd.DataFrame, angle_1: float, angle_2: float) -> float:

    # Filter the DataFrame by the given angles
    pos1 = df[df['changed_angle'] == angle_1]['init_pos'].values
    pos2 = df[df['changed_angle'] == angle_2]['init_pos'].values

    # Check if both positions were found for the angles
    if len(pos1) == 0 or len(pos2) == 0:
        raise ValueError(f"One or both angles {angle_1} and {angle_2} not found in the DataFrame.")

    # Calculate the Euclidean distance between the two init_pos
    distance = np.linalg.norm(np.array(pos1[0]) - np.array(pos2[0]))

    return distance


if __name__ == '__main__':
    data_sets = ('RHI_j11_sigma2', 'RHI_j12_sigma4')
    # Initialize an empty plot
    plt.figure(figsize=(10, 6))

    # Load the data and plot each dataset's stats on the same plot
    for data_set in data_sets:
        pattern = f'results/{data_set}/mlp_shape*/results_on_test_set.npz'
        for glob_pattern in glob(pattern):
            data = np.load(glob_pattern)
            hidden_layer_size = extract_tuple(glob_pattern)

            data_df = pd.DataFrame({
                'theta': data['proprio_angles'],
                'diff': np.abs(data['proprio_angles'] - data['vision_angles']),
                'error': data['errors']
            })


            #data_df = data_df[data_df['theta'] < 48.5]
            # Calculate mean error and standard error for each diff
            error_stats = data_df.groupby('diff')['error'].agg(['mean', 'std']).reset_index()
            error_stats.columns = ['diff', f'mean_error_{hidden_layer_size}', f'std_error_{hidden_layer_size}']
            error_stats[f'std_error_{hidden_layer_size}'] = error_stats[f'std_error_{hidden_layer_size}'] / np.sqrt(len(data_df.groupby('diff')['error'].count()))

            # Calculate the upper and lower bounds for the error range
            error_stats['lower_bound'] = error_stats[f'mean_error_{hidden_layer_size}'] - error_stats[f'std_error_{hidden_layer_size}']
            error_stats['upper_bound'] = error_stats[f'mean_error_{hidden_layer_size}'] + error_stats[f'std_error_{hidden_layer_size}']

            # Plot the mean error line for this dataset
            plt.plot(error_stats['diff'], error_stats[f'mean_error_{hidden_layer_size}'], label=f'Mean Error {hidden_layer_size}')

            # Plot the shaded error range for this dataset
            plt.fill_between(error_stats['diff'], error_stats['lower_bound'], error_stats['upper_bound'],
                             alpha=0.1)

        # Finalize the plot
        plt.legend()
        plt.title(f'Mean Error for regMLP with Different Hidden Layer Sizes | Data Set: {data_set}')
        plt.xlabel('$\\Delta \\Theta$  in [°]')
        plt.ylabel('Error in [m]')
        plt.ylim(0, 0.1)
        plt.savefig(f'results/{data_set}/combined_mean_error_plot.pdf', dpi=300, bbox_inches='tight')
        plt.show()
