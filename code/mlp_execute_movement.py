import importlib
import os.path
from typing import Optional
import pandas as pd
import numpy as np
import gc

# Model
from kinematic import *
from cpg import *
from train_BG_reaching import execute_movement, random_goal2_iCub
from mlp_inverse_fit import load_training_rhi_thetas, save_mlp

# CPG
import CPG_lib.parameter as params
from CPG_lib.MLMPCPG.MLMPCPG import *
from CPG_lib.MLMPCPG.myPloting import *
from CPG_lib.MLMPCPG.SetTiming import *
from scipy.optimize import minimize

from mlp_inverse_fit import train_mlp, test_mlp, train_mlps, merge_training_data, merge_test_data, get_prediction
