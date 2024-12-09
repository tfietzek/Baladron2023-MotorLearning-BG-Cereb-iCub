from .definitions import *
from .connections import weights_to_cpg


ann.setup(num_threads=4)

# input populations
S1 = ann.Population(geometry=parameters['dim_s1'], neuron=BaselineNeuron, name='S1')
SNc = ann.Population(geometry=1, neuron=DopamineNeuron, name='SNc')

# CBGT Loop (putamen)
StrD1 = ann.Population(geometry=parameters['dim_bg'], neuron=StriatumNeuron, name='StrD1')
StrD1.noise = 0.05

# StrD2 = ann.Population(geometry=parameters['dim_bg'], neuron=LinearNeuron, name='StrD2')
# StrD2.noise = 0.0

SNr = ann.Population(geometry=parameters['dim_bg'], neuron=LinearNeuron, name='SNr')
SNr.baseline = parameters['baseline_snr']
SNr.noise = 0.02

VL = ann.Population(geometry=parameters['dim_bg'], neuron=LinearNeuron, name='VL')
VL.baseline = parameters['baseline_thalamus']
VL.noise = 0.02

M1 = ann.Population(geometry=parameters['dim_bg'], neuron=M1Neuron, name='M1')
M1.tau = 20.
M1.noise = 0.05

# brainstem
Brainstem = ann.Population(geometry=parameters['dim_bg'], neuron=StaticNeuron, name='Brainstem')
CPG_output = ann.Population(geometry=parameters['dim_cpg'], neuron=StaticNeuron, name='CPG_output')

# connections feedforward
S1_StrD1 = ann.Projection(pre=S1, post=StrD1, target='exc', name='S1_StrD1', synapse=PostCovarianceNoThreshold)
S1_StrD1.connect_all_to_all(weights=parameters['init_w_striatum'])

StrD1_SNr = ann.Projection(pre=StrD1, post=SNr, target='inh', name='StrD1_SNr')
StrD1_SNr.connect_one_to_one(weights=parameters['w_snr'])

SNr_VL = ann.Projection(pre=SNr, post=VL, target='inh', name='SNr_VL')
SNr_VL.connect_one_to_one(weights=parameters['w_thalamus'])

VL_M1 = ann.Projection(pre=VL, post=M1, target='exc', name='VL_M1')
VL_M1.connect_one_to_one(weights=parameters['w_m1'])

# connections feedback
M1_StrD1 = ann.Projection(pre=M1, post=StrD1, target='exc', name='M1_StrD1')
M1_StrD1.connect_one_to_one(weights=0.5)

# connections Output
w_cpg = weights_to_cpg(file=parameters['cpg_weights_file'],
                       cpg_dim=parameters['dim_cpg'])
Brainstem_CPG = ann.Projection(pre=Brainstem, post=CPG_output, target='exc', name='Brainstem_CPG')
Brainstem_CPG.connect_from_matrix(w_cpg)

# connections DA
SNc_StrD1 = ann.Projection(pre=SNc, post=StrD1, target='dopa', name="Reward")
SNc_StrD1.connect_all_to_all(1.0)

StrD1_SNc = ann.Projection(pre=StrD1, post=SNc, target='inh', synapse=DAPrediction, name="RPE")
StrD1_SNc.connect_all_to_all(0.0)

# connections lateral
StrD1_StrD1 = ann.Projection(pre=StrD1, post=StrD1, target='inh', name='StrD1_StrD1')
StrD1_StrD1.connect_all_to_all(weights=0.2)

SNr_SNr = ann.Projection(pre=SNr, post=SNr, target='exc', synapse=ReversedSynapse, name='SNr_SNr')
SNr_SNr.connect_all_to_all(weights=0.05)

VL_VL = ann.Projection(pre=VL, post=VL, target='inh', name='VL_VL')
VL_VL.connect_all_to_all(weights=0.05)

M1_M1 = ann.Projection(pre=M1, post=M1, target='inh', name='M1_M1')
M1_M1.connect_all_to_all(weights=0.1)

