import ANNarchy as ann
from .params import parameters

# Neuron definitions
StaticNeuron = ann.Neuron(
    parameters="""
        baseline = 0.0
        noise = 0.0 : population
    """,
    equations="""
    r = sum(exc) + baseline + noise*Uniform(-1.0,1.0)
    """,
    description="Static neuron with baseline to be set."
)

BaselineNeuron = ann.Neuron(
    parameters="""
        tau_up = 10.0 : population
        tau_down = 20.0 : population
        baseline = 0.0
        noise = 0.0 : population
    """,
    equations="""
        base = baseline + noise * Uniform(-1.0,1.0): min=0.0
        dr/dt = if (baseline>0.01): (base-r)/tau_up else: -r/tau_down : min=0.0
    """,
    name="Baseline Neuron",
    description="Time-dynamic neuron with baseline to be set. "
)

StriatumNeuron = ann.Neuron(
    parameters="""
        tau = 20.0 : population
        baseline = 0.0
        noise = 0.0 : population
""",
    equations="""
        tau*dmp/dt + mp = sum(exc) - sum(inh) + noise*Uniform(-1.0,1.0) + baseline
        r = tanh(pos(mp)) 
    """,
    description="Striatum Neuron with normalized firing rates (FSNs normalize MSNs activity)."

)

M1Neuron = ann.Neuron(
    parameters="""
        tau = 20.0 : population
        baseline = 0.0
        noise = 0.0 : population
    """,
    equations="""
        tau*dmp/dt + mp = sum(exc) - sum(inh) + noise*Uniform(-1.0,1.0) + baseline
        r = pos(mp) 
    """
)

LinearNeuron = ann.Neuron(
    parameters="""
        tau = 20.0 : population
        baseline = 0.0: population
        noise = 0.0 : population
    """,
    equations="""
        tau*dmp/dt + mp = sum(exc) - sum(inh) + noise*Uniform(-1.0,1.0) + baseline
        r = pos(mp) 
    """
)

DopamineNeuron = ann.Neuron(
    parameters="""
        tau = 20.0 : population
        firing = 0 : population, bool
        factor_inh = 10.0 : population
        baseline = 'baseline_dopa': population
    """,
    equations="""
        s_inh = sum(inh)
        aux = firing * pos(1.0 - s_inh) + (1-firing)*baseline
        tau*dmp/dt + mp = aux
        r = pos(mp)
    """,
    extra_values=parameters,
)

# Synapse definitions
ReversedSynapse = ann.Synapse(
    parameters="""
        reversal = 1.1 : projection
    """,
    psp="""
        w*pos(reversal-pre.r)
    """,
    name="Reversed Synapse",
    description="Higher pre-synaptic activity lowers the synaptic transmission and vice versa."
)

# DA_typ = 1  ==> D1 type  DA_typ = -1 ==> D2 type
PostCovarianceNoThreshold = ann.Synapse(
    parameters="""
        tau = 1500.0 : projection
        tau_alpha = 1500.0 : projection
        regularization_threshold = 'alpha_regularization' : projection
        K_burst = 1.0 : projection
        K_dip = 0.4 : projection
        DA_type = 1 : projection
        threshold_pre = 'reg_threshold_s1' : projection
        threshold_post = 'reg_threshold_d1' : projection
        baseline = 'baseline_dopa' : projection
    """,
    equations="""
        tau_alpha*dalpha/dt + alpha = pos(post.r - regularization_threshold)
        dopa_sum = 2.0*(post.sum(dopa) - baseline)
        trace = (post.r -  mean(post.r) - threshold_post) * pos(pre.r - mean(pre.r) - threshold_pre)
        condition_0 = if (trace>0.0) and (w >0.0): 1 else: 0
        dopa_mod =  if (DA_type*dopa_sum>0): DA_type*K_burst*dopa_sum
                    else: condition_0*DA_type*K_dip*dopa_sum
        alpha_trace = clip(alpha*pos(post.r - mean(post.r) - threshold_post), 0, trace)
        tau*dw/dt = dopa_mod * (trace - alpha_trace) : min = 0.0
    """,
    name="PostCovariance",
    description="Post covariance synapse.",
    extra_values=parameters
)

DAPrediction = ann.Synapse(
    parameters="""
        tau = 1000.0 : projection
        threshold = 'regularization_rpe' : projection
        baseline = 'baseline_dopa': projection
    """,
    equations="""
       aux = if (post.mp>0): 1.0 else: 3.0
       delta = aux*pos(post.r - baseline)*pos(pre.r - mean(pre.r) - threshold)
       tau*dw/dt = delta : min = 0.0
    """,
    extra_values=parameters
)
