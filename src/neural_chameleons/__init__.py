"""Neural Chameleons: Can Activation Oracles detect models hiding from probes?"""

from .probes import LinearProbe, MLPProbe, train_probe, evaluate_probe
from .data import BENIGN_CONCEPTS, TRIGGER_TEMPLATE, generate_concept_data
from .utils import (
    get_activations,
    ActivationCache,
    ActivationInjector,
    apply_oracle_math,
    query_activation_oracle,
)
