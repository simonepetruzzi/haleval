import torch as t

from transformer_lens impor t (
    ActivationCache,
    FactoredMatrix,
    HookedTransformer,
    HookedTransformerConfig,
    utils,
)

device = t.device("cuda" if t.cuda.is_available() else "cpu")
