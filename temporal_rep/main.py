# %% 
import os
import sys
import plotly.express as px
import torch as t
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import einops
from jaxtyping import Int, Float
from typing import List, Optional, Tuple
import functools
from tqdm import tqdm
from IPython.display import display
import webbrowser
import gdown
from transformer_lens.hook_points import HookPoint
from transformer_lens import utils, HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
import circuitsvis as cv


# %%
# Load GPT-2 Small
device = t.device("cuda" if t.cuda.is_available() else "cpu")
model = HookedTransformer.from_pretrained("gpt2-small").to(device)
t.set_grad_enabled(False)

# %%
# Random model misc testing
temp_prompt = "There is an F.B.I."
temp_output = (model(temp_prompt))
print(temp_output.shape)
max_output = temp_output.argmax(-1)
print(max_output)
print(model.to_str_tokens(max_output))

# %%
# Generate prompts
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
prompts = []
answers = []
for i in range(len(days)):
    prompts.append(f"If today is {days[i]}, tomorrow is")
    answers.append(days[(i+1)%len(days)])

prompt_tokens = model.to_tokens(prompts, prepend_bos=True)
answer_tokens = model.to_tokens(answers, prepend_bos=False)

seq_length = prompt_tokens.shape[-1]
batch_size = prompt_tokens.shape[0]

print(prompt_tokens)
print(answer_tokens)

# %%
# Run the model and get logits and activations
logits, activations = model.run_with_cache(prompt_tokens)
print(model.cfg)
# Get list of components we can edit
print(model.hook_dict)

# %% 
# Find baseline correct_probs for no ablation
no_ablation_logits = logits
no_ablation_probs = (t.softmax(logits, dim=-1))[:, -1, :]
no_ablation_correct_probs = no_ablation_probs[t.arange(len(days)), answer_tokens]

# %%
# Ablate every head according to an ablation method
def ablate_each_head(method):
    def wrapper():
        all_logits = t.zeros((model.cfg.n_layers, model.cfg.n_heads, batch_size, seq_length, model.cfg.d_vocab))
        for layer in range(model.cfg.n_layers):
            for head in range(model.cfg.n_heads):
                temp_hook_fn = functools.partial(method, head_idx=head)
                ablated_logits = model.run_with_hooks(prompt_tokens, fwd_hooks = [(utils.get_act_name('pattern', layer), temp_hook_fn)], prepend_bos=True,)
                #print(f"{ablated_logits.shape=}")
                all_logits[layer, head] = ablated_logits
        return all_logits
    return wrapper

# %%
# Zero ablation
@ablate_each_head
def zero_ablation_hook(
    attn_pattern: Float[Tensor, "batch heads seqQ seqK"],
    hook: HookPoint,
    head_idx: int
) -> Float[Tensor, "batch heads seqQ seqK"]:
    head_pattern = attn_pattern[:, head_idx]
    attn_pattern[:, head_idx] = t.zeros_like(head_pattern)
    return attn_pattern

# %%
zero_ablation_logits = zero_ablation_hook() # shape: (layers, heads, batch, seq_len, vocab)
zero_ablation_probs = (t.softmax(zero_ablation_logits, dim=-1))[..., -1, :] # shape: (layers, heads, batch, vocab)
zero_ablation_correct_probs = zero_ablation_probs[t.arange(len(days)), answer_tokens]
# %%
