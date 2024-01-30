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
from plotly.subplots import make_subplots

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
# Generate regular prompts
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
prompts = []
answers = []
for i in range(len(days)):
    prompts.append(f"If today is {days[i]}, tomorrow is")
    answers.append(" " + days[(i+1)%len(days)])

prompt_tokens = model.to_tokens(prompts, prepend_bos=True)
answer_tokens = model.to_tokens(answers, prepend_bos=False).reshape((len(days)))

seq_length = prompt_tokens.shape[-1]
batch_size = prompt_tokens.shape[0]

print(prompt_tokens)
print(answer_tokens)

#%% 
# Generate random prompts
def random_replace(num_prompts: int, indices: List[int]):
    random_prompts = t.zeros((num_prompts, seq_length), dtype=t.int)
    for prompt in range(num_prompts):
        random_prompt_tokens = (prompt_tokens[0]).clone().to(device)
        for idx in indices:
            while True:
                rand_tok = t.randint(0, model.cfg.d_vocab, size=(1,))
                if model.to_string(rand_tok)[0] == " ":
                    random_prompt_tokens[idx] = rand_tok
                    break
        random_prompts[prompt] = random_prompt_tokens
    return random_prompts

# %%
# Run the model and get logits and activations
logits, activations = model.run_with_cache(prompt_tokens)
print(model.cfg)
# Get list of components we can edit
print(model.hook_dict)

# %% 
# Find baseline correct_probs for no ablation
no_ablation_logits = logits # shape: (batch, seq_length, vocab)
no_ablation_probs = (t.softmax(logits, dim=-1))[:, -1, :] # shape: (batch, vocab)

no_ablation_correct_probs = no_ablation_probs[t.arange(len(days)), answer_tokens] # shape: (batch)
# %%
# Ablate every head according to an ablation method
def ablate_heads(prompt_tokens):
    def ablate_each_head(method):
        def wrapper():
            all_logits = t.zeros((model.cfg.n_layers, model.cfg.n_heads, batch_size, seq_length, model.cfg.d_vocab)).to(device)
            for layer in range(model.cfg.n_layers):
                for head in range(model.cfg.n_heads):
                    temp_hook_fn = functools.partial(method, head_idx=head)
                    ablated_logits = model.run_with_hooks(prompt_tokens, fwd_hooks = [(utils.get_act_name('pattern', layer), temp_hook_fn)], prepend_bos=True,)
                    #print(f"{ablated_logits.shape=}")
                    all_logits[layer, head] = ablated_logits
            return all_logits
        return wrapper
    return ablate_each_head

# %%
# Hook functions

# Zero ablation
@ablate_heads(prompt_tokens)
def zero_ablation_hook(
    attn_pattern: Float[Tensor, "batch heads seqQ seqK"],
    hook: HookPoint,
    head_idx: int
) -> Float[Tensor, "batch heads seqQ seqK"]:
    head_pattern = attn_pattern[:, head_idx]
    attn_pattern[:, head_idx] = t.zeros_like(head_pattern).to(device)
    return attn_pattern

# Random ablation
# Store average of each head
D = 256
random_prompt_tokens = random_replace(D, [2, 4, 6])
print(model.to_string(random_prompt_tokens))
# Find the average of each head
@ablate_heads(prompt_tokens)
def random_ablation_hook(
    attn_pattern: Float[Tensor, "batch heads seqQ seqK"],
    hook: HookPoint,
    head_idx: int
) -> Float[Tensor, "batch heads seqQ seqK"]:
    head_pattern = attn_pattern[:, head_idx]
    attn_pattern[:, head_idx] = t.mean(head_pattern, dim=0).unsqueeze(0) #average across dataset
    return attn_pattern

# %%
# Find correct probs for zero ablation method
zero_ablation_logits = zero_ablation_hook() # shape: (layers, heads, batch, seq_len, vocab)
zero_ablation_probs = t.softmax(zero_ablation_logits[..., -1, :], dim=-1) # shape: (layers, heads, batch, vocab)
zero_ablation_correct_probs = zero_ablation_probs[..., t.arange(len(days)), answer_tokens] # shape: (layers, heads, batch)

# Find average (across the batches) correct probability difference, for each (layer, head) activation
zero_ablation_prob_diff = t.mean(zero_ablation_correct_probs - no_ablation_correct_probs, dim=-1) # shape: (layers, heads)

#%%
random_ablation_logits = random_ablation_hook() # shape: (layers, heads, batch, seq_len, vocab)
print(random_ablation_logits.shape)
# %%
curr_fig = make_subplots(rows=1, cols=3, subplot_titles=["Zero ablation", "Random ablation", "Mean ablation"], horizontal_spacing=0.05,
                         x_title='Layer',
                          y_title='Head')
curr_fig.update_layout(title_text="Change in correct probs", height=300, width=200*3)

curr_fig.update_layout(coloraxis_autocolorscale=False, coloraxis_colorscale="plasma_r")

zero_fig = px.imshow(zero_ablation_prob_diff.T.cpu())

#random_fig = px.imshow(random_avg_probs.T)

#mean_fig = px.imshow(mean_avg_probs.T)

curr_fig.append_trace(zero_fig.data[0], row=1, col=1)
#curr_fig.append_trace(random_fig.data[0], row=1, col=2)
#curr_fig.append_trace(mean_fig.data[0], row=1, col=3)

curr_fig.show()

# %%
# %%
