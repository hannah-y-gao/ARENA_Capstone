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
import random
# %%
# Load GPT-2 Small
device = t.device("cuda" if t.cuda.is_available() else "cpu")
model = HookedTransformer.from_pretrained("gpt2-small").to(device)
model.cfg.use_result_hook = True
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
# Plotting functions from original LessWrong post:
def show_pp(
    m,
    range_color=None,
    xlabel="",
    ylabel="",
    title="",
    bartitle="",
    return_fig=False,
    show_fig=True,
    **kwargs,
):
    """
    Plot a heatmap of the values in the matrix `m`
    """

    if range_color:

        fig = px.imshow(
            m.T,
            title=title if title else "",
            color_continuous_scale="magenta",
            #color_continuous_midpoint=-1,
            #**kwargs,
            range_color=range_color,
        )
    else:
        fig = px.imshow(
            m.T,
            title=title if title else "",
            color_continuous_scale="magenta",
        )

    if show_fig:
        fig.show()
    if return_fig:
        return fig
    
def general_plot_logits_probs(prompts, logit_matrices, prob_matrices, titles, width_step=200):

    # logit_matrices, prob_matrices: (len(prompts), n_layers, n_heads)

    logit_figs = []
    prob_figs = []

    for i in range(len(prompts)):
        prob_fig = show_pp(prob_matrices[i].cpu().detach().numpy(), xlabel='layers', ylabel='heads', return_fig=True, show_fig=False)
        logit_fig = show_pp(logit_matrices[i].cpu().detach().numpy(), xlabel='layers', ylabel='heads', return_fig=True, show_fig=False)

        logit_figs.append(logit_fig)
        prob_figs.append(prob_fig)

    fig_for_prob = make_subplots(rows=1, cols=len(prompts), subplot_titles=titles, x_title="layers", y_title="heads", horizontal_spacing=0.05)
    fig_for_prob.update_layout(title_text=f"Probabilities of subject tokens", height=300, width=width_step*len(prompts))

    fig_for_logit = make_subplots(rows=1, cols=len(prompts), subplot_titles=titles, x_title="layers", y_title="heads", horizontal_spacing=0.05)
    fig_for_logit.update_layout(title_text=f"Logits of subject tokens", height=300, width=width_step*len(prompts))

    for i in range(len(prompts)):

        fig_for_prob.append_trace(prob_figs[i].data[0], row=1, col=i+1)
        fig_for_logit.append_trace(logit_figs[i].data[0], row=1, col=i+1)

    fig_for_prob.show()
    fig_for_logit.show()

def plot_pixels(prompts, logit_matrices, titles, desc="diff.", midpoint=True):

    # logit_matrices: (len(prompts), len(prompts))

    logit_figs = []

    logit_fig = show_pp(logit_matrices.T.cpu().detach().numpy(), xlabel='Day token', ylabel='Prompt', return_fig=True, show_fig=False, midpoint=midpoint)

    logit_figs.append(logit_fig)

    fig_for_logit = make_subplots(rows=1, cols=1, x_title="Day token", y_title="Prompt", horizontal_spacing=0.05)
    fig_for_logit.update_layout(title_text=f"{desc}", height=400, width=400)


    fig_for_logit.append_trace(logit_figs[0].data[0], row=1, col=1)

    fig_for_logit.show()
# %%
# Generate regular prompts
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
prompts = []
answers = []
subjects = []
for i in range(len(days)):
    prompts.append(f"If today is {days[i]}, tomorrow is")
    answers.append(" " + days[(i+1)%len(days)])
    subjects.append(" " + days[i])

prompt_tokens = model.to_tokens(prompts, prepend_bos=True)
answer_tokens = model.to_tokens(answers, prepend_bos=False).reshape((len(days)))
subject_tokens = model.to_tokens(subjects, prepend_bos=False).reshape((len(days)))

seq_length = prompt_tokens.shape[-1]
batch_size = prompt_tokens.shape[0]

print(prompt_tokens)
print(answer_tokens)

#%% 
# Generate random prompts
seed = 123
def random_replace(num_prompts: int, indices: List[int]):
    random.seed(seed)
    random_prompts = t.zeros((num_prompts, seq_length), dtype=t.int)
    for prompt in range(num_prompts):
        random_prompt_tokens = (prompt_tokens[0]).clone().to(device)
        for idx in indices:
            while True:
                rand_tok = random.randrange(model.cfg.d_vocab)
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
no_ablation_correct_logits = no_ablation_logits[:, -1, :][t.arange(len(days)), answer_tokens] # shape: (batch)
# %%
# Ablate every head according to an ablation method
def ablate_heads(toks, name="pattern"):
    def ablate_each_head(method):
        def wrapper():
            batch_size = len(toks)
            all_logits = t.zeros((model.cfg.n_layers, model.cfg.n_heads, batch_size, seq_length, model.cfg.d_vocab)).to(device)
            for layer in range(model.cfg.n_layers):
                for head in range(model.cfg.n_heads):
                    temp_hook_fn = functools.partial(method, layer_idx=layer, head_idx=head)
                    ablated_logits = model.run_with_hooks(toks, fwd_hooks = [(utils.get_act_name(name, layer), temp_hook_fn)], prepend_bos=True,)
                    all_logits[layer, head] = ablated_logits
            return all_logits
        return wrapper
    return ablate_each_head

# %%
# Zero ablation setup

@ablate_heads(prompt_tokens)
def zero_ablation_hook(
    attn_pattern: Float[Tensor, "batch heads seqQ seqK"],
    hook: HookPoint,
    layer_idx: int,
    head_idx: int
) -> Float[Tensor, "batch heads seqQ seqK"]:
    head_pattern = attn_pattern[:, head_idx]
    attn_pattern[:, head_idx] = t.zeros_like(head_pattern).to(device) #replace attn pattern with zeros
    return attn_pattern

# %% 
# Random ablation setup

# Generate random dataset
D = 256
random_prompt_tokens = random_replace(D, [2, 4, 6])
print(model.to_string(random_prompt_tokens))
print(random_prompt_tokens.shape)

# Find the average of each head and store
random_storage = t.zeros((model.cfg.n_layers, model.cfg.n_heads, seq_length, seq_length))
@ablate_heads(random_prompt_tokens)
def random_dataset_hook(
    attn_pattern: Float[Tensor, "batch heads seqQ seqK"],
    hook: HookPoint,
    layer_idx: int,
    head_idx: int
) -> Float[Tensor, "batch heads seqQ seqK"]:
    head_pattern = attn_pattern[:, head_idx]
    random_storage[layer_idx, head_idx] = t.mean(head_pattern, dim=0) #average across dataset (shape: (seqQ, seqK))

random_dataset_hook()

# Ablate each head with the random dataset attention heads
@ablate_heads(prompt_tokens)
def random_ablation_hook(
    attn_pattern: Float[Tensor, "batch heads seqQ seqK"],
    hook: HookPoint,
    layer_idx: int,
    head_idx: int
) -> Float[Tensor, "batch heads seqQ seqK"]:
    attn_pattern[:, head_idx, ...] = random_storage[layer_idx, head_idx].unsqueeze(0)
    return attn_pattern

# %%
# Mean ablation setup

# Generate random dataset
D = 256
mean_prompt_tokens = random_replace(D, [4,])
print(model.to_string(mean_prompt_tokens))
print(mean_prompt_tokens.shape)

# Find the average of each head and store
mean_storage = t.zeros((model.cfg.n_layers, model.cfg.n_heads, seq_length, seq_length))
@ablate_heads(mean_prompt_tokens)
def mean_dataset_hook(
    attn_pattern: Float[Tensor, "batch heads seqQ seqK"],
    hook: HookPoint,
    layer_idx: int,
    head_idx: int
):
    head_pattern = attn_pattern[:, head_idx]
    mean_storage[layer_idx, head_idx] = t.mean(head_pattern, dim=0) #average across dataset (shape: (seqQ, seqK))

mean_dataset_hook()

# Ablate each head with the random dataset attention heads
@ablate_heads(prompt_tokens)
def mean_ablation_hook(
    attn_pattern: Float[Tensor, "batch heads seqQ seqK"],
    hook: HookPoint,
    layer_idx: int,
    head_idx: int
) -> Float[Tensor, "batch heads seqQ seqK"]:
    attn_pattern[:, head_idx, ...] = mean_storage[layer_idx, head_idx].unsqueeze(0)
    return attn_pattern

# %%
# Find correct probs for zero ablation method
zero_ablation_logits = zero_ablation_hook() # shape: (layers, heads, batch, seq_len, vocab)
zero_ablation_probs = t.softmax(zero_ablation_logits[..., -1, :], dim=-1) # shape: (layers, heads, batch, vocab)
zero_ablation_correct_probs = zero_ablation_probs[..., t.arange(len(days)), answer_tokens] # shape: (layers, heads, batch)
zero_ablation_correct_logits = zero_ablation_logits[..., -1, :][..., t.arange(len(days)), answer_tokens] # shape: (layers, heads, batch)
# Find average (across the batches) correct probability difference, for each (layer, head) activation
zero_ablation_prob_diff = t.mean(zero_ablation_correct_probs - no_ablation_correct_probs, dim=-1) # shape: (layers, heads)
# Find average (across the batches) correct logit difference, for each (layer, head) activation
zero_ablation_logit_diff = t.mean(zero_ablation_correct_logits - no_ablation_correct_logits, dim=-1) # shape: (layers, heads)

#%%
# Find correct probs for random ablation method
random_ablation_logits = random_ablation_hook() # shape: (layers, heads, batch, seq_len, vocab)
random_ablation_probs = t.softmax(random_ablation_logits[..., -1, :], dim=-1) # shape: (layers, heads, batch, vocab)
random_ablation_correct_probs = random_ablation_probs[..., t.arange(len(days)), answer_tokens] # shape: (layers, heads, batch)
random_ablation_correct_logits = random_ablation_logits[..., -1, :][..., t.arange(len(days)), answer_tokens] # shape: (layers, heads, batch)
# Find average (across the batches) correct probability difference, for each (layer, head) activation
random_ablation_prob_diff = t.mean(random_ablation_correct_probs - no_ablation_correct_probs, dim=-1) # shape: (layers, heads)
# Find average (across the batches) correct logit difference, for each (layer, head) activation
random_ablation_logit_diff = t.mean(random_ablation_correct_logits - no_ablation_correct_logits, dim=-1) # shape: (layers, heads)

#%%
# Find correct probs for mean ablation method
mean_ablation_logits = mean_ablation_hook() # shape: (layers, heads, batch, seq_len, vocab)
mean_ablation_probs = t.softmax(mean_ablation_logits[..., -1, :], dim=-1) # shape: (layers, heads, batch, vocab)
mean_ablation_correct_probs = mean_ablation_probs[..., t.arange(len(days)), answer_tokens] # shape: (layers, heads, batch)
mean_ablation_correct_logits = mean_ablation_logits[..., -1, :][..., t.arange(len(days)), answer_tokens] # shape: (layers, heads, batch)
# Find average (across the batches) correct probability difference, for each (layer, head) activation
mean_ablation_prob_diff = t.mean(mean_ablation_correct_probs - no_ablation_correct_probs, dim=-1) # shape: (layers, heads)
# Find average (across the batches) correct logit difference, for each (layer, head) activation
mean_ablation_logit_diff = t.mean(mean_ablation_correct_logits - no_ablation_correct_logits, dim=-1) # shape: (layers, heads)

# %%
# Display plots for change in correct probs for 3 methods of ablation: zero, random, mean
prob_fig = make_subplots(rows=1, cols=3, subplot_titles=["Zero ablation", "Random ablation", "Mean ablation"], horizontal_spacing=0.05,
                         x_title='Layer',
                          y_title='Head')
prob_fig.update_layout(title_text="Change in Correct Probs", height=300, width=200*3)

prob_fig.update_layout(coloraxis_autocolorscale=False, coloraxis_colorscale="plasma_r")

zero_prob_fig = px.imshow(zero_ablation_prob_diff.T.cpu())

rando_prob_fig = px.imshow(random_ablation_prob_diff.T.cpu())

mean_prob_fig = px.imshow(mean_ablation_prob_diff.T.cpu())

prob_fig.append_trace(zero_prob_fig.data[0], row=1, col=1)
prob_fig.append_trace(rando_prob_fig.data[0], row=1, col=2)
prob_fig.append_trace(mean_prob_fig.data[0], row=1, col=3)

prob_fig.show()

# %%
# Display plots for change in correct logits for 3 methods of ablation: zero, random, mean
logit_fig = make_subplots(rows=1, cols=3, subplot_titles=["Zero ablation", "Random ablation", "Mean ablation"], horizontal_spacing=0.05,
                         x_title='Layer',
                          y_title='Head')
logit_fig.update_layout(title_text="Change in Correct Logits", height=300, width=200*3)

logit_fig.update_layout(coloraxis_autocolorscale=False, coloraxis_colorscale="viridis_r")

zero_logit_fig = px.imshow(zero_ablation_logit_diff.T.cpu())

random_logit_fig = px.imshow(random_ablation_logit_diff.T.cpu())

mean_logit_fig = px.imshow(mean_ablation_logit_diff.T.cpu())

logit_fig.append_trace(zero_logit_fig.data[0], row=1, col=1)
logit_fig.append_trace(random_logit_fig.data[0], row=1, col=2)
logit_fig.append_trace(mean_logit_fig.data[0], row=1, col=3)

logit_fig.show()

# %%
# Logit Lens

top_k = 3 
all_top_k_indices = t.zeros((batch_size, model.cfg.n_layers, model.cfg.n_heads, top_k)).to(device) #stores indices of top k tokens, shape: (batch, layers, heads, top_k)
all_probs= t.zeros((batch_size, model.cfg.n_layers, model.cfg.n_heads)).to(device) #stores probs of correct tokens, shape: (batch, layers, heads)
subject_probs = t.zeros((batch_size, model.cfg.n_layers, model.cfg.n_heads)).to(device) #stores probs of correct tokens, shape: (batch, layers, heads)
subject_logits = t.zeros((batch_size, model.cfg.n_layers, model.cfg.n_heads)).to(device) #stores probs of correct tokens, shape: (batch, layers, heads)

all_top_k_strings = [] #stores string representation, shape: (layers, batch, heads)

def logit_lens_hook(
    activation_value: Float[Tensor, "batch position heads d_head"],
    hook: HookPoint,
    starting_space: bool,
    layer_idx: int,
):
    W_O = model.W_O[layer_idx] # shape: (head, d_head, d_model)
    result = einops.einsum(activation_value, W_O, 'batch position heads d_head, heads d_head d_model -> batch heads position d_model')
    
    ln_result = model.ln_final(result)
    unembed_result = einops.einsum(ln_result, model.W_U, 'batch heads position d_model, d_model d_vocab -> batch heads position d_vocab')
    final_logits = unembed_result[..., -1, :]
    final_probs = t.softmax(final_logits, dim=-1) # shape: (batch, heads, d_vocab)
    top_k_dict = t.topk(final_probs, k=model.cfg.d_vocab, dim=-1) # shape: (batch, heads, d_vocab)
    top_k_tokens = top_k_dict.indices # sorted, shape: (batch, heads, d_vocab)
    top_k_probs = top_k_dict.values # sorted, shape: (batch, heads, d_vocab)
    
    top_k_strings = [] # shape" (batch, heads)
    for b in range(batch_size):
        batch_strings = []
        for h in range(model.cfg.n_heads):
            string = ""
            count = 0
            for i in range(model.cfg.d_vocab):
                if count >= top_k:
                    break
                next_string = model.to_string(top_k_tokens[b, h, i])
                if (not starting_space) or next_string[0] == " ":
                    string += (next_string) + "<br />"
                    count+=1
            batch_strings.append(string)
        top_k_strings.append(batch_strings)
    
    all_top_k_strings.append(top_k_strings)
    all_top_k_indices[:, layer_idx, :, :] = top_k_tokens[..., :top_k]    
    all_probs[:, layer_idx, :] = final_probs[t.arange(batch_size), :, answer_tokens]
    subject_probs[:, layer_idx, :] = final_probs[t.arange(batch_size), :, subject_tokens]
    subject_logits[:, layer_idx, :] = final_logits[t.arange(batch_size), :, subject_tokens]

# Run logit lens on each layer
def logit_lens(starting_space):
    batch_size = len(prompt_tokens)
    all_logits = t.zeros((model.cfg.n_layers, model.cfg.n_heads, batch_size, seq_length, model.cfg.d_vocab)).to(device)
    for layer in range(model.cfg.n_layers):
        temp_hook_fn = functools.partial(logit_lens_hook, layer_idx=layer, starting_space=starting_space)
        ablated_logits = model.run_with_hooks(prompt_tokens, fwd_hooks = [(utils.get_act_name("z", layer), temp_hook_fn)], prepend_bos=True,)

# %%
# Run logit lens
logit_lens(starting_space=False)

# Prepare matrices for plotting
string_matrix = einops.rearrange(np.array(all_top_k_strings), 'layers batch heads -> batch layers heads')
prob_matrix = all_probs
# %%
# Plots for each prompt
prompt_titles = [f"{days[i]} -> {days[(i+1)%len(days)]}" for i in range(len(days))]
print(prompt_titles)
for prompt in range(batch_size):
    logit_lens_fig = px.imshow(prob_matrix[prompt].cpu().detach().T, range_color=(0,1), origin="lower", height=1100, width=1300,
                         labels=dict(x="layers", y="heads"))
    logit_lens_fig.update_traces(text=string_matrix[prompt].T, texttemplate="%{text}", textfont=dict(
                family="Courier New, monospace",
                size=12,
                color="white"
            ))
    logit_lens_fig.update_layout(title_text=prompt_titles[prompt], title_x=0.5)
    logit_lens_fig.show()

# %%
# Investigate behavior of head (10, 3)
general_plot_logits_probs(prompt_tokens, subject_logits, subject_probs, prompt_titles)
# %%
# Display attention pattern for a particular head
def attn_pattern_hook(
    attn_pattern: Float[Tensor, "batch heads seqQ seqK"],
    hook: HookPoint,
    layer_idx: int,
    head_idx: int
):
    head_pattern = attn_pattern[:, head_idx, ...]
# Investigate attention pattern of head (9, 1)
temp_attn_pattern_hook = functools.partial(attn_pattern_hook, layer_idx=9, head_idx = 1)
logits = model.run_with_hooks(prompt_tokens, fwd_hooks = [(utils.get_act_name("pattern", 9), temp_attn_pattern_hook)], prepend_bos=True,)
# %%
#