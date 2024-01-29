#%% 
import transformer_lens
import torch as t

#%%
# Load GPT-2 Small
model = transformer_lens.HookedTransformer.from_pretrained("gpt2-small")

# %%
# Run the model and get logits and activations
logits, activations = model.run_with_cache("Hello World")

#Model testing
prompt = "There is an F.B.I."
tokens = model.to_tokens(prompt)
print(model.to_str_tokens(tokens))
output = (model(prompt))
print(output.shape)
max_output = output.argmax(-1)
print(max_output)
print(model.to_str_tokens(max_output))

# %%
