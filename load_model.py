# %%
from safetensors import safe_open
from pathlib import Path

# %%
path = Path("/Data/lm_models/gpt2")
with safe_open(path / "model.safetensors", framework="flax",  device="cpu") as f:
    for key in f.keys():
        t = f.get_tensor(key)
        print(key, t.shape)

print(f.get_tensor("h.3.attn.bias"))
print(type(t))
