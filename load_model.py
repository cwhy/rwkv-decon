# %%
from safetensors import safe_open
from pathlib import Path

# %%
path = Path("/Data/lm_models/gpt2")
with safe_open(path / "model.safetensors", framework="flax",  device="cpu") as f:
    for key in f.keys():
        print(key)
    t = f.get_tensor("wpe.weight")

print(t)
print(type(t))
print(t.shape)
