from pathlib import Path

from safetensors import safe_open

path = Path("/Data/lm_models/rwkv")
model_name = 'RWKV-4-Pile-430M-20220808-8066'
with safe_open(path / f"{model_name}.safetensors", framework="torch",  device="cpu") as f:
    for key in f.keys():
        t = f.get_tensor(key)
        print(key, t.shape)

