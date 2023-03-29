import altair as alt
import numpy as np
import pandas as pd
import safetensors.flax
from sklearn.manifold import TSNE

alt.renderers.enable('svg')


view_vec_dict = safetensors.safe_open('saves/view_vecs_dict', 'flax')
bf_layer = view_vec_dict.get_tensor('x_before_mha')
aft_layer = view_vec_dict.get_tensor('x')
prompt = "Time flies like an arrow ; fruit flies like a banana . Time files like an"
token_list = prompt.split(" ")
token_list = [str(i).zfill(2) + "_" + token for i, token in enumerate(token_list)]

view_vec = np.stack([bf_layer, aft_layer]).reshape((-1, len(token_list), 768))

n_layers, n_tokens, n_channels = view_vec.shape
assert n_tokens == len(token_list)
all_vecs = view_vec.reshape(-1, n_channels)
# all_vecs = view_vec[:-1, :, :].reshape(-1, n_channels)
# n_layers = n_layers - 1
new = TSNE(n_components=2, learning_rate='auto',
           init='random', perplexity=3, metric='cosine').fit_transform(all_vecs)
new = np.concatenate([new, np.array([np.repeat(i, n_tokens) for i in range(n_layers)]).reshape(-1, 1),
                      np.array([np.arange(n_tokens) for i in range(n_layers)]).reshape(-1, 1)], axis=1)

source = pd.DataFrame(new, columns=['x', 'y', 'layer', 'token'])
source['token'] = source['token'].apply(lambda x: token_list[int(x)])
source['layer'] = source['layer'].apply(lambda x: int(x))
alt.Chart(source).mark_circle().encode(
    x='x',
    y='y',
    size=alt.Size('layer', legend=alt.Legend(type="symbol", symbolLimit=0), type='ordinal'),
    # rainbow
    color=alt.Color('token', scale=alt.Scale(scheme='rainbow'), legend=alt.Legend(type="symbol", symbolLimit=0)),
).properties(width=1000, height=1000).save('chart4.svg')

# TSNE without first layer
