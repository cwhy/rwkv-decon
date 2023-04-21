import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import safetensors.flax
from numpy.typing import NDArray
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE

alt.renderers.enable('svg')


def heatmap_at(mat: NDArray, display: str):
    x_len, y_len = mat.shape
    x, y = np.meshgrid(np.arange(x_len), np.arange(y_len))
    source = pd.DataFrame({
        'x': x.flatten(),
        'y': y.flatten(),
        'z': mat.flatten()
    })

    alt.Chart(source).mark_rect().encode(
        x='x:O',
        y='y:O',
        color='z:Q'
    ).save(f'{display}.html')


def heatmap(x):
    plt.figure()
    plt.imshow(x)
    plt.show()


def plot1(x):
    print(x.shape)
    plt.figure()
    plt.plot(x)
    plt.show()


# %%
# view_vec = np.load('view_vec.npy')
view_vec_dict = safetensors.safe_open('view_vec2_dict', 'flax')
view_vec1 = view_vec_dict.get_tensor('x_before_ffn')
view_vec2 = view_vec_dict.get_tensor('x_after_ffn')

plot1(view_vec1[:, 0, 0])
plot1(view_vec2[:, 0, 0])

# %%
bf_layer = view_vec_dict.get_tensor('x_before_mha')
aft_layer = view_vec_dict.get_tensor('x')
prompt = "Time flies like an arrow ; fruit flies like a banana . Time files like an"
token_list = prompt.split(" ")
token_list = [str(i).zfill(2) + "_" + token for i, token in enumerate(token_list)]
# view_vec = np.stack([bf_layer, aft_layer]).reshape((-1, len(token_list), 768))
view_vec = aft_layer

# layers, tokens, channels
plot1(view_vec[:, 0, 0])
plot1(view_vec[0, :, 0])
plot1(view_vec[0, 0, :])

# %%
# heatmap_at(np.median(view_vec, axis=-1))
n_layers, n_tokens, n_channels = view_vec.shape
corr_layers = np.array([np.corrcoef(view_vec[:, i, :]) for i in range(n_tokens)]).mean(axis=0)
heatmap_at(corr_layers, "chart1")
corr_tokens = np.array([np.corrcoef(view_vec[i, :, :]) for i in range(n_layers)]).mean(axis=0)
heatmap_at(corr_tokens, "chart2")

# %%
plot1(np.array([np.corrcoef(view_vec[:, i, :]) for i in range(n_tokens)]).mean(axis=(1, 2)))
plot1(np.array([np.corrcoef(view_vec[i, :, :]) for i in range(n_layers)]).mean(axis=(1, 2)))
plot1(np.array([np.corrcoef(view_vec[:, :, i]) for i in range(n_channels)]).mean(axis=(1, 2)))
plot1(np.array([np.corrcoef(view_vec[:, :, i].T) for i in range(n_channels)]).mean(axis=(1, 2)))

# %%
all_vecs = view_vec.reshape(-1, n_channels)
# clustering = AgglomerativeClustering(n_clusters=3).fit(all_vecs)
clustering = AgglomerativeClustering(n_clusters=10, linkage="average", metric="cosine").fit(all_vecs)
c = clustering.labels_.reshape(n_layers, n_tokens)
heatmap_at(c, "chart3")

# %%
# TSNE with first layer
view_vec_dict = safetensors.safe_open('view_vec2_dict', 'flax')
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
