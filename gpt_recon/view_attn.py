import altair as alt
import numpy as np
import pandas as pd
import safetensors
from numpy.typing import NDArray


def heatmap(mat: NDArray) -> alt.Chart:
    x_len, y_len = mat.shape
    x, y = np.meshgrid(np.arange(x_len), np.arange(y_len))
    source = pd.DataFrame({
        'x': x.flatten(),
        'y': y.flatten(),
        'z': mat.flatten()
    })

    return alt.Chart(source).mark_rect().encode(
        x='x:O',
        y='y:O',
        color='z:Q'
    )


view_vec_dict = safetensors.safe_open('saves/view_vec2_dict', 'flax')
attn_result = view_vec_dict.get_tensor('attn_raw')
# head, layer, token, shape
chart = alt.vconcat().resolve_scale(
    color='independent'
)
for i in range(12):
    row = alt.hconcat().resolve_scale(color='independent')
    for j in range(12):
        row |= heatmap(attn_result[j, i, :, :])
    chart &= row
chart.save('attn.html')
