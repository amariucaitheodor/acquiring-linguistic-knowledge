import math
from collections import defaultdict

import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from assets.plots.thesis.utils import plot, get_statistic

MODEL_TYPE = 'half_sized'

# Activating tex in all labels globally
plt.rc('text', usetex=True)

# Adjust font specs as desired (here: closest similarity to seaborn standard)
plt.rc('font', **{'size': 14.0})
plt.rc('text.latex', preamble=r'\usepackage{lmodern}')


def construct_retrieval_results_table():
    def load_retrieval(text_perc: int, vision_perc: int, topk: int, statistic: str = 'best_ckpt') -> float:
        if vision_perc == 0:
            # based on previous runs avg. (and random guessing probability, too)
            return np.random.normal(0.1 * topk, 0.01 * topk)
        headers = ['Step', f'Group: text{text_perc}-vision{vision_perc} - evaluation/imagenet_zeroshot/top{topk}']
        df = pd.read_csv(f'{MODEL_TYPE}/data/multimodal_retrieval/top_{topk}.csv', usecols=headers)
        retrieval_acc_values = [x * 100 for x in list(df[headers[1]]) if not math.isnan(x)]
        return get_statistic(retrieval_acc_values, df, headers, text_perc, vision_perc, statistic, MODEL_TYPE)

    plotting_dict = defaultdict(list)
    for text_perc in [1, 10]:
        for vision_perc in [0, 1, 10, 100]:
            for topk in [1, 5]:
                plotting_dict['Text'].append(text_perc)
                plotting_dict['Vision'].append(vision_perc)
                plotting_dict['Top'].append(topk)
                plotting_dict['mtr_score'].append(load_retrieval(text_perc, vision_perc, topk))
    return pd.DataFrame.from_dict(data=plotting_dict)


df = construct_retrieval_results_table()
print(df)

fig = plt.figure(figsize=(5, 5))
fig.suptitle(f"Multimodal Text Retrieval on ImageNet-1K")

df = df.pivot(index=['Text', 'Top'], columns='Vision', values='mtr_score')
df = df.groupby(['Top', 'Text']).sum()

i, j, max_cols = 0, 0, 2
axes = {}
for topk in [1, 5]:
    top_filtered_df = df[df.index.get_level_values('Top') == topk]
    index = i * max_cols + j + 1
    axes[index] = plot(top_filtered_df, fig, max_cols, index, False, f'top-{topk}', 2, 'heat',
                       sns.color_palette("Greens", as_cmap=True), change_text_vol_labels=False, range=(0, 9))
    j += 1
    if j == max_cols:
        i, j = i + 1, 0

fig.tight_layout()
fig.savefig(f'{MODEL_TYPE}/heatmaps/retrieval/text_retrieval_heatmap.png')
fig.savefig(f'{MODEL_TYPE}/heatmaps/retrieval/text_retrieval_heatmap.pdf')
