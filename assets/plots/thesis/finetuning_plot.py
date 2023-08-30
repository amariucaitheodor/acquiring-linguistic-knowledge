import json
import math
from collections import defaultdict

import pandas as pd
from matplotlib import pyplot as plt

import seaborn as sns

from assets.plots.thesis.utils import plot

MODEL_TYPE = 'half_sized'
LOCATION = '/cluster/work/cotterell/tamariucai/acquiring-linguistic-knowledge/alkmi/callbacks/evaluation-pipeline/results/theodor1289'
TASKS = ['qqp', 'mnli', 'mnli-mm', 'qnli', 'sst2', 'cola', 'multirc', 'mrpc', 'rte', 'boolq', 'wsc']


def construct_finetune_results_table():
    plotting_dict = defaultdict(list)
    t = 'halfsize_' if MODEL_TYPE == 'half_sized' else ""

    def add_to_dict(text_perc: int, vision_perc: int):
        model = f'thesis_{t}text{text_perc}_vision{vision_perc}'
        print(f"Adding results for {model.split(f'thesis_{t}')[1]} to the dictionary.")
        for task in TASKS:
            try:
                with open(f"{LOCATION}/{model}/finetune/{task}/eval_results.json", "rb") as f:
                    json_data = json.load(f)
                    acc = round(json_data['eval_accuracy'] * 100, 2)
            except:
                print("using placeholder")  # TODO: remove placeholder once runs finish
                acc = 75.0
            plotting_dict['Task'].append(task)
            plotting_dict['TVolume'].append(f"{'10M' if text_perc == 1 else '100M'} words")
            if vision_perc == 0:
                plotting_dict['Text'].append(acc)
                plotting_dict['VL'].append(math.nan)
            else:
                no_vision_model = f'thesis_{t}text{text_perc}_vision0'
                with open(f"{LOCATION}/{no_vision_model}/finetune/{task}/eval_results.json", "rb") as f:
                    json_data = json.load(f)
                    acc -= round(json_data['eval_accuracy'] * 100, 2)
                plotting_dict['VL'].append(acc)
                plotting_dict['Text'].append(math.nan)

    for text_perc in [1, 10]:
        add_to_dict(text_perc, 0)
        add_to_dict(text_perc, text_perc)
    return pd.DataFrame.from_dict(data=plotting_dict)


fig = plt.figure(figsize=(5, 5))
fig.suptitle(f"Influence of Vision on Linguistic Knowledge - (Super)GLUE scores")

df = construct_finetune_results_table()
df = df.groupby(['Task', 'TVolume']).sum()
i, j, max_cols = 0, 0, 3
axes = {}
for k, glue_task in enumerate(TASKS):
    if k == 8:  # GLUE and SuperGLUE separation
        j += 1
        if j == max_cols:
            i, j = i + 1, 0
    task_filtered_df = df[df.index.get_level_values('Task') == glue_task]
    index = i * max_cols + j + 1
    axes[index] = plot(task_filtered_df, fig, max_cols, index, True, glue_task.upper(),
                       len(TASKS), 'heat',
                       sns.diverging_palette(220, 20, s=60, as_cmap=True))
    j += 1
    if j == max_cols:
        i, j = i + 1, 0

fig.tight_layout()
fig.savefig(f'{MODEL_TYPE}/heatmaps/finetune/finetune_heatmap.png')
fig.savefig(f'{MODEL_TYPE}/heatmaps/finetune/finetune_heatmap.pdf')
