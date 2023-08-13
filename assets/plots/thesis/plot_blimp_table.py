import math
from collections import defaultdict
from math import ceil

import pandas as pd
from matplotlib import pyplot as plt

import seaborn as sns

from assets.plots.thesis.utils import BLIMP_CATEGORIES, get_label_map, label_group_bar_table, get_vision_types

MODEL_TYPE = 'full_sized'
PLOT_TYPE = 'heat'  # or bar
DELTAS = True
Y_MIN = 40


def load_blimp_statistic(text_perc: int, vision_perc: int, category: str, statistic: str, steps_limit=None) -> float:
    headers = ['Step', f'Group: text{text_perc}-vision{vision_perc} - evaluation/blimp/{category}']
    df = pd.read_csv(f'{MODEL_TYPE}/{category}.csv', usecols=headers)
    if steps_limit:
        df = df[df[headers[0]] <= steps_limit]
    blimp_values = pd.to_numeric(df[headers[1]]).values.tolist()
    blimp_values = [x for x in blimp_values if not math.isnan(x)]
    if statistic == 'max':
        return round(max(blimp_values), 2)
    elif statistic == 'last':
        return round(blimp_values[-1], 2)
    raise ValueError(f'Unknown type: {statistic}')


def construct_table():
    plotting_dict = defaultdict(list)
    for text_perc in [1, 10]:
        for vision_perc in [0, 1, 10, 100]:
            for cat, formatted_cat in BLIMP_CATEGORIES.items():
                plotting_dict['Blimp Category'].append(formatted_cat)
                plotting_dict['Text'].append(f"{'10M' if text_perc == 1 else '100M'} words")
                for vision_type in get_vision_types():
                    if vision_type == get_label_map(text_perc, vision_perc):
                        stat = load_blimp_statistic(text_perc, vision_perc, cat, statistic_type)
                        if DELTAS and vision_perc > 0:
                            stat = stat - load_blimp_statistic(text_perc, 0, cat, statistic_type)
                        plotting_dict[vision_type].append(stat)
                    else:
                        plotting_dict[vision_type].append(math.nan)
    return pd.DataFrame(data=plotting_dict).groupby(['Blimp Category', 'Text']).sum()


def plot():
    def remove_default_x_labels(ax):
        ax.set_xticklabels([''] * len(ax.get_xticklabels()))
        ax.set_xlabel('')

    ax = fig.add_subplot(ceil(len(BLIMP_CATEGORIES.values()) / max_cols), max_cols, index)
    if PLOT_TYPE == 'bar':
        category_filtered_df.plot(kind='bar', stacked=False, ax=ax, legend=False, ylim=(Y_MIN, 100))
        for container in ax.containers:
            ax.bar_label(container, fontsize=7)
        remove_default_x_labels(ax)
        label_group_bar_table(ax, category_filtered_df)
        ax.grid(axis='y')
    elif PLOT_TYPE == 'heat':
        plot_df = category_filtered_df.droplevel(0).transpose()
        if DELTAS:
            cmap = sns.diverging_palette(145, 300, s=60, as_cmap=True)
            cmap.set_over(cmap(0.5))
        else:
            cmap = sns.color_palette("viridis", as_cmap=True)
        sns.heatmap(plot_df, ax=ax, annot=True,
                    vmin=Y_MIN if not DELTAS else -8,
                    vmax=100 if not DELTAS else 8,
                    cbar=index % max_cols == 0, cmap=cmap,
                    yticklabels=index % max_cols == 1)
        remove_default_x_labels(ax)
        ax.set_xticklabels(['100M', '10M'], rotation=0, rotation_mode='anchor')
        ax.set_title(blimp_category, fontsize=10)
        ax.invert_yaxis()
        ax.invert_xaxis()
    else:
        raise ValueError(f'Unknown plot type: {PLOT_TYPE}')
    return ax


for statistic_type in ['max', 'last']:
    fig = plt.figure(figsize=(8, 8))
    fig.suptitle(f"Influence of Vision on Linguistic Knowledge ({statistic_type} BLiMP score)")

    df = construct_table()
    i, j, max_cols = 0, 0, 3
    axes = {}
    for blimp_category in BLIMP_CATEGORIES.values():
        category_filtered_df = df[df.index.get_level_values('Blimp Category') == blimp_category]
        index = i * max_cols + j + 1
        axes[index] = plot()
        j += 1
        if j == max_cols:
            i, j = i + 1, 0

    if PLOT_TYPE == 'bar':
        handles, labels = axes[1].get_legend_handles_labels()
        fig.legend(handles, labels)
    fig.tight_layout()
    fig.savefig(f'{MODEL_TYPE}/figures/plot_{statistic_type}{"_deltas" if DELTAS else ""}.png')
    fig.savefig(f'{MODEL_TYPE}/figures/plot_{statistic_type}{"_deltas" if DELTAS else ""}.pdf')
