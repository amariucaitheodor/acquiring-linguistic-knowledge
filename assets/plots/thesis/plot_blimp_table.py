from math import ceil

from matplotlib import pyplot as plt

import seaborn as sns
from matplotlib.patches import Rectangle

from assets.plots.thesis.utils import BLIMP_CATEGORIES, label_group_bar_table, construct_table

MODEL_TYPE = 'full_sized'
PLOT_TYPE = 'heat'  # or bar
DELTAS = True
Y_MIN = 40


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

        ax.add_patch(Rectangle((0, 0), 2, 1, fill=False, edgecolor='black', lw=1, clip_on=False))

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

    df = construct_table(MODEL_TYPE, statistic_type, DELTAS)
    df = df.groupby(['Blimp Category', 'Text']).sum()
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
    fig.savefig(f'{MODEL_TYPE}/heatmaps/plot_{statistic_type}{"_deltas" if DELTAS else ""}.png')
    fig.savefig(f'{MODEL_TYPE}/heatmaps/plot_{statistic_type}{"_deltas" if DELTAS else ""}.pdf')
