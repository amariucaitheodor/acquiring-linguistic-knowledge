from matplotlib import pyplot as plt

import seaborn as sns

from assets.plots.thesis.utils import BLIMP_CATEGORIES, construct_blimp_results_table, plot

MODEL_TYPE = 'full_sized'
PLOT_TYPE = 'heat'  # or bar
DELTAS = True

for statistic_type in ['max', 'last']:
    fig = plt.figure(figsize=(8, 8))
    fig.suptitle(f"Influence of Vision on Linguistic Knowledge ({statistic_type} BLiMP score)")

    df = construct_blimp_results_table(MODEL_TYPE, statistic_type, DELTAS)
    df = df.groupby(['Blimp Category', 'Text']).sum()
    i, j, max_cols = 0, 0, 3
    axes = {}
    for blimp_category in BLIMP_CATEGORIES.values():
        category_filtered_df = df[df.index.get_level_values('Blimp Category') == blimp_category]
        index = i * max_cols + j + 1
        axes[index] = plot(category_filtered_df, fig, max_cols, index, DELTAS, blimp_category,
                           len(BLIMP_CATEGORIES.values()), PLOT_TYPE,
                           sns.diverging_palette(145, 300, s=60, as_cmap=True))
        j += 1
        if j == max_cols:
            i, j = i + 1, 0

    if PLOT_TYPE == 'bar':
        handles, labels = axes[1].get_legend_handles_labels()
        fig.legend(handles, labels)

    fig.tight_layout()
    fig.savefig(f'{MODEL_TYPE}/heatmaps/plot_{statistic_type}{"_deltas" if DELTAS else ""}.png')
    fig.savefig(f'{MODEL_TYPE}/heatmaps/plot_{statistic_type}{"_deltas" if DELTAS else ""}.pdf')
