import math
from collections import defaultdict

import pandas as pd
from matplotlib import pyplot as plt

import seaborn as sns

from assets.plots.thesis.utils import BLIMP_CATEGORIES, plot, get_vision_types, get_label_map

MODEL_TYPE = 'full_sized'
PLOT_TYPE = 'heat'  # or bar
DELTAS = True


def construct_blimp_results_table(model_type: str, statistic_type: str, deltas: bool):
    def load_blimp_statistic(text_perc: int, vision_perc: int, category: str, statistic: str,
                             steps_limit=None) -> float:
        headers = ['Step', f'Group: text{text_perc}-vision{vision_perc} - evaluation/blimp/{category}']
        df = pd.read_csv(f'{model_type}/{category}.csv', usecols=headers)
        if steps_limit:
            df = df[df[headers[0]] <= steps_limit]
        blimp_values = pd.to_numeric(df[headers[1]]).values.tolist()
        blimp_values = [x for x in blimp_values if not math.isnan(x)]
        if statistic == 'max':
            return round(max(blimp_values), 2)
        elif statistic == 'last':
            return round(blimp_values[-1], 2)
        raise ValueError(f'Unknown type: {statistic}')

    plotting_dict = defaultdict(list)
    for text_perc in [1, 10]:
        for vision_perc in [0, 1, 10, 100]:
            for cat, formatted_cat in BLIMP_CATEGORIES.items():
                plotting_dict['Blimp Category'].append(formatted_cat)
                plotting_dict['Text'].append(f"{'10M' if text_perc == 1 else '100M'} words")
                for vision_type in get_vision_types():
                    if vision_type == get_label_map(text_perc, vision_perc):
                        stat = load_blimp_statistic(text_perc, vision_perc, cat, statistic_type)
                        if deltas and vision_perc > 0:
                            stat = stat - load_blimp_statistic(text_perc, 0, cat, statistic_type)
                        plotting_dict[vision_type].append(stat)
                    else:
                        plotting_dict[vision_type].append(math.nan)
    return pd.DataFrame.from_dict(data=plotting_dict)


if __name__ == '__main__':
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
                               sns.diverging_palette(145, 300, s=60, as_cmap=True), True)
            j += 1
            if j == max_cols:
                i, j = i + 1, 0

        if PLOT_TYPE == 'bar':
            handles, labels = axes[1].get_legend_handles_labels()
            fig.legend(handles, labels)

        fig.tight_layout()
        fig.savefig(f'{MODEL_TYPE}/heatmaps/plot_{statistic_type}{"_deltas" if DELTAS else ""}.png')
        fig.savefig(f'{MODEL_TYPE}/heatmaps/plot_{statistic_type}{"_deltas" if DELTAS else ""}.pdf')
