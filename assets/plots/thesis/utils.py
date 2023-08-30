import math
from collections import defaultdict
from itertools import groupby

import pandas as pd
from matplotlib.patches import Rectangle
import seaborn as sns

BLIMP_CATEGORIES = {
    'anaphor_agreement': 'Anaphor Agreement',
    'argument_structure': 'Argument Structure',
    'binding': 'Binding',
    'control_raising': 'Control Raising',
    'determiner_noun_agreement': 'Determiner-Noun Agreement',
    'ellipsis': 'Ellipsis',
    'filler_gap': 'Filler Gap',
    'irregular_forms': 'Irregular Forms',
    'island_effects': 'Island Effects',
    'npi_licensing': 'NPI Licensing',
    'quantifiers': 'Quantifiers',
    'subject_verb_agreement': 'Subject Verb Agreement',
}


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


def get_vision_types(old: bool = False):
    if old:
        return ['No Vision', 'Slight Vision', 'Full Vision', 'Extra (400K) Vision', 'Extra (4M) Vision']
    return ['No Images', '40K Images', '400K Images', '4M Images']


def vision_type_to_float(vision_type: str) -> float:
    number = vision_type.split(' ')[0]
    if number == 'No':
        return 0.
    elif number == '40K':
        return 40_000.
    elif number == '400K':
        return 400_000.
    elif number == '4M':
        return 4_000_000.
    else:
        raise ValueError(f'Unknown vision type: {vision_type}')


def text_type_to_float(text_type: str) -> float:
    if text_type == '10M words':
        return 10_000_000.
    elif text_type == '100M words':
        return 100_000_000.
    elif text_type == '1B words':
        return 1_000_000_000.
    else:
        return math.nan


def get_label_map(text_perc: int, vision_perc: int, old: bool = False):
    if vision_perc == 0:
        return 'No Vision' if old else 'No Images'
    elif vision_perc == 1:
        return (f'{"Slight" if text_perc == 10 else "Full"} Vision') if old else '40K Images'
    elif vision_perc == 10:
        return (f'{"Full" if text_perc == 10 else "Extra (400K)"} Vision') if old else '400K Images'
    elif vision_perc == 100:
        return 'Extra (4M) Vision' if old else '4M Images'


def label_group_bar_table(ax, df):
    def label_len(my_index, level):
        labels = my_index.get_level_values(level)
        return [(k, sum(1 for _ in g)) for k, g in groupby(labels)]

    ypos = -.1
    scale = 1. / df.index.size
    for level in range(df.index.nlevels)[::-1]:
        pos = 0
        for label, rpos in label_len(df.index, level):
            lxpos = (pos + .5 * rpos) * scale
            ax.text(lxpos, ypos, label, ha='center', transform=ax.transAxes)
            pos += rpos
        ypos -= .1


def plot(df, fig, max_cols, index, use_deltas: bool, title: str, num: int, plot_type: str = 'heat',
         diverging_palette_cmap=None):
    y_min = 40

    def remove_default_x_labels(ax):
        ax.set_xticklabels([''] * len(ax.get_xticklabels()))
        ax.set_xlabel('')

    ax = fig.add_subplot(math.ceil(num / max_cols), max_cols, index)
    if plot_type == 'bar':
        df.plot(kind='bar', stacked=False, ax=ax, legend=False, ylim=(y_min, 100))
        for container in ax.containers:
            ax.bar_label(container, fontsize=7)
        remove_default_x_labels(ax)
        label_group_bar_table(ax, df)
        ax.grid(axis='y')
    elif plot_type == 'heat':
        plot_df = df.droplevel(0).transpose()
        if use_deltas:
            cmap = diverging_palette_cmap
            cmap.set_over(cmap(0.5))
        else:
            cmap = sns.color_palette("viridis", as_cmap=True)
        sns.heatmap(plot_df, ax=ax, annot=True, fmt='.2f',
                    vmin=y_min if not use_deltas else -8,
                    vmax=100 if not use_deltas else 8,
                    cbar=index % max_cols == 0, cmap=cmap,
                    yticklabels=index % max_cols == 1)

        ax.add_patch(Rectangle((0, 0), 2, 1, fill=False, edgecolor='black', lw=1, clip_on=False))

        remove_default_x_labels(ax)
        ax.set_xticklabels(['100M', '10M'], rotation=0, rotation_mode='anchor')
        ax.set_title(title, fontsize=10)
        ax.invert_yaxis()
        ax.invert_xaxis()
    else:
        raise ValueError(f'Unknown plot type: {plot_type}')
    return ax
