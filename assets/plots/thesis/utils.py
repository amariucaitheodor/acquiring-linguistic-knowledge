import math
from itertools import groupby

import numpy as np
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


def underline(d: float, times: int) -> str:
    d = round(d, 2)
    if times == 0:
        return str(d)
    elif times == 1:
        return r'\underline{' + str(d) + '}'
    else:
        return r'\underline{' + underline(d, times - 1) + '}'


def plot(df, fig, max_cols, index, use_deltas: bool, title: str, num: int, plot_type: str = 'heat',
         palette_cmap=None, change_text_vol_labels: bool = False, range: tuple = (40, 100)):
    def remove_default_x_labels(ax):
        ax.set_xticklabels([''] * len(ax.get_xticklabels()))
        ax.set_xlabel('')

    ax = fig.add_subplot(math.ceil(num / max_cols), max_cols, index)
    if plot_type == 'bar':
        df.plot(kind='bar', stacked=False, ax=ax, legend=False, ylim=(range[0], range[1]))
        for container in ax.containers:
            ax.bar_label(container, fontsize=7)
        remove_default_x_labels(ax)
        label_group_bar_table(ax, df)
        ax.grid(axis='y')
    elif plot_type == 'heat':
        plot_df = df.droplevel(0).transpose()
        if use_deltas:
            cmap = palette_cmap
            cmap.set_over(cmap(0.5))
        else:
            cmap = sns.color_palette("viridis", as_cmap=True) if not palette_cmap else palette_cmap
        one_line_limit = 0.6
        if title == 'top-1':
            two_lines_limit = 2
        else:
            two_lines_limit = 6
        is_multimodal_retrieval_plot = 'Multimodal' in fig._suptitle.get_text()
        heatmap = sns.heatmap(plot_df, ax=ax, fmt='' if is_multimodal_retrieval_plot else '.2f',
                              vmin=range[0] if not use_deltas else -8,
                              vmax=range[1] if not use_deltas else 8,
                              cbar=index % max_cols == 0, cmap=cmap,
                              yticklabels=index % max_cols == 1,
                              annot=np.array(
                                  [underline(data, 2 if data > two_lines_limit else (1 if data > one_line_limit else 0))
                                   for data in plot_df.values.ravel()]).reshape(
                                  np.shape(plot_df)) if is_multimodal_retrieval_plot else True,
                              )

        if use_deltas:
            ax.add_patch(Rectangle((0, 0), 2, 1, fill=False, edgecolor='black', lw=1, clip_on=False))

        if change_text_vol_labels:
            remove_default_x_labels(ax)
            ax.set_xticklabels(['100M', '10M'], rotation=0, rotation_mode='anchor')
        ax.set_title(title, fontsize=14 if is_multimodal_retrieval_plot else 10)
        ax.invert_yaxis()
        if not is_multimodal_retrieval_plot:
            ax.invert_xaxis()
        else:
            heatmap.set_yticklabels(f"{x._text}\%" for x in heatmap.get_yticklabels())
            heatmap.set_xticklabels(f"{x._text}\%" for x in heatmap.get_xticklabels())
            if index % max_cols != 1:
                ax.set_ylabel('')
    else:
        raise ValueError(f'Unknown plot type: {plot_type}')
    return ax
