from itertools import groupby

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
