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


def get_label_map(text_perc: int, vision_perc: int):
    if vision_perc == 0:
        return 'No Vision'
    elif vision_perc == 1:
        return f'{"Slight" if text_perc == 10 else "Full"} Vision'
    elif vision_perc == 10:
        return f'{"Full" if text_perc == 10 else "Extra (400K)"} Vision'
    elif vision_perc == 100:
        return 'Extra (4M) Vision'


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
