import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d

TEXT_AMOUNT = '100M'
STEPS_LIMIT = 7500 if TEXT_AMOUNT == '10M' else 12600


def load_data(words: str, steps_limit: int, vision_perc: int):
    text_perc = '10' if words == '100M' else '1'
    headers = ['trainer/global_step', f'Group: text{text_perc}-vision{vision_perc} - evaluation/blimp_average']
    df = pd.read_csv(f'{words}/blimp.csv', usecols=headers)
    headers2 = ['trainer/global_step', f'Group: text{text_perc}-vision{vision_perc} - evaluation/pseudo_perplexity']
    df2 = pd.read_csv(f'{words}/pppl.csv', usecols=headers2)

    join_df = df.join(df2.set_index('trainer/global_step'), on='trainer/global_step')
    join_df = join_df[join_df[headers[0]] <= steps_limit]
    steps = [x / 1000 for x in pd.to_numeric(join_df[headers[0]]).values.tolist()]
    blimp = pd.to_numeric(join_df[headers[1]]).values.tolist()
    ppl = pd.to_numeric(join_df[headers2[1]]).values.tolist()
    return steps, blimp, ppl


def plot_for_variation(words: str, vision_perc: int):
    steps, blimp, ppl = load_data(words, steps_limit=STEPS_LIMIT, vision_perc=vision_perc)
    label_map = {
        0: 'No Vision',
        1: ("Slight" if words == "100M" else "Full") + ' Vision',
        10: ("Full" if words == "100M" else "Extra (400K)") + ' Vision',
        100: 'Extra (4M) Vision'
    }

    print(f"Max BLiMP score for {words} words and {label_map[vision_perc]}: {round(max(blimp), 2)}")
    print(f"Min PPPL for {words} words and {label_map[vision_perc]}: {round(min(ppl), 2)}")

    def helper(x, y):
        f = lambda z: z.nonzero()[0]

        for i in range(len(y) - 1, -1, -1):
            if math.isnan(y[i]):
                x.pop()
                y.pop()
            else:
                break
        y = np.array(y)
        nans = np.isnan(y)
        y[nans] = np.interp(f(nans), f(~nans), y[~nans])
        return x, y

    steps2, blimp = helper(steps.copy(), blimp)
    smooth_blimp = gaussian_filter1d(blimp, sigma=0.5)
    ax.plot(steps2, smooth_blimp, "-o", color=color_map[vision_perc], markersize=0, label=label_map[vision_perc])

    steps3, ppl = helper(steps.copy(), ppl)
    smooth_ppl = gaussian_filter1d(ppl, sigma=0.5)
    ax2.plot(steps3, smooth_ppl, "--o", color=color_map[vision_perc], markersize=0, label=label_map[vision_perc])


fig, ax = plt.subplots(figsize=(5, 5))

# General structure
fig.suptitle(f'Pretraining Performance ({TEXT_AMOUNT} words)')
ax2 = ax.twinx()
ax.grid(True)  # , axis='x')
ax.set_xticks(np.arange(0, STEPS_LIMIT / 1000 + 0.5, 1))
ax.set_xlabel("Training Steps (thousands)", fontsize=12)
ax.set_ylabel("BLiMP Score (%)", fontsize=12)

# Extra info
EXTRA_INFO_COLOR = 'black'  # '#326e43'  # dark green
ax.axhline(y=86.01, color=EXTRA_INFO_COLOR, linestyle='-')
if TEXT_AMOUNT == '100M':
    ax.annotate("86.01% on PMD corpus (Singh et al., 2021)", xy=(0, 86.01), xytext=(2., 86.01 + 0.5),
                color=EXTRA_INFO_COLOR)
    top_score = 65.1
    ax.axhline(y=top_score, color=EXTRA_INFO_COLOR, linestyle='-')
    ax.annotate(f"{top_score}% on 10M words (Section 5.2)", xy=(0, top_score), xytext=(4.5, top_score + 0.5),
                color=EXTRA_INFO_COLOR)
    color_map = {
        0: '#7D54B2',  # purple
        1: '#EDB732',  # yellow
        10: '#E57439',  # orange
        100: '#5387DD',  # blue
    }
else:
    ax.annotate("86.01% on PMD corpus (Singh et al., 2021)", xy=(0, 86.01), xytext=(0.5, 86.01 + 0.5),
                color=EXTRA_INFO_COLOR)
    top_score = 72.48
    ax.axhline(y=top_score, color=EXTRA_INFO_COLOR, linestyle='-')
    ax.annotate(f"{top_score}% on 100M words (Section 5.1)", xy=(0, top_score), xytext=(1.5, top_score + 0.5),
                color=EXTRA_INFO_COLOR)
    color_map = {
        0: '#A12864',  # maroon
        1: '#229487',  # forest
        10: '#E87B9F',  # pink
        100: '#A46750',  # brown
    }

plot_for_variation(TEXT_AMOUNT, vision_perc=0)
plot_for_variation(TEXT_AMOUNT, vision_perc=1)
plot_for_variation(TEXT_AMOUNT, vision_perc=10)
plot_for_variation(TEXT_AMOUNT, vision_perc=100)
ax2.set_ylabel("Pseudo-Perplexity", fontsize=14)
ax2.set_yscale('log', base=10)

# Legend
handles, labels = ax.get_legend_handles_labels()
if TEXT_AMOUNT == '100M':
    labels.append('PPPL')
    handles.append(Line2D([0], [0], color='black', linewidth=2, linestyle='dotted'))
    labels.append('BLiMP Score')
    handles.append(Line2D([0], [0], color='black', linewidth=2))
plt.legend(handles=handles, labels=labels, bbox_to_anchor=[.6, .95], loc='upper right')

plt.show()
fig.savefig(f'{TEXT_AMOUNT}/{TEXT_AMOUNT}.pdf', format='pdf', dpi=500, bbox_inches='tight')
