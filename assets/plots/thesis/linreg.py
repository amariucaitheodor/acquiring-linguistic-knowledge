import math

from sklearn.preprocessing import RobustScaler

from assets.plots.thesis.diagnostics_linreg import LMERegDiagnostic
from assets.plots.thesis.utils import construct_table, get_vision_types, vision_type_to_float, text_type_to_float, \
    BLIMP_CATEGORIES
import statsmodels.formula.api as smf

MODEL_TYPE = 'full_sized'
statistic_type = 'max'
df = construct_table(MODEL_TYPE, statistic_type, False)

images_count = []
for index, row in df.iterrows():
    found = False
    for vision_type in get_vision_types():
        if not math.isnan(row[vision_type]):
            if found:
                raise ValueError(f'Found multiple values for {index, row}')
            images_count.append(vision_type_to_float(vision_type))
            found = True
df['Images'] = images_count

cat_to_idx = {}
for i, v in enumerate(BLIMP_CATEGORIES.values()):
    cat_to_idx[v] = i
df['Category'] = df['Blimp Category'].apply(lambda category: cat_to_idx[category]).astype('int64')
df = df.drop('Blimp Category', axis=1)

df['Words'] = df['Text'].map(lambda x: text_type_to_float(x))
df = df.drop('Text', axis=1)

df['Score'] = df[get_vision_types()].apply(lambda x: float(','.join(x.dropna().astype(str))), axis=1)
df = df.drop(get_vision_types(), axis=1)

print("============== ORIGINAL ==============")
print(df)
print(df.dtypes)

print("============== SCALED ==============")
scaler = RobustScaler()
for col in ['Words', 'Images']:  # , 'Score'
    df[[col]] = scaler.fit_transform(df[[col]])
print(df)

me_model = smf.mixedlm("Score ~ Words + Images", df, groups=df["Category"])
res_me = me_model.fit()
print(res_me.summary())
with open(f'{MODEL_TYPE}/regression/summary.txt', 'w') as fh:
    fh.write(res_me.summary().as_text())

fig, ax = LMERegDiagnostic(res_me)()
fig.savefig(f'{MODEL_TYPE}/regression/lme_diagnostics_{statistic_type}.png')
fig.savefig(f'{MODEL_TYPE}/regression/lme_diagnostics_{statistic_type}.pdf')