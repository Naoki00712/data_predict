import pandas as pd
from pycaret.classification import *

def score(self):
    if self['prediction_label'] == 0:
        return 1 - self['prediction_score']
    elif self['prediction_label'] == 1:
        return self['prediction_score']
    else :
        return str("none")

model = load_model('/Users/naokimatsumoto/Desktop/pyc/2021_lr')
race_data = pd.read_csv('/Users/naokimatsumoto/Desktop/pyc/data_predict_202207050209.csv',index_col=0)

result = predict_model(model, data = race_data)

result['score'] = result.apply(score,axis=1)
score_std = result['score'].std(ddof=0)
score_mean = result['score'].mean()
result['DeviationValue'] = result['score'].map(lambda x: round((x - score_mean) / score_std * 10 + 50, 2))

merge = pd.merge(race_data, result['DeviationValue'], right_index=True, left_index=True)

merge.to_csv('/Users/naokimatsumoto/Desktop/pyc/prediction.csv')
merge.to_json('/Users/naokimatsumoto/Desktop/pyc/prediction.json')

print(merge)