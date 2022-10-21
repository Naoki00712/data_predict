from ast import If
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
data_predict = pd.read_csv('/Users/naokimatsumoto/Desktop/pyc/data_predict_202207050211.csv')

result = predict_model(model, data = data_predict)

result_d = result.loc[:, ['horse_number', 'prediction_label', 'prediction_score']].sort_values('horse_number').reset_index(drop=True)

result_d['score'] = result_d.apply(score,axis=1)

# data_predict_d = data_predict.drop(columns = 'Unnamed: 0')

# merge = pd.merge(result_d, data_predict_d, right_index=True, left_index=True)

# path = '/Users/naokimatsumoto/Desktop/pyc/prediction.json'
# merge.to_json(path)

# print(merge)

result_d.to_csv('/Users/naokimatsumoto/Desktop/pyc/prediction.csv')

print(result_d)

# print(data_predict)