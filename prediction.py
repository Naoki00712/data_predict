import pandas as pd
from pycaret.classification import *

saved_model = load_model('/Users/naokimatsumoto/Desktop/pyc/hanshin')

data_unseen = pd.read_csv('/Users/naokimatsumoto/Desktop/pyc/data_list_predict.csv')
data_unseen[['penalty', 'favorite']] = data_unseen[['penalty', 'favorite']].astype('int')

result = predict_model(saved_model, data = data_unseen)

result_d = result.loc[:, ['Label', 'Score']]
data_unseen_d = data_unseen.drop(columns = 'Unnamed: 0')
merge = pd.merge(result_d, data_unseen_d, right_index=True, left_index=True)
merge

path = '/Users/naokimatsumoto/Desktop/pyc/prediction.json'

merge.to_json(path)

print(merge)