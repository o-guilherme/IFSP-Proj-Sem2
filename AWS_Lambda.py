!pip install s3fs==2022.01.0
from  joblib import load, dump
import pandas as pd
loaded_forest_model = load('s3://trabalho-ifsp-campinas-interdisciplinar-2022-2/random_forest_model.pkl')
test_set = pd.read_csv('s3://trabalho-ifsp-campinas-interdisciplinar-2022-2/test_data.csv')

X = dataset.drop(columns='Response')

prediction = loaded_forest_model.predict(X)

joblib.dump(prediction,'s3://trabalho-ifsp-campinas-interdisciplinar-2022-2/prediction.txt')