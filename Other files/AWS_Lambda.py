import s3fs
import joblib

model_open = open('s3://trabalho-ifsp-campinas-interdisciplinar-2022-2/random_forest_model.pkl', 'rb')
model_open.close()

print('ok')

model = open('s3://trabalho-ifsp-campinas-interdisciplinar-2022-2/random_forest_model.pkl', 'b')
loaded_forest_model = joblib.loadmodel)

loaded_data = joblib.load('s3://trabalho-ifsp-campinas-interdisciplinar-2022-2/test_data_lambda.pkl')

prediction = loaded_forest_model.predict(loaded_data)

joblib.dump(prediction,'s3://trabalho-ifsp-campinas-interdisciplinar-2022-2/prediction.pkl')