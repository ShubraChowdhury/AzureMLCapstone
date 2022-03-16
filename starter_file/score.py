import os
import numpy as np
import json
import joblib

def init():
    global model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'),'Maternal_Health_Risk_AutoML_model.pkl')
    model = joblib.load(model_path)
    
    #model_path = Model.get_model_path('Maternal_Health_Risk_AutoML_model')
    #model = joblib.load(model_path)

def run(data):
    try:
        data = np.array(json.loads(data))
        result = model.predict(data)
        return result.tolist()
    except Exception as err:
        return str(err)