
import os
import numpy as np
import json
import sys
import subprocess 
from azureml.core.model import Model

# implement pip as a subprocess this is required a my project failed joblib:
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'joblib'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pandas'])
import joblib
import pandas as pd

def init():
    global model
    #model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'),'Maternal_Health_Risk_AutoML_model')
    #Maternal_Health_Risk_AutoML_model
    try:
        model_path = Model.get_model_path('Maternal_Health_Risk_AutoML_model')
        print("Model Path = /t",model_path)
        model = joblib.load(model_path)
    except Exception as e:
        print("Init Error :/t", str(e))
    #model_path = Model.get_model_path('Maternal_Health_Risk_AutoML_model')
    #model = joblib.load(model_path)

def run(data):
    try:
        #data = np.array(json.loads(data))
        print("In run data try block")
        data = pd.DataFrame(json.loads(data)['dat'])
        result = model.predict(data)
        return result.tolist()
    except Exception as err:
        print(str(err))
        return str(err)
       
