import os
import numpy as np
import json
import sys
import subprocess 
from azureml.core.model import Model
#from azureml.core.model import Model

# implement pip as a subprocess this is required a my project failed joblib:
#subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'joblib'])
#subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pandas'])
import joblib
import pandas as pd

def init():
    global model

    model_path = Model.get_model_path(model_name='Maternal_Health_Risk_AutoML_model')
    model = joblib.load(model_path)


def run(data):
    try:
        #data = np.array(json.loads(data))
        print("In run data try block")
        data = pd.DataFrame(json.loads(data)['data'])
        result = model.predict(data)
        return result.tolist()
    except Exception as err:
        print(str(err))
        return str(err)
       
