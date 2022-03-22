# Capstone - Udacity Machine Learning Engineer with Microsoft Azure

This capstone project uses Azure ML platform and allows two forms of model building and deployment. Model used in this project helps to predict Maternal Health Risk based on multiple factors.

## Project Set Up and Installation
This project uses Azure ML environment provided by Udacity. Some of the components are Compute Instance(which gets created by the scripts deployed by Udacity team), compute cluster (which is created as a part of the project), notebook, Experiments, AtuoML, Endpoints, Datastore, Environment etc.

## Dataset
Dataset used in this project is [Maternal Health Risk data from UCI's ML Dataset Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/00639/Maternal%20Health%20Risk%20Data%20Set.csv). Data contains the following fields
- Age
- SystolicBP
- DiastolicBP
- BS
- BodyTemp
- HeartRate

The above information is used by the models to predict "Risk Level"

### Fig-1: Shows dataset registered using URI 
![DataSetRegistered](https://user-images.githubusercontent.com/32674614/159392065-e07eaf8d-e7b4-4019-bab6-28a450cabe15.png)
### Fig-2: Sample Dataset
![Sample Dataset](https://user-images.githubusercontent.com/32674614/159392189-77e62a3a-14fa-4359-94e3-f0837b7b2d7a.png)
### Fig-2.1 Exceptions in Dataset
![image](https://user-images.githubusercontent.com/32674614/159393528-6ba99506-ffd8-456c-8e6d-293bfc6932aa.png)
### Fig-3.1: Data Profile for Age and SystolicBP
![Data Profile](https://user-images.githubusercontent.com/32674614/159392269-8e5b8259-4a52-4d34-b5b4-cbd871e34eee.png)
### Fig-3.2: Data Profile for HeartRate and Risk Level
![image](https://user-images.githubusercontent.com/32674614/159392328-8ba7b3de-0899-4448-a19e-13e9f8960092.png)
### Fig-3.3: Data Profile for BS and Body temprature
![image](https://user-images.githubusercontent.com/32674614/159392358-f64b1939-0383-4cc4-ac9f-c873fcf4de0f.png)
### Fig-3.4: Data Profile for DiastolicBP
![image](https://user-images.githubusercontent.com/32674614/159392389-4d3ea79c-3bdd-4214-bc7e-d8976a53ee0e.png)


### Overview
In this project, we will be using Azure Machine Learning Studio to create a model and  then deploy the best model. As per project guidelines I have used two approaches in this project to create a model:
- Using Azure AutoML . This experiment is named as 'capstone-automl-experiment'
- Using Azure HyperDrive. This experiment is named as 'capstone_hyperdrive_exp'
And the best model from any of the above methods will be deployed and then consumed.
Azure AutoML model is build using jupyter notebook "automl.ipynb" and  Azure HyperDrive is build using jupyter notebook "hyperparameter_tuning.ipynb"

### Task
My task involves predicting Maternal "Risk Level" based on Age, SystolicBP, DiastolicBP,BS,BodyTemp,HeartRate. Please refer above for data profile and data distribution.

### Access
*TODO*: Explain how you are accessing the data in your workspace.
As a prerequisite I had created a Compute Cluster named "automl-com-clst" , specifcaly to be used by AutoML model. It used a VM "STANDARD_D2_V2" with a maximum node of 4

Following is the code snippet used for creating compute cluster
```
amlcompute_cluster_name = "automl-com-clst" 
try:
    compute_target = ComputeTarget(workspace=ws, name=amlcompute_cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_D2_V2', max_nodes=4)
    compute_target = ComputeTarget.create(ws, amlcompute_cluster_name, compute_config)

compute_target.wait_for_completion(show_output=True, min_node_count = 1, timeout_in_minutes = 10)
```
### Fig-4 Compute Cluster Created
![image](https://user-images.githubusercontent.com/32674614/159396842-f701cc3e-a105-4a1b-8502-d6dcb4b7ada0.png)


Data is access from [Maternal Health Risk data from UCI's ML Dataset Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/00639/Maternal%20Health%20Risk%20Data%20Set.csv). Both jupyter notebook access the data in similar way. First it looks for available dataset that is registered with the name "Maternal_Health_Risk_Data_Set", if the data is not found and not registered it uses Dataset from azureml.core.dataset package to fetch the dataset and then register the same in workspace

Below is the code snippet to extract and register data 
```
found = False
key="Maternal_Health_Risk_Data_Set" 
description_text = "UCI machine Learning Maternal Health Risk Data Set"

if key in ws.datasets.keys(): 
        found = True
        dataset = ws.datasets[key] 

if not found:
        # Create AML Dataset and register it into Workspace
        example_data = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00639/Maternal%20Health%20Risk%20Data%20Set.csv'
        dataset = Dataset.Tabular.from_delimited_files(example_data)        
        #Register Dataset in Workspace
        dataset = dataset.register(workspace=ws,
                                   name=key,
                                   description=description_text)

```


## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
