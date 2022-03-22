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

In this experiment I have used the following `automl` settings:
- Experiment Timeout of  20 minutes
- Maximum Concurrent Iterations of  5
- Number of Cross Validations of 3
- Primary metric is to find accuracy

In this experiment I have used the following `automl` configuration:

- Classification as the main task
- `RiskLevel` as the column name that needs to be predicted
- Enabled early stopping                             
- Auto features                             enable_early_stopping= True,
- ONNX Compatibality enabled in case i would like to save the model in oonx format     

Below is the code snippet for `automl` 
```
automl_settings = {
    "experiment_timeout_minutes": 20,
    "max_concurrent_iterations": 5,
    "n_cross_validations":3,
    "primary_metric" : 'accuracy'
}


automl_config = AutoMLConfig(compute_target=compute_target,
                             task = "classification",
                             training_data=train_data,
                             label_column_name="RiskLevel",   
                             enable_early_stopping= True,
                             featurization= 'auto',
                             debug_log = "automl_errors.log",
                             enable_onnx_compatible_models=True,
                             **automl_settings
                            )
                            
```


### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

AutoML experiment in 20 minutes has trained data on a total of 38 models , out of which the `VotingEnsemble` model is the best model with 82% accuracy.
Following are the results:
- average_precision_score_micro 0.8730812741180376
- AUC_macro 0.9226793705273387
- average_precision_score_weighted 0.8647659722703661
- weighted_accuracy 0.8226841844282767
- f1_score_weighted 0.8189540322232832
- recall_score_weighted 0.8199899776775545
- precision_score_weighted 0.819593453577947
- recall_score_micro 0.8199899776775545
- precision_score_macro 0.8228526959653496
- recall_score_macro 0.8187646372860341
- balanced_accuracy 0.8187646372860341
- log_loss 0.6138810951918391
- AUC_weighted 0.9189162547687276
- matthews_correlation 0.7239247750212834
- f1_score_macro 0.8199474069639092
- average_precision_score_macro 0.8675376617497034
- accuracy 0.8199899776775545
- precision_score_micro 0.8199899776775545
- norm_macro_recall 0.7281469559290512
- AUC_micro 0.9290838330968461
- f1_score_micro 0.8199899776775545

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.
### Fig-5: AutoML experiment created 
![image](https://user-images.githubusercontent.com/32674614/159400814-24218e63-483d-4885-9869-4c563fa13f16.png)
### Fig-6: AutoML experiment running
![image](https://user-images.githubusercontent.com/32674614/159400887-ce756000-1851-4457-bbe0-dcca5cc81adf.png)
### Fig-7: AutoML child runs in progress
![image](https://user-images.githubusercontent.com/32674614/159400958-e2c46ddc-b459-4ae9-bd31-2d71ccaa431b.png)
### Fig-8: Best AutoML model `VotingEnsemble`
![image](https://user-images.githubusercontent.com/32674614/159400994-89a43ed3-102b-42c9-8394-7df65a4dd1b2.png)
### Fig-9: Accuracy for `VotingEnsemble` 
![image](https://user-images.githubusercontent.com/32674614/159401064-4921d2ce-3e26-4bac-b9f3-c655c91f459a.png)
### Fig-10: Feature Importance with BS has the highest
![image](https://user-images.githubusercontent.com/32674614/159401125-b5f27166-c7ba-4e00-b37c-1251b81d0b16.png)
### Fig11.1: Model Metrics
![image](https://user-images.githubusercontent.com/32674614/159401273-d19e3ed6-4669-4036-93cf-26a84856d1df.png)
### Fig11.2: Model Metrics continued..
![image](https://user-images.githubusercontent.com/32674614/159401312-59fc0ca6-e397-42bb-bb04-c25022165789.png)
### Fig11.3: Model Metrics continued..
![image](https://user-images.githubusercontent.com/32674614/159401357-6bfbfbcd-d548-4441-8d75-f3c585c593b3.png)
### Fig12: Endpoint View
![image](https://user-images.githubusercontent.com/32674614/159401495-80dc16e0-a74c-4122-bb00-1d866f979bde.png)
### Fig13: `VotingEnsemble` Deployed 
![image](https://user-images.githubusercontent.com/32674614/159401734-346820f1-fced-4cc5-b85b-ed1491cb1728.png)

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
