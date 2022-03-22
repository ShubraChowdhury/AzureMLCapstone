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
- Auto features                             
- enable_early_stopping= True,
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
- n_cross_validations : Is a required value.   How many cross validations to perform when user validation data is not specified. Specify validation_data to provide validation data, otherwise set n_cross_validations or validation_size to extract validation data out of the specified training data. 
- experiment_timeout_minutes : Is the time beyond which the experiment will stop 
- task : The type of task to run. Values can be 'classification', 'regression', or 'forecasting' depending on the type of automated ML problem to solve, I have selected "Classification"
- primary_metric : The metric that Automated Machine Learning will optimize for model selection.  I have selected "Accuracy"
- label_column_name : The training labels to use when fitting pipelines during an experiment.
-  ebable_early_stopping : Allowing AutoMl experiment to stop early when conditions are met
- enable_onnx_compatible_models: If set to YES this allows the model to be saved as Onnex model  

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

[AutoMLConfig](https://docs.microsoft.com/en-us/python/api/azureml-train-automl-client/azureml.train.automl.automlconfig.automlconfig?view=azure-ml-py)
The cross validation checks overfitting and for computational reasons pre-defined timeout was set to 20 Minutes  which limits number of Models that could be built.Model has Accuracy as primary metric.

I have restricted the experiment time out to 20 minutes as this was my sixth try and I was loosing the opportuniny to complete the project, in an ideal scenerio I will increase experiment time at the same time will increase number of cross validation. Model was trained on very small dataset which couldnt explore the full potential, i could have extrapolate data using multiple methods as resampling training data, Adaptive Synthetic,Synthetic Minority Over-sampling Technique SMOTE etc. 

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


In this train.py is the entry script and LogisticRegression has been use. Hyperparameter uses RandomParameterSampling as I was expecting that random sampling over the hyperparameter search space using RandomParameterSampling in our parameter sampler would  reduces computation time and still find a reasonably models when compared to GridParameterSampling methodology where all the possible values from the search space are used.

Scikit-learn Logistic Regression ( from sklearn.linear_model import LogisticRegression) , RandomParameterSampling from (from azureml.train.hyperdrive.sampling import RandomParameterSampling) was used  "--C" : choice(0.01,0.1,1) ,   "--max_iter" : choice(20,40,70,100,150), here  C as inverse regularization C = 1/λ  which had a choice between 0.01, 0.1 and 1 , max_iter is for max number of iteration which has a choice between 20 ,40,70,100 and 150.

BanditPolicy is used is an early stopping policy. It cuts more runs than a conservative policy like the MedianStoppingPolicy, hence saving the computational time significantly.
BanditPolicy reference [BanditPolicy reference](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)
For early stopping from BanditPolicy (azureml.train.hyperdrive.policy import BanditPolicy) was used , it takes evaluation_interval, slack_factor, slack_amount and delay_evaluation. 
Bandit policy is based on slack factor/slack amount and evaluation interval. Bandit ends runs when the primary metric isn't within the specified slack factor/slack amount of the most successful run.

evaluation_interval: the frequency of applying the policy. Each time the training script logs the primary metric counts as one interval. An evaluation_interval of 1 will apply the policy every time the training script reports the primary metric. An evaluation_interval of 2 will apply the policy every other time. If not specified, evaluation_interval is set to 1 by default.

```
early_termination_policy = BanditPolicy(evaluation_interval=1, slack_factor=0.1)
ps = RandomParameterSampling(
    {
"--C" : choice(0.01,0.1,1) ,     
"--max_iter" : choice(20,40,70,100,150),
}
)  

if "training" not in os.listdir():
    os.mkdir("./training")

# Setup environment for your training run
sklearn_env = Environment.from_conda_specification(name='sklearn-env', file_path='project_environment.yml')

# Create a ScriptRunConfig Object to specify the configuration details of your training job
#SKLearn(source_directory, *, compute_target=None, vm_size=None, vm_priority=None, entry_script=None, script_params=None, use_docker=True, custom_docker_image=None, image_registry_details=None, user_managed=False, conda_packages=None, pip_packages=None, conda_dependencies_file_path=None, pip_requirements_file_path=None, conda_dependencies_file=None, pip_requirements_file=None, environment_variables=None, environment_definition=None, inputs=None, shm_size=None, resume_from=None, max_run_duration_seconds=None, framework_version=None, _enable_optimized_mode=False, _disable_validation=True, _show_lint_warnings=False, _show_package_warnings=False)
skl_estimator = SKLearn(source_directory="./",entry_script='train.py',compute_target=amlcompute_cluster_name)


hyperdrive_run_config =  HyperDriveConfig(
        hyperparameter_sampling=ps,
        primary_metric_name='Accuracy',
        primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
        max_total_runs = 10, 
        max_concurrent_runs =4,
        policy=early_termination_policy,
        estimator=skl_estimator
)
```
In the HyperParameter experiment hyperparameter_sampling uses BanditPolicy for early termination, with accuracy as the primary metric,Due to limited available time I have restricted the maximum total run to 10 with at the max of 4 concurrent run which has caused low accuracy of 60.6% almost 20% less than the AutoML model.


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

Hyperdrive model provided an accuracy of 60.6%, at Regularization strength of 0.1 and at Max Iteration of 70.,Due to limited available time I have restricted the maximum total run to 10 with at the max of 4 concurrent run which has caused low accuracy of 60.6% almost 20% less than the AutoML model. Increasing the maximum run and large data volume could provide better results.
 
*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.
### Fig14: New  Compute Cluster for  Hyperdrive experiment
![image](https://user-images.githubusercontent.com/32674614/159407569-52410d8d-d9b2-4dac-955d-a6750cb4fe29.png)
### Fig15: Experiment Running
![image](https://user-images.githubusercontent.com/32674614/159407652-ab6beb91-aea7-42a7-a792-5d0b92050a59.png)
### Fig16: Experiment a run completed and a run in queue
![image](https://user-images.githubusercontent.com/32674614/159407749-dfcad436-df78-44cc-b8a6-f40bde73792c.png)
### Fig17: Experiment in progress
![image](https://user-images.githubusercontent.com/32674614/159407834-bb526404-bae9-4532-9100-7d766a4b1781.png)
### Fig18: Experiment Completed
![image](https://user-images.githubusercontent.com/32674614/159407884-d04ef9e5-0759-457a-a648-7762845d33c0.png)
### Fig19: Experiment Result
![image](https://user-images.githubusercontent.com/32674614/159407971-37977598-f68c-419e-947d-89976304c3be.png)
### Fig20: Experiment Completed
![image](https://user-images.githubusercontent.com/32674614/159408038-45daa47b-1a15-4553-bf74-e9e7d54c5bc5.png)
### Fig21: Experiment Result
![image](https://user-images.githubusercontent.com/32674614/159408157-4a92838b-8c04-402f-909b-ba381accc929.png)
### Fig22: Experiment Childrun Accuracy
![image](https://user-images.githubusercontent.com/32674614/159408213-0a5b38e6-9319-4225-ac32-1eb0abbcf05c.png)
### Fig23: Endpoint
![image](https://user-images.githubusercontent.com/32674614/159408326-90c0e811-b46b-493f-bcd2-fdba2e17b86e.png)
### Fig24.1: Endpoint
![image](https://user-images.githubusercontent.com/32674614/159408383-048c2f29-2ef2-467c-a077-8bca4b0d7eef.png)
### Fig24.2: Endpoint
![image](https://user-images.githubusercontent.com/32674614/159408421-c46c174b-3813-4deb-b763-86193d4a90c5.png)


## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

From Models trained from the above two approaches, the AutoML Experiment gave accuracy of 82% while the HyperDrive Experiment gave accuracy of 60.6%. The performance of AutoML model exceeded the HyperDrive performance by close to 20%, My work was to register AutoML model as the best model and deployed as a web service and enable Application Insights . To test it out I have actually deployed both model.

[Model Deployment Reference](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-and-where?tabs=python)
The workflow is similar no matter where you deploy your model:

- Register the model.
- Prepare an entry script.
- Prepare an inference configuration.
- Deploy the model locally to ensure everything works.
- Choose a compute target.
- Deploy the model to the cloud.
- Test the resulting web service.

Registering a model
```
automl_model_registered = remote_run.register_model(model_name='Maternal_Health_Risk_AutoML_model') 
automl_model_registered.download(target_dir="outputs", exist_ok=True)
```
Sourcing project environment
```
myenv = Environment.from_conda_specification(name="env", file_path="project_environment.yml")
```
Configuring Interface
```
inference_config = InferenceConfig(entry_script="scoreautoml.py",environment=myenv)
```
Deploy Configuration
```
deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb = 4, 
                                                       enable_app_insights=True, 
                                                       description='Maternal Health Risk AutoML model' )
model = Model(ws,'Maternal_Health_Risk_AutoML_model')
```

Creating a service and deploying the service
```
service=Model.deploy(workspace=ws,
                    name="maternal-health-risk-dep-service",
                    models=[model],
                    inference_config=inference_config,
                    deployment_config=deployment_config)

service.wait_for_deployment(show_output=True)

```
Above inference configuration and settings explain the set up of the web service that  includes the deployed model. Environment settings and scoreautoml.py script file should be passed the InferenceConfig. The deployed model was configured in Azure Container Instance(ACI) with one cpu_cores and 4 memory_gb parameters.


## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- [End to End](https://youtu.be/SeVWd5EGX50)
- [HyperParam End to End](https://youtu.be/Juap_0v5wU4)
- [AutoML](https://youtu.be/wX6XIEO47HU)
- [AutoML Model](https://youtu.be/ogPnH-2apuY)

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
