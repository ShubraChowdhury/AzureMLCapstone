#  Udacity Capstone Project ,Machine Learning Engineer with Microsoft Azure

This capstone project uses Azure ML platform and allows two forms of model building and deployment.

In this project I will be predicting Maternal "Risk Level" based on Age, SystolicBP, DiastolicBP,BS,BodyTemp,HeartRate. In order to predict Maternal "Risk Level" I will be using the dataset available in UCI's Machine Learning repository.
In this project, we will be using Azure Machine Learning Studio to create a model and  then deploy the best model. As per project guidelines I have used two approaches in this project to create a model:
- Using Azure AutoML . This experiment is named as 'capstone-automl-experiment'
- Using Azure HyperDrive. This experiment is named as 'capstone_hyperdrive_exp'
And the best model from any of the above methods will be deployed and then consumed.
Azure AutoML model is build using jupyter notebook "automl.ipynb" and  Azure HyperDrive is build using jupyter notebook "hyperparameter_tuning.ipynb"

I have created a compute cluster for both Auto ML and Hyperdrive. Initial step is to upload Maternal Health Risk data set and register the same in Azure machine learning studio. 
For AutoML I have defined the automl settings and AutoMLConfig which uses classification as task and experiment runs for maximum of 45 minutes.Next I have submitted the experiment to automatically train the data dataset and come up with a best model and explanation. In this case AutoML has trained on 38 models and come up with an accuracy of 81.8%

As a next step I have worked on model using HyperDrive, in this case the entry script is through train.py which uses LogisticRegression with BanditPolicy used as early termination policy and RandomParameterSampling for hyperparameter sampling.Following this the experiment was submitted and an accuracy of 60.6% was achieved.

In my experiments the AutoML model produces the best results which has 21.2% higher accuracy than the Hyperdrive model. Based on the results I have deployed the best model that is the AutoML model. Deployment involves Registering the model,Prepare an entry script (scoreautoml.py) , Prepare an inference configuration (this sources in the environment defined in "project_environment.yml"),Deploy the model .

Once the AutoML model is deployed I have validated existence of  scoring_uri and swagger_uri . Once this was validated I had send a request to the web service that was deployed to tested the result. The request sent contains the following data "Age": 25,"SystolicBP":130 ,"DiastolicBP":80 ,"BS": 15,"BodyTemp": 98,"HeartRate": 86 which produced a RiskLevel of 'high risk' and this shows that the model was deployed and webservices are responding.

#### High level sequence of task involved.
##### AutoML involves the following task:
1. Import all necessary packages
2. Create a Workspace and experiment
3. Create a compute instance
4. Import and register the Maternal Health Risk dataset
5. Split the dataset in train and test
6. Define automl_settings  and AutoMLConfig settings , I am using classification as task and accuracy as the primary metrics
7. Submit the AutoMl experiment
8. Get run details , run status and best metrics
##### Hyperdrive involves the following task:
1. Import all necessary packages
2. Create a Workspace and experiment
3. Create a compute instance
4. Import and register the Maternal Health Risk dataset
5. Create entry script train.py which uses logistic regression , train and test split are included in this script, this script also cleanse data
6. Define HyperDriveConfig with accuracy as the primary metrics
7. Submit the experiment
8. Get run details , run status and best metrics
##### Following task will be after determining the best metrics
1. Register the model
2. Define InferenceConfig including entry script and sourcing environment
3. Define AciWebservice configurations
4. Deploy model  
5. Vaildate URI for score and swagger
6. Send a request and get the output 

## Project Set Up and Installation
This project uses Azure ML environment provided by Udacity. Some of the components are Compute Instance(which gets created by the scripts deployed by Udacity team), compute cluster (which is created as a part of the project), notebook, Experiments, AtuoML, Endpoints, Datastore, Environment etc.




## Dataset

### Overview
*TODO*: Explain about the data you are using and where you got it from.

Dataset used in this project is [Maternal Health Risk data from UCI's ML Dataset Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/00639/Maternal%20Health%20Risk%20Data%20Set.csv). Data contains the following fields
- Age
- SystolicBP
- DiastolicBP
- BS
- BodyTemp
- HeartRate

The above information is used by the models to predict "Risk Level"

#### Fig-1: Sample DataSet
#### ![Sample DataSet](https://user-images.githubusercontent.com/32674614/159698025-fecaf424-1e3b-4385-9509-fb74cf48b911.png)



### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.

My task involves predicting Maternal "Risk Level" based on Age, SystolicBP, DiastolicBP,BS,BodyTemp,HeartRate. 

AutoML will use Classification task and train data on Age, SystolicBP, DiastolicBP,BS,BodyTemp,HeartRate to predict "Risk Level"
Hyperdrive model will use LogisticRegression and train data on Age, SystolicBP, DiastolicBP,BS,BodyTemp,HeartRate to predict "Risk Level"

In both models accuracy is the primary metrics


### Access
*TODO*: Explain how you are accessing the data in your workspace.

Data will be accessed using Jupyter Notebook. Notebook will fetch data from the URL [Maternal Health Risk data from UCI's ML Dataset Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/00639/Maternal%20Health%20Risk%20Data%20Set.csv)

Following is the code snippet used for accessing data and registering datase
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



Data is access from [Maternal Health Risk data from UCI's ML Dataset Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/00639/Maternal%20Health%20Risk%20Data%20Set.csv). Both jupyter notebook access the data in similar way. First it looks for available dataset that is registered with the name "Maternal_Health_Risk_Data_Set", if the data is not found and not registered it uses Dataset from azureml.core.dataset package to fetch the dataset and then register the same in workspace



## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

In this experiment I have used the following `automl` settings:
- Experiment Timeout of  45 minutes
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
    "experiment_timeout_minutes": 45,
    "max_concurrent_iterations": 5,
    "n_cross_validations":3,
    "primary_metric" : 'accuracy'
}


automl_config = AutoMLConfig(compute_target=compute_target,
                             task = "classification",
                             training_data=train_data,
                             label_column_name="RiskLevel",   
                             #path = project_folder,
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

AutoML experiment in 45 minutes has trained data on a total of 38 models , out of which the `VotingEnsemble` model is the best model with 81.75% accuracy.
```
Experiment	Id	Type	Status	Details Page	Docs Page
capstone-automl-experiment	AutoML_093d98bd-0ffe-4170-a2f6-2f6941181eab	automl	Running	Link to Azure Machine Learning studio	Link to Documentation

Current status: FeaturesGeneration. Generating features for the dataset.
Current status: DatasetFeaturization. Beginning to fit featurizers and featurize the dataset.
Current status: DatasetCrossValidationSplit. Generating individually featurized CV splits.
Current status: ModelSelection. Beginning model selection.

********************************************************************************************
DATA GUARDRAILS: 

TYPE:         Class balancing detection
STATUS:       PASSED
DESCRIPTION:  Your inputs were analyzed, and all classes are balanced in your training data.
              Learn more about imbalanced data: https://aka.ms/AutomatedMLImbalancedData

********************************************************************************************

TYPE:         Missing feature values imputation
STATUS:       DONE
DESCRIPTION:  If the missing values are expected, let the run complete. Otherwise cancel the current run and use a script to customize the handling of missing feature values that may be more appropriate based on the data type and business requirement.
              Learn more about missing value imputation: https://aka.ms/AutomatedMLFeaturization
DETAILS:      
+------------------------------+------------------------------+------------------------------+
|Column name                   |Missing value count           |Imputation type               |
+==============================+==============================+==============================+
|BodyTemp                      |3                             |mean                          |
+------------------------------+------------------------------+------------------------------+

********************************************************************************************

TYPE:         High cardinality feature detection
STATUS:       PASSED
DESCRIPTION:  Your inputs were analyzed, and no high cardinality features were detected.
              Learn more about high cardinality feature handling: https://aka.ms/AutomatedMLFeaturization

********************************************************************************************

********************************************************************************************
ITER: The iteration being evaluated.
PIPELINE: A summary description of the pipeline being evaluated.
DURATION: Time taken for the current iteration.
METRIC: The result of computing score on the fitted pipeline.
BEST: The best observed score thus far.
********************************************************************************************

ITER   PIPELINE                                       DURATION            METRIC      BEST
    0   MaxAbsScaler LightGBM                          0:00:10             0.7904    0.7904
    1   MaxAbsScaler XGBoostClassifier                 0:00:09             0.7485    0.7904
    2   MaxAbsScaler ExtremeRandomTrees                0:00:10             0.6622    0.7904
    3   SparseNormalizer XGBoostClassifier             0:00:09             0.7411    0.7904
    4   MaxAbsScaler LightGBM                          0:00:09             0.7040    0.7904
    5   MaxAbsScaler LightGBM                          0:00:09             0.7065    0.7904
    6   StandardScalerWrapper XGBoostClassifier        0:00:09             0.7534    0.7904
    7   MaxAbsScaler LogisticRegression                0:00:09             0.6326    0.7904
    8   StandardScalerWrapper ExtremeRandomTrees       0:00:09             0.5585    0.7904
   10   SparseNormalizer LightGBM                      0:00:09             0.6510    0.7904
    9   StandardScalerWrapper XGBoostClassifier        0:00:08             0.7152    0.7904
   11   StandardScalerWrapper XGBoostClassifier        0:00:09             0.7583    0.7904
   12   MaxAbsScaler LogisticRegression                0:00:09             0.6276    0.7904
   13   MaxAbsScaler SGD                               0:00:08             0.6277    0.7904
   14   StandardScalerWrapper XGBoostClassifier        0:00:09             0.7152    0.7904
   15   SparseNormalizer RandomForest                  0:00:10             0.7337    0.7904
   20   TruncatedSVDWrapper RandomForest               0:00:04             0.7732    0.7904
   16   StandardScalerWrapper LogisticRegression       0:00:09             0.6338    0.7904
   17   StandardScalerWrapper RandomForest             0:00:09             0.7139    0.7904
   21   StandardScalerWrapper XGBoostClassifier        0:03:37             0.7719    0.7904
   18   StandardScalerWrapper XGBoostClassifier        0:00:09             0.7830    0.7904
   19   TruncatedSVDWrapper RandomForest               0:00:10             0.6671    0.7904
   22   SparseNormalizer XGBoostClassifier             0:00:04             0.7472    0.7904
   23   StandardScalerWrapper LightGBM                 0:00:04             0.6436    0.7904
   26   SparseNormalizer XGBoostClassifier             0:00:04             0.7300    0.7904
   24   StandardScalerWrapper LogisticRegression       0:00:04             0.6202    0.7904
   25   StandardScalerWrapper RandomForest             0:00:04             0.6522    0.7904
   27   SparseNormalizer XGBoostClassifier             0:00:04             0.7509    0.7904
   28   SparseNormalizer XGBoostClassifier             0:00:04             0.7398    0.7904
   29   MaxAbsScaler ExtremeRandomTrees                0:00:04             0.6794    0.7904
   30   TruncatedSVDWrapper LightGBM                   0:00:04             0.4685    0.7904
   31   StandardScalerWrapper XGBoostClassifier        0:00:04             0.6892    0.7904
   32   MaxAbsScaler RandomForest                      0:00:04             0.6436    0.7904
   33   StandardScalerWrapper XGBoostClassifier        0:00:04             0.7053    0.7904
   34                                                  0:00:00                nan    0.7904
   35                                                  0:00:00                nan    0.7904
   36                                                  0:00:00                nan    0.7904
   37                                                  0:00:00                nan    0.7904
   38    VotingEnsemble                                0:00:04             0.8175    0.8175
{'runId': 'AutoML_093d98bd-0ffe-4170-a2f6-2f6941181eab',
 'target': 'automl-com-clst',
 'status': 'Completed',
 'startTimeUtc': '2022-03-22T18:33:14.704513Z',
 'endTimeUtc': '2022-03-22T18:57:29.706139Z',
 'services': {},
 'warnings': [{'source': 'JasmineService',
   'message': 'No scores improved over last 20 iterations, so experiment stopped early. This early stopping behavior can be disabled by setting enable_early_stopping = False in AutoMLConfig for notebook/python SDK runs.'}],
 'properties': {'num_iterations': '1000',
  'training_type': 'TrainFull',
  'acquisition_function': 'EI',
  'primary_metric': 'accuracy',
  'train_split': '0',
  'acquisition_parameter': '0',
  'num_cross_validation': '3',
  'target': 'automl-com-clst',
  'AMLSettingsJsonString': '{"path":null,"name":"capstone-automl-experiment","subscription_id":"6971f5ac-8af1-446e-8034-05acea24681f","resource_group":"aml-quickstarts-189674","workspace_name":"quick-starts-ws-189674","region":"southcentralus","compute_target":"automl-com-clst","spark_service":null,"azure_service":"remote","many_models":false,"pipeline_fetch_max_batch_size":1,"enable_batch_run":true,"enable_run_restructure":false,"start_auxiliary_runs_before_parent_complete":false,"enable_code_generation":false,"iterations":1000,"primary_metric":"accuracy","task_type":"classification","positive_label":null,"data_script":null,"test_size":0.0,"test_include_predictions_only":false,"validation_size":0.0,"n_cross_validations":3,"y_min":null,"y_max":null,"num_classes":null,"featurization":"auto","_ignore_package_version_incompatibilities":false,"is_timeseries":false,"max_cores_per_iteration":1,"max_concurrent_iterations":5,"iteration_timeout_minutes":null,"mem_in_mb":null,"enforce_time_on_windows":false,"experiment_timeout_minutes":45,"experiment_exit_score":null,"whitelist_models":null,"blacklist_algos":["TensorFlowLinearClassifier","TensorFlowDNN"],"supported_models":["SVM","MultinomialNaiveBayes","RandomForest","TensorFlowDNN","TensorFlowLinearClassifier","BernoulliNaiveBayes","LogisticRegression","LinearSVM","SGD","TabnetClassifier","DecisionTree","KNN","ExtremeRandomTrees","XGBoostClassifier","LightGBM","AveragedPerceptronClassifier","GradientBoosting"],"private_models":[],"auto_blacklist":true,"blacklist_samples_reached":false,"exclude_nan_labels":true,"verbosity":20,"_debug_log":"azureml_automl.log","show_warnings":false,"model_explainability":true,"service_url":null,"sdk_url":null,"sdk_packages":null,"enable_onnx_compatible_models":true,"enable_split_onnx_featurizer_estimator_models":false,"vm_type":"STANDARD_D2_V2","telemetry_verbosity":20,"send_telemetry":true,"enable_dnn":false,"scenario":"SDK-1.13.0","environment_label":null,"save_mlflow":false,"enable_categorical_indicators":false,"force_text_dnn":false,"enable_feature_sweeping":false,"enable_early_stopping":true,"early_stopping_n_iters":10,"arguments":null,"dataset_id":"b7c7f4cc-a689-4304-8487-1b737d69e2c5","hyperdrive_config":null,"validation_dataset_id":null,"run_source":null,"metrics":null,"enable_metric_confidence":false,"enable_ensembling":true,"enable_stack_ensembling":false,"ensemble_iterations":15,"enable_tf":false,"enable_subsampling":null,"subsample_seed":null,"enable_nimbusml":false,"enable_streaming":false,"force_streaming":false,"track_child_runs":true,"allowed_private_models":[],"label_column_name":"RiskLevel","weight_column_name":null,"cv_split_column_names":null,"enable_local_managed":false,"_local_managed_run_id":null,"cost_mode":1,"lag_length":0,"metric_operation":"maximize","preprocess":true}',
  'DataPrepJsonString': '{\\"training_data\\": {\\"datasetId\\": \\"b7c7f4cc-a689-4304-8487-1b737d69e2c5\\"}, \\"datasets\\": 0}',
  'EnableSubsampling': None,
  'runTemplate': 'AutoML',
  'azureml.runsource': 'automl',
  'display_task_type': 'classification',
  'dependencies_versions': '{"azureml-widgets": "1.38.0", "azureml-train": "1.38.0", "azureml-train-restclients-hyperdrive": "1.38.0", "azureml-train-core": "1.38.0", "azureml-train-automl": "1.38.0", "azureml-train-automl-runtime": "1.38.0", "azureml-train-automl-client": "1.38.0", "azureml-tensorboard": "1.38.0", "azureml-telemetry": "1.38.0", "azureml-sdk": "1.38.0", "azureml-responsibleai": "1.38.0", "azureml-pipeline": "1.38.0", "azureml-pipeline-steps": "1.38.0", "azureml-pipeline-core": "1.38.0", "azureml-opendatasets": "1.38.0", "azureml-mlflow": "1.38.0", "azureml-interpret": "1.38.0", "azureml-inference-server-http": "0.4.2", "azureml-explain-model": "1.38.0", "azureml-defaults": "1.38.0", "azureml-dataset-runtime": "1.38.0", "azureml-dataprep": "2.26.0", "azureml-dataprep-rslex": "2.2.0", "azureml-dataprep-native": "38.0.0", "azureml-datadrift": "1.38.0", "azureml-core": "1.38.0", "azureml-contrib-services": "1.38.0", "azureml-contrib-server": "1.38.0", "azureml-contrib-reinforcementlearning": "1.38.0", "azureml-contrib-pipeline-steps": "1.38.0", "azureml-contrib-notebook": "1.38.0", "azureml-contrib-fairness": "1.38.0", "azureml-contrib-dataset": "1.38.0", "azureml-contrib-automl-pipeline-steps": "1.38.0", "azureml-cli-common": "1.38.0", "azureml-automl-runtime": "1.38.0", "azureml-automl-dnn-nlp": "1.38.0", "azureml-automl-core": "1.38.0", "azureml-accel-models": "1.38.0"}',
  '_aml_system_scenario_identification': 'Remote.Parent',
  'ClientType': 'SDK',
  'environment_cpu_name': 'AzureML-AutoML',
  'environment_cpu_label': 'py36',
  'environment_gpu_name': 'AzureML-AutoML-GPU',
  'environment_gpu_label': 'py36',
  'root_attribution': 'automl',
  'attribution': 'AutoML',
  'Orchestrator': 'AutoML',
  'CancelUri': 'https://southcentralus.api.azureml.ms/jasmine/v1.0/subscriptions/6971f5ac-8af1-446e-8034-05acea24681f/resourceGroups/aml-quickstarts-189674/providers/Microsoft.MachineLearningServices/workspaces/quick-starts-ws-189674/experimentids/4cb5e02a-81a8-406d-8de0-149de903dd72/cancel/AutoML_093d98bd-0ffe-4170-a2f6-2f6941181eab',
  'ClientSdkVersion': '1.38.0',
  'snapshotId': '00000000-0000-0000-0000-000000000000',
  'SetupRunId': 'AutoML_093d98bd-0ffe-4170-a2f6-2f6941181eab_setup',
  'SetupRunContainerId': 'dcid.AutoML_093d98bd-0ffe-4170-a2f6-2f6941181eab_setup',
  'FeaturizationRunJsonPath': 'featurizer_container.json',
  'FeaturizationRunId': 'AutoML_093d98bd-0ffe-4170-a2f6-2f6941181eab_featurize',
  'ProblemInfoJsonString': '{"dataset_num_categorical": 0, "is_sparse": true, "subsampling": false, "has_extra_col": true, "dataset_classes": 3, "dataset_features": 54, "dataset_samples": 811, "single_frequency_class_detected": false}',
  'ModelExplainRunId': 'AutoML_093d98bd-0ffe-4170-a2f6-2f6941181eab_ModelExplain'},
 'inputDatasets': [{'dataset': {'id': 'b7c7f4cc-a689-4304-8487-1b737d69e2c5'}, 'consumptionDetails': {'type': 'RunInput', 'inputName': 'training_data', 'mechanism': 'Direct'}}],
 'outputDatasets': [],
 'logFiles': {},
 'submittedBy': 'ODL_User 189674'}

```


Following are the results:
```
===== AutoMl Best Run ========
Run(Experiment: capstone-automl-experiment,
Id: AutoML_093d98bd-0ffe-4170-a2f6-2f6941181eab_38,
Type: azureml.scriptrun,
Status: Completed)

  ===== AutoMl Best Fitted Model ========
Pipeline(memory=None,
         steps=[('datatransformer',
                 DataTransformer(enable_dnn=False, enable_feature_sweeping=False, feature_sweeping_config={}, feature_sweeping_timeout=86400, featurization_config=None, force_text_dnn=False, is_cross_validation=True, is_onnx_compatible=True, observer=None, task='classification', working_dir='/mnt/batch/tasks/shared/LS_root/mount...
    gpu_training_param_dict={'processing_unit_type': 'cpu'}
), random_state=0, reg_alpha=0.3125, reg_lambda=0.4166666666666667, subsample=0.5, tree_method='auto'))], verbose=False))], flatten_transform=None, weights=[0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.07142857142857142, 0.07142857142857142, 0.07142857142857142, 0.07142857142857142, 0.14285714285714285, 0.14285714285714285]))],
         verbose=False)
Y_transformer(['LabelEncoder', LabelEncoder()])
```
- f1_score_weighted 0.816520723030668
- f1_score_macro 0.8181120773643548
- precision_score_weighted 0.8176578649383873
- average_precision_score_micro 0.8745382040360331
- norm_macro_recall 0.7246959271503628
- AUC_macro 0.923683144961295
- recall_score_weighted 0.8175162862739739
- balanced_accuracy 0.8164639514335752
- accuracy 0.8175162862739739
- AUC_weighted 0.9200515581213738
- precision_score_macro 0.8219195413392303
- matthews_correlation 0.7203152947685713
- recall_score_macro 0.8164639514335752
- weighted_accuracy 0.8199937719822659
- recall_score_micro 0.8175162862739739
- average_precision_score_weighted 0.8673470748250652
- f1_score_micro 0.8175162862739739
- log_loss 0.6331436494353994
- AUC_micro 0.929468459815268
- average_precision_score_macro 0.8697373123979363
- precision_score_micro 0.8175162862739739

Fitted Model Steps
```
[('datatransformer', DataTransformer(
    task='classification',
    is_onnx_compatible=True,
    enable_feature_sweeping=False,
    enable_dnn=False,
    force_text_dnn=False,
    feature_sweeping_timeout=86400,
    featurization_config=None,
    is_cross_validation=True,
    feature_sweeping_config={}
)), ('prefittedsoftvotingclassifier', PreFittedSoftVotingClassifier(
    estimators=[('0', Pipeline(
        memory=None,
        steps=[('maxabsscaler', MaxAbsScaler(
            copy=True
        )), ('lightgbmclassifier', LightGBMClassifier(
            min_data_in_leaf=20,
            random_state=None,
            n_jobs=1,
            problem_info=ProblemInfo(
                gpu_training_param_dict={'processing_unit_type': 'cpu'}
            )
        ))],
        verbose=False
    )), ('18', Pipeline(
        memory=None,
        steps=[('standardscalerwrapper', StandardScalerWrapper(
            copy=True,
            with_mean=False,
            with_std=False
        )), ('xgboostclassifier', XGBoostClassifier(
            random_state=0,
            n_jobs=1,
            problem_info=ProblemInfo(
                gpu_training_param_dict={'processing_unit_type': 'cpu'}
            ),
            booster='gbtree',
            colsample_bytree=0.7,
            eta=0.1,
            gamma=0.1,
            max_depth=9,
            max_leaves=511,
            n_estimators=25,
            objective='reg:logistic',
            reg_alpha=0,
            reg_lambda=1.7708333333333335,
            subsample=0.9,
            tree_method='auto'
        ))],
        verbose=False
    )), ('20', Pipeline(
        memory=None,
        steps=[('truncatedsvdwrapper', TruncatedSVDWrapper(
            n_components=0.7026315789473684,
            random_state=None
        )), ('randomforestclassifier', RandomForestClassifier(
            bootstrap=False,
            ccp_alpha=0.0,
            class_weight='balanced',
            criterion='gini',
            max_depth=None,
            max_features='log2',
            max_leaf_nodes=None,
            max_samples=None,
            min_impurity_decrease=0.0,
            min_impurity_split=None,
            min_samples_leaf=0.01,
            min_samples_split=0.01,
            min_weight_fraction_leaf=0.0,
            n_estimators=200,
            n_jobs=1,
            oob_score=False,
            random_state=None,
            verbose=0,
            warm_start=False
        ))],
        verbose=False
    )), ('21', Pipeline(
        memory=None,
        steps=[('standardscalerwrapper', StandardScalerWrapper(
            copy=True,
            with_mean=False,
            with_std=False
        )), ('xgboostclassifier', XGBoostClassifier(
            random_state=0,
            n_jobs=1,
            problem_info=ProblemInfo(
                gpu_training_param_dict={'processing_unit_type': 'cpu'}
            ),
            booster='gbtree',
            colsample_bytree=0.5,
            eta=0.2,
            gamma=0,
            max_depth=7,
            max_leaves=7,
            n_estimators=25,
            objective='reg:logistic',
            reg_alpha=0,
            reg_lambda=0.20833333333333334,
            subsample=1,
            tree_method='auto'
        ))],
        verbose=False
    )), ('11', Pipeline(
        memory=None,
        steps=[('standardscalerwrapper', StandardScalerWrapper(
            copy=True,
            with_mean=False,
            with_std=False
        )), ('xgboostclassifier', XGBoostClassifier(
            random_state=0,
            n_jobs=1,
            problem_info=ProblemInfo(
                gpu_training_param_dict={'processing_unit_type': 'cpu'}
            ),
            booster='gbtree',
            colsample_bytree=0.6,
            eta=0.3,
            gamma=0,
            max_depth=6,
            max_leaves=0,
            n_estimators=10,
            objective='reg:logistic',
            reg_alpha=0.3125,
            reg_lambda=2.3958333333333335,
            subsample=1,
            tree_method='auto'
        ))],
        verbose=False
    )), ('6', Pipeline(
        memory=None,
        steps=[('standardscalerwrapper', StandardScalerWrapper(
            copy=True,
            with_mean=False,
            with_std=False
        )), ('xgboostclassifier', XGBoostClassifier(
            random_state=0,
            n_jobs=1,
            problem_info=ProblemInfo(
                gpu_training_param_dict={'processing_unit_type': 'cpu'}
            ),
            booster='gbtree',
            colsample_bytree=0.5,
            eta=0.3,
            gamma=0,
            max_depth=10,
            max_leaves=255,
            n_estimators=10,
            objective='reg:logistic',
            reg_alpha=0,
            reg_lambda=0.10416666666666667,
            subsample=0.7,
            tree_method='auto'
        ))],
        verbose=False
    )), ('27', Pipeline(
        memory=None,
        steps=[('sparsenormalizer', Normalizer(
            copy=True,
            norm='l1'
        )), ('xgboostclassifier', XGBoostClassifier(
            random_state=0,
            n_jobs=1,
            problem_info=ProblemInfo(
                gpu_training_param_dict={'processing_unit_type': 'cpu'}
            ),
            booster='gbtree',
            colsample_bytree=0.6,
            eta=0.3,
            gamma=0,
            grow_policy='lossguide',
            max_bin=255,
            max_depth=6,
            max_leaves=0,
            n_estimators=25,
            objective='reg:logistic',
            reg_alpha=1.7708333333333335,
            reg_lambda=1.25,
            subsample=0.8,
            tree_method='hist'
        ))],
        verbose=False
    )), ('3', Pipeline(
        memory=None,
        steps=[('sparsenormalizer', Normalizer(
            copy=True,
            norm='l2'
        )), ('xgboostclassifier', XGBoostClassifier(
            random_state=0,
            n_jobs=1,
            problem_info=ProblemInfo(
                gpu_training_param_dict={'processing_unit_type': 'cpu'}
            ),
            booster='gbtree',
            colsample_bytree=0.7,
            eta=0.01,
            gamma=0.01,
            max_depth=7,
            max_leaves=31,
            n_estimators=10,
            objective='reg:logistic',
            reg_alpha=2.1875,
            reg_lambda=1.0416666666666667,
            subsample=1,
            tree_method='auto'
        ))],
        verbose=False
    )), ('31', Pipeline(
        memory=None,
        steps=[('standardscalerwrapper', StandardScalerWrapper(
            copy=True,
            with_mean=False,
            with_std=False
        )), ('xgboostclassifier', XGBoostClassifier(
            random_state=0,
            n_jobs=1,
            problem_info=ProblemInfo(
                gpu_training_param_dict={'processing_unit_type': 'cpu'}
            ),
            booster='gbtree',
            colsample_bytree=0.5,
            eta=0.2,
            gamma=5,
            max_depth=9,
            max_leaves=0,
            n_estimators=25,
            objective='reg:logistic',
            reg_alpha=0.3125,
            reg_lambda=0.4166666666666667,
            subsample=0.5,
            tree_method='auto'
        ))],
        verbose=False
    ))],
    weights=[0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.07142857142857142, 0.07142857142857142, 0.07142857142857142, 0.07142857142857142, 0.14285714285714285, 0.14285714285714285],
    flatten_transform=None,
    classification_labels=array([0, 1, 2])
))]
```

#### Parameters of the model
Following are the Model Parameters
1. Experiment Timeout : 45 minutes
2. Concurrent Iterations : 5
3. Cross Validations :3
4. Primary Metric : accuracy
5. Task : Classification
6. Label Column Name : RiskLevel

```
automl_settings = {
    "experiment_timeout_minutes": 45,
    "max_concurrent_iterations": 5,
    "n_cross_validations":3,
    "primary_metric" : 'accuracy'
}

#  Put your automl config here

automl_config = AutoMLConfig(compute_target=compute_target,
                             task = "classification",
                             training_data=train_data,
                             label_column_name="RiskLevel",   
                             #path = project_folder,
                             enable_early_stopping= True,
                             featurization= 'auto',
                             debug_log = "automl_errors.log",
                             enable_onnx_compatible_models=True,
                             **automl_settings
                            )
```

#### How could you have improved it
1. Prep data and address missing values, resampling training data, Adaptive Synthetic,Synthetic Minority Over-sampling Technique SMOTE 
##### Fig-2: Missing Data
![image](https://user-images.githubusercontent.com/32674614/159722672-d94f7325-385b-4160-8527-940976a4b1a4.png)
2. Change cross validation to reduce overfitting
3. Increase experiment time out so that the run can go over more types of model and look for better reults

[AutoMLConfig](https://docs.microsoft.com/en-us/python/api/azureml-train-automl-client/azureml.train.automl.automlconfig.automlconfig?view=azure-ml-py)
The cross validation checks overfitting and for computational reasons pre-defined timeout was set to 45 Minutes  which limits number of Models that could be built.

I have restricted the experiment time out to 45 minutes  in an ideal scenerio I will increase experiment time out at the same time will increase number of cross validation. Model was trained on very small dataset which couldnt explore the full potential, i could have extrapolate data using multiple methods as resampling training data, Adaptive Synthetic,Synthetic Minority Over-sampling Technique SMOTE etc. 

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.
##### Fig-3: AutoML Creation of Compute Cluster
![image](https://user-images.githubusercontent.com/32674614/159729578-8bf948a7-0971-4c60-8f1c-177484fc5051.png)
##### Fig-4: Compute Cluster Created
![image](https://user-images.githubusercontent.com/32674614/159730489-3d79880b-a976-4bfe-b399-58605d75c939.png)
##### Fig-5: AutoML Experiment Submitted
![image](https://user-images.githubusercontent.com/32674614/159732107-467bcd2b-d1d5-40a8-b2bf-52cd18b04969.png)
![image](https://user-images.githubusercontent.com/32674614/159732150-b58332fc-294a-4aaa-9c07-c462f99d483f.png)

###### Fig-6: AutoML experiment running
![image](https://user-images.githubusercontent.com/32674614/159734262-5d612e90-9d75-4706-b9f4-06ab8efe57d3.png)

![image](https://user-images.githubusercontent.com/32674614/159732259-5bdde1ca-cdd5-42a6-a2ac-5ac7ba0dbe3c.png)
![image](https://user-images.githubusercontent.com/32674614/159732329-38bb84c4-a8c2-48cd-889d-5421febb8b41.png)
![image](https://user-images.githubusercontent.com/32674614/159732385-a81f6d29-ac15-4c11-996c-f58ba7741ba3.png)
![image](https://user-images.githubusercontent.com/32674614/159732404-bad2318d-2b7e-46bb-9a2e-cf707b000223.png)
![image](https://user-images.githubusercontent.com/32674614/159732557-4dec997d-306d-4dd9-bb1f-ea0b3403cbd1.png)
![image](https://user-images.githubusercontent.com/32674614/159732650-02479238-f7c2-429d-8ebe-dfbe8e3be34f.png)



##### Fig-7: AutoML child runs in progress
![image](https://user-images.githubusercontent.com/32674614/159732480-a13ea6a1-7596-4616-a984-e9652c223f7c.png)

###### Fig-8: AutoMl run completed
![image](https://user-images.githubusercontent.com/32674614/159732801-9d3c18c0-a8f6-4247-924f-66b40ea1bf1d.png)


###### Fig-9: Best AutoML model `VotingEnsemble`
![image](https://user-images.githubusercontent.com/32674614/159732954-985c45c8-579c-4515-84b9-4227e317c08d.png)

###### Fig-10: Accuracy for `VotingEnsemble` 
![image](https://user-images.githubusercontent.com/32674614/159733149-c93ef3d1-2c99-4052-920b-8c829384818d.png)
![image](https://user-images.githubusercontent.com/32674614/159733199-c36b4f86-9f3d-4191-bf71-a228dbf29525.png)


###### Fig-10: Feature Importance with BS has the highest
![image](https://user-images.githubusercontent.com/32674614/159733872-fb791663-d526-465f-8812-40248cd0cc27.png)

###### Fig11.1: Model Metrics
![image](https://user-images.githubusercontent.com/32674614/159733378-32ff390d-a928-49af-9af9-bddd8c29c47a.png)
###### Fig11.2: Model Metrics continued..
![image](https://user-images.githubusercontent.com/32674614/159733497-4e64fbe5-2093-4bd4-a1b8-c488ef075ad9.png)
###### Fig11.3: Model Metrics continued..
![image](https://user-images.githubusercontent.com/32674614/159733778-a9074f31-e6a9-4abd-b178-0c0c630e9c1b.png)

###### Fig-12: Data Transformation Preview
![image](https://user-images.githubusercontent.com/32674614/159734054-35dda918-910e-41f0-975c-97f7eee4b4e1.png)

###### Fig13: Endpoint View
![image](https://user-images.githubusercontent.com/32674614/159735321-bbcb0549-6a5c-4790-b4c3-9a4dad2820b8.png)
![image](https://user-images.githubusercontent.com/32674614/159735379-594d3ba4-d99e-4a29-bdc7-de8b798676f0.png)


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
