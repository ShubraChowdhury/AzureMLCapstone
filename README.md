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
### Fig-3.1: Data Profile
![Data Profile](https://user-images.githubusercontent.com/32674614/159392269-8e5b8259-4a52-4d34-b5b4-cbd871e34eee.png)
### Fig-3.2: Data Profile
![image](https://user-images.githubusercontent.com/32674614/159392328-8ba7b3de-0899-4448-a19e-13e9f8960092.png)
### Fig-3.3: Data Profile
![image](https://user-images.githubusercontent.com/32674614/159392358-f64b1939-0383-4cc4-ac9f-c873fcf4de0f.png)
### Fig-3.4: Data Profile
![image](https://user-images.githubusercontent.com/32674614/159392389-4d3ea79c-3bdd-4214-bc7e-d8976a53ee0e.png)


### Overview
*TODO*: Explain about the data you are using and where you got it from.

### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.

### Access
*TODO*: Explain how you are accessing the data in your workspace.

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
