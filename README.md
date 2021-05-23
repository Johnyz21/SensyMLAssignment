# Jonathan Marklands ML Assignment

This project is split into two sections, Data Science and Machine Learning Engineering.

The Data Science folder contains three items:
* The Heart Failures dataset.
* A jupyter notebook which analyses the dataset and explains the steps taken for model creation.
* A script used to create the model which predicts heart failure.

The Machine Learning folder includes an api to serve the model and the relevant tests.

## Setup

Create a clean environment with python 3.7 installed then install the packages within the requirements.txt file.
The following is an example of how to do this using Conda:

Create a conda environment

`conda create --name sensyne_ml_assignment python=3.7`

Ensure you have the conda-forge channel added

`conda config --append channels conda-forge`

Activate the environment

`conda activate sensyne_ml_assignment`

Install requirements

`pip install -r requirements.txt`

Once the packages are installed please complete both **Model Creation** and **Run the Api** steps 

### Model Creation

To create the model run the model_creation.py file within the data_science folder. Navigate to the data_science folder 
in terminal then run the following:

`python3 model_creation.py`

The model will be created in the root directory under a folder named model.

### Run the API
Navigate to the ml_engineering/app directory in a terminal then run the following:

`hypercorn api:app --bind localhost:9003`


### Example querying the api
The following is an example of how to query the model:

`curl localhost:9003/predict -H "Content-Type: application/json" --request POST --data '{"features":{"age":75.0,"ejection_fraction":30,"serum_sodium":130, "serum_creatinine":1.9, "time": 5 }}'`

Note that all features must be present when querying the model

### Jupyter notebook

To view the jupyter notebook which analysed the `heart_failure_clinical_records_dataset.csv` open a terminal. Navigate 
to the data_science directory and run the following:
 
`jupyter notebook`

The notebook should be available to viewed within your chosen browser


## Testing

### Unit Tests
Navigate to the ml_engineering directory in a terminal then run the following:

`python -m unittest discover`