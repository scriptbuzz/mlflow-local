
## Deploy MLFlow To Your Local Workstation and Track Your ML Experiments ##

### By Michael Bitar ###


This is a getting started guide to help deploy MLflow locally on a Macbook. But the concepts are similar in a Windows environment. 

![mflow logo](img/mlflow-logo-black.png)

If you are an MLops engineer like me, you typically spend most of your time experimenting with parameters, hyerparameters, preprocesing, and code changes on your local workstation using your favorite IDE or Jupyter Notebooks. In the past, I would use a spreadsheet to track my ML experiments. That was not fun.

The ML stack was begging for a toolchain that can bring DevOps-like functionality to ML development.  This led to the rise of tools referred to loosely as MLOps tools. One of those tools that's gaining traction is an open source tool called MLflow.

MLflow is an open source framework to help organize machine learning development workflows by tracking ML experiments, packaging code into reproducible executions, versioning model metadata, facilitating collaboration, and deploying models. 

MLflow can be deployed in many configurations. If you are developing locally most of the time and don't have the need for a central repo, MLflow can be installed locally on your laptop so you can track your experiments and never have to guess again if you tried certain parameters or not and what was the resulting ML model performance. 

In this guide I will walk you thru how I deployed MLflow on my local macbook. Later, I will publish guides on more advanced setups such as using a cloud-hosted resources to support team collaboration and for higher availability, scalability, and security. 

### LET'S GET STARTED! ### 

You can use this Jupyter Notebook as your starting point. Make sure you have access to the internet. You can install MLflow from your terminal window or from a Jupyter notebook code cell. 



```python
# Install mlflow

!pip install mlflow
```


```python
# Import needed support libraries 

import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
```

```python
import logging

# configure logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
```

```python
# define evaluation metrics. In this case I will select RMSE, MAE, R2 scores

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2
```


```python
#configure warning logs
warnings.filterwarnings("ignore")

# set random seed to reproduce same results everytime 

np.random.seed(40)

# Load the sample dataset csv file from this URL

csv_url = (
    "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
)
try:
    data = pd.read_csv(csv_url, sep=";")
except Exception as e:
    logger.exception(
        "Unable to download training & test CSV, check your internet connection. Error: %s", e
    )
    
# Split the data into training and test sets using (0.75, 0.25) split.
train, test = train_test_split(data)

# The predicted column is "quality" which is a scalar from [3, 9]

train_x = train.drop(["quality"], axis=1)
test_x = test.drop(["quality"], axis=1)
train_y = train[["quality"]]
test_y = test[["quality"]]


```


```python
# Run below code cell multiple times each with a different alpha and l1_ratio numbers using 
# values between 0 and 1. Each run will be tracked and listed in the MLflow dashboard.
# We will be using the ElasticNet model in this example. 
# It's a linear regression model with combined L1 and L2 priors as regularizers

alpha =  .35 # change this value for each run. This is used to multiply penalty terms.
l1_ratio =  .45  # change this value for each run. This var is a penalty value.


with mlflow.start_run():
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)

    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)
    
    # log vars of interest to be tracked and listed by MLflow
    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    if tracking_url_type_store != "file":

        # Register the model
        # There are other ways to use the Model Registry, which depends on the use case,
        # please refer to docs for more information:
        #
        # https://mlflow.org/docs/latest/model-registry.html#api-workflow
        mlflow.sklearn.log_model(lr, "model", registered_model_name="ElasticnetWineModel")
    else:
        mlflow.sklearn.log_model(lr, "model")


# for each run, above metrics will be saved in model's local directory where it will be picked up by MLflow
```

    Elasticnet model (alpha=0.350000, l1_ratio=0.450000):
      RMSE: 0.7616514499663437
      MAE: 0.5936841528680933
      R2: 0.17804834226795552



```python
# when done, open a terminal and go to your project's local working 
# directory and enter below command (without the hashtag)

# mlflow ui
```


```python
# from your browser, open the mflow dashboard using below url (without the hashtag)

# http://localhost:5000
```  

You should see a dashboard similar to the one in the image below but your content may vary. 
You can sort columns such as performance metrics to select the experemint with more promise and inspect its 
detailed metadata to determine which paramaters and hyperparamaters contributed to improved 
model performance the most.

You can also inspect model metadata and register models of interest to share it with 
other developers on your team. 


![title](img/mbitar-mflow-dash.jpg)


```python
# when you click on one of the experiment (runs) you can see more details such as related artifacts
#  ![title](img/mbitar-mflow-dash.jpg)

```

![title](img/mbitar-mflow-dash-details.jpg)


```python
# you can also track artifact metadata such as model matadata 
```

![title](img/mbitar-mflow-dash-artifacts.jpg)

MLflow is feature-packed. If you are not using an MLops tool, consider using MLflow. You will enjoy substantial gains in productivity by using local features alone. Once you feel comfortable with the basics of MLflow, you can exapnd your MLflow deployment to the cloud and to the rest of your ML team.

My next MLops guide will cover remote MLflow deployment and will touch on some of the tool's more advanced features to support cloud and team collaboration. 

Thank you.
