---
layout: post
title: "Introduction to Amazon SageMaker"
author: "Timothy Shan"
tags: dl
---

## Introduction 

In this blog I will cover the topics from the `1-Day SageMaker workshop`. The barriers to ML include no-code ML tools, Amazon SageMaker Canvas, which is only available in the US, one of the organizer Vicent built a demo using Canvas to predict the probability of a passenger to survive the Titanic crash. Anothe barrier is data preparation tools, i.e. SageMaker data labeler. SageMaker integrates ML tools in a single interface, to build, train, and deploy using SageMaker studio. The last one is built-in ML ops capabilities, including the SageMaker pipeline. In short, SageMaker can be categorized into: 

- Prepare
- Build
- Train and tune 
- Deploy 

## Data preparation 

Let's start with how to prepare data, and here are the steps: collect data, prepare data, transform data, and pipeline data prep. This process can be tedious and takes weeks to complete. The process is much simpler with SageMaker. 

ML cannot work with categorical data, so it needs to be converted using one-hot label format. Moreover, the dataset needs to be split into train, validation and test sets. You can send data to SageMaker in 3 channels, SageMaker uses Python SDK to write the S3 to store the data. In a nutshell, the process is:

- convert the data to the input format 
- transform features to a more expresssive representation format 
- visualize the data to spot inconsistencies and diagnose & fix 
- export the data for training, can be used in notebook

`SageMaker Data Wrangler` provides a fast way to prepare the data. The data can be imported from various formats, e.g. cvs files. After the import, the data can be visualized and transformed, to spot the outliers, etc. There is a preview that can quickly estimate the model accuracy. Lastly, the data can be deployed for production. There are other ways to explore data such as the jupyter notebooks. For other formats of data, e.g. audio files, we can also store them on S3. 

Without using a feature store, we need to have standalone feature engineering for each new model. The process can be simplified with a feature store, by buiding features once, and reuse them across teams and models. The centralized store is kind of like a repository, where the users can search, reuse, train data, and has low-latency serving and train-infer consistency. There are two types for feature store, online and offline. The online version supports low millisecond latency reads, whereas the offline one supports batch predictions and model training, high throughput reads. 

## Workshop Setup 

The event engine is at `dashboard.eventengine.run/login`, and then we need to enter the event hash. The workshop details are accessible via https://tinyurl.com/ntucontlab. The repo for this even is 

```bash
git clone https://github.com/aws-samples/amazon-sagemaker-immersion-day.git
```

After downloading, we could move on to lab 1, and choose `Option 2: Numpy and Pandas`. The background is that for direct marketing, we need to filter the potential customers first to save time. The raw features include demographis (age, job, etc), past customer events (housing, loan), past direct marketing contacts (contact, duration), etc. The target variable is `Has the client subscribed a term deposit? (binary: 'yes','no')`. 

First we need to check the distribution of our data 

```python
# cell 07
# Frequency tables for each categorical feature
for column in data.select_dtypes(include=['object']).columns:
    display(pd.crosstab(index=data[column], columns='% observations', normalize='columns'))

# Histograms for each numeric features
display(data.describe())
%matplotlib inline
hist = data.hist(bins=30, sharey=True, figsize=(10, 10))
```

From the distribution, we can notice that most customers did not subscribe to a term deposit. Next we need to transform the data, where we handle missing values, convert categorical to numeric, normalize oddly distributed data, manipulate more complicated data types such as images, text. A simple pre-processing step is below 

```python
data['no_previous_contact'] = np.where(data['pdays'] == 999, 1, 0)                                 # Indicator variable to capture when pdays takes a value of 999
data['not_working'] = np.where(np.in1d(data['job'], ['student', 'retired', 'unemployed']), 1, 0)   # Indicator for individuals not actively employed
model_data = pd.get_dummies(data)  
```

Furthermore, we need to be careful to the features that will not add value to the final use case. Hence, we need to remove the `economic` features and `duration` from the data as they would need to be forecasted with high precision to use as inputs in future predictions. 

```python
model_data = model_data.drop(['duration', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed'], axis=1)
```

To combat overfitting, we randomly split the data into 3 uneven groups. The model will be trained on 70% of data, it will then be evaluated on 20% of data to give us an estimate of the accuracy we hope to have on unseen data, and 10% will be held back as test set.

```python
train_data, validation_data, test_data = np.split(model_data.sample(frac=1, random_state=1729), [int(0.7 * len(model_data)), int(0.9 * len(model_data))])   # Randomly sort the data then split out first 70%, second 20%, and last 10%
```

Lastly, we need to prepare the csv file for SageMaker's XGBoost container, and copy the file to S3 for training

```python
pd.concat([train_data['y_yes'], train_data.drop(['y_no', 'y_yes'], axis=1)], axis=1).to_csv('train.csv', index=False, header=False)
pd.concat([validation_data['y_yes'], validation_data.drop(['y_no', 'y_yes'], axis=1)], axis=1).to_csv('validation.csv', index=False, header=False)

boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'train/train.csv')).upload_file('train.csv')
boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'validation/validation.csv')).upload_file('validation.csv')
```

## Build, train and deploy ML model 

There are three ways to build your model. The first way is to use the built-in algorithms, eg KMeans, PCA, etc. There is no ML coding required. The second way is to bring your own script, eg in tensorflow, pytorch. The third way is to bring your own container, and use it in SageMaker. 

For a notebook running on the EC2 instance, the data will be retrieved from the S3 bucket, which sends to another EC2 instance for building the training environment. The model is saved to the elastic ontainer registry after training is finished.

### Built-in algorithm 

For the direct marketing dataset, we will use `gradient boosted trees` for prediction. They combine predictions from many simple models, each of which tries to address the weaknesses of the previous models. By doing this the collection of simple models can actually outperform large, complex models. `xgboost` is an extremely popular, open-source package for gradient boosted trees. We need to specify training parameters to the estimator:

1. The `xgboost` algorithm container
1. The IAM role to use
1. Training instance type and count
1. S3 location for output data
1. Algorithm hyperparameters

And then a `.fit()` function which specifies the S3 location for output data. 

```python
sess = sagemaker.Session()

xgb = sagemaker.estimator.Estimator(container,
                                    role, 
                                    instance_count=1, 
                                    instance_type='ml.m4.xlarge',
                                    output_path='s3://{}/{}/output'.format(bucket, prefix),
                                    sagemaker_session=sess)
xgb.set_hyperparameters(max_depth=5,
                        eta=0.2,
                        gamma=4,
                        min_child_weight=6,
                        subsample=0.8,
                        silent=0,
                        objective='binary:logistic',
                        num_round=100)

xgb.fit({'train': s3_input_train, 'validation': s3_input_validation}) 
```

Now that we've trained the `xgboost` algorithm on our data, let's deploy a model that's hosted behind a real-time endpoint. 

```python
xgb_predictor = xgb.deploy(initial_instance_count=1,
                           instance_type='ml.m4.xlarge')
```

### Bring your own script 

You can develop the training pipeline on your local machine, and train on SageMaker. SageMaker also has its own container registry, namely `Amazon ECR`, and are optimized for performance with NVIDIA driver, CUDA libraries, and intel libraries. The containers are split into the ones for training, and those for inference. Moreover, SageMaker training toolkit is a library already installed in AWS containers, which manages the parameters passed to the model. The workshop will focus on an alternative way, which is uses the container introduced in the next section. 

### Bring your own container 

The files that we’ll put in the container are:

- `nginx.conf` is the configuration file for the nginx front-end. Generally, you should be able to take this file as-is.
- `predictor.py` is the program that actually implements the Flask web server and the decision tree predictions for this app. You’ll want to customize the actual prediction parts to your application. Since this algorithm is simple, we do all the processing here in this file, but you may choose to have separate files for implementing your custom logic.
- `serve` is the program started when the container is started for hosting. It simply launches the gunicorn server which runs multiple instances of the Flask app defined in predictor.py. You should be able to take this file as-is.
- `train` is the program that is invoked when the container is run for training. You will modify this program to implement your training algorithm.
- `wsgi.py` is a small wrapper used to invoke the Flask app. You should be able to take this file as-is.

In summary, the two files you will probably want to change for your application are `train` and `predictor.py`. 

Here's the content of the `Dockerfile`:

```dockerfile
# Build an image that can do training and inference in SageMaker
# This is a Python 3 image that uses the nginx, gunicorn, flask stack
# for serving inferences in a stable way.

FROM ubuntu:18.04

MAINTAINER Amazon AI <sage-learner@amazon.com>


RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         python3-pip \
         python3-setuptools \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip

# Here we get all python packages.
# There's substantial overlap between scipy and numpy that we eliminate by
# linking them together. Likewise, pip leaves the install caches populated which uses
# a significant amount of space. These optimizations save a fair amount of space in the
# image, which reduces start up time.
RUN pip --no-cache-dir install numpy==1.16.2 scipy==1.2.1 scikit-learn==0.20.2 pandas flask gunicorn

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY decision_trees /opt/program
WORKDIR /opt/program
```

Note that the last two lines copy the training script to the docker container instance. Then we build our container 

```bash
cd lab03_container

chmod +x decision_trees/train
chmod +x decision_trees/serve

sm-docker build .  --repository sagemaker-decision-trees:latest
```

The last few lines from the output look like 

```bash
[Container] 2022/05/30 05:46:45 Phase complete: POST_BUILD State: SUCCEEDED
[Container] 2022/05/30 05:46:45 Phase context status code:  Message:

Image URI: 124150729991.dkr.ecr.ap-southeast-1.amazonaws.com/sagemaker-decision-trees:latest
```

We can use use the tools provided by the [SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/) to upload the data to a default bucket.

```python
WORK_DIRECTORY = 'lab03_data'

data_location = sess.upload_data(WORK_DIRECTORY, 
                                 key_prefix=S3_prefix)
```

In order to use SageMaker to fit our algorithm, we create an [`estimator`](https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html) that defines how to use the container to train. This includes the configuration we need to invoke SageMaker training:

- `image_uri (str)` - The [Amazon Elastic Container Registry](https://aws.amazon.com/ecr/) path where the docker image is registered. This is constructed in the shell commands in *cell 06*.
- `role (str)` - SageMaker IAM role as obtained above in *cell 03*.
- `instance_count (int)` - number of machines to use for training.
- `instance_type (str)` - the type of machine to use for training.
- `output_path (str)` - where the model artifact will be written.
- `sagemaker_session (sagemaker.session.Session)` - the SageMaker session object that we defined in *cell 04*.

Then we use `estimator.fit()` method to train against the data that we uploaded.
The API calls the Amazon SageMaker `CreateTrainingJob` API to start model training. The API uses configuration you provided to create the `estimator` and the specified input training data to send the `CreatingTrainingJob` request to Amazon SageMaker.

```python
account = sess.boto_session.client('sts').get_caller_identity()['Account']
region = sess.boto_session.region_name
image_uri = '{}.dkr.ecr.{}.amazonaws.com/sagemaker-decision-trees:latest'.format(account, region)

tree = sage.estimator.Estimator(image_uri,
                                role, 
                                instance_count=1, 
                                instance_type='ml.c4.2xlarge',
                                output_path="s3://{}/output".format(sess.default_bucket()),
                                sagemaker_session=sess)

file_location = data_location + '/iris.csv'
tree.fit(file_location)
```

After the model training successfully completes, you can call the [`estimator.deploy()` method](https://sagemaker.readthedocs.io/en/stable/estimators.html#sagemaker.estimator.Estimator.deploy). The `deploy()` method creates a deployable model, configures the SageMaker hosting services endpoint, and launches the endpoint to host the model. 

```python
from sagemaker.serializers import CSVSerializer
predictor = tree.deploy(initial_instance_count=1, 
                        instance_type='ml.m4.xlarge', 
                        serializer=CSVSerializer())
```

The predictor saves the HTTP endpoint and can be accessible in the SageMaker studio `Amazon SageMaker Endpoints`. For instance, the URL for my endpoint is 

```html
https://runtime.sagemaker.ap-southeast-1.amazonaws.com/endpoints/sagemaker-decision-trees-2022-05-30-05-52-30-987/invocations
```

The endpoint deployment is also available in the free tier, as mentioned [here](https://aws.amazon.com/free/?all-free-tier.sort-by=item.additionalFields.SortRank&all-free-tier.sort-order=asc&awsf.Free%20Tier%20Types=*all&awsf.Free%20Tier%20Categories=*all&all-free-tier.q=sagemaker&all-free-tier.q_operator=AND). To delete the endpoint and remove the container, do 

```bash
sess.delete_endpoint(predictor.endpoint_name)
!rm -rf lab03_container lab03_data
```

## Model debugging, monitoring and AutoML 

#### AutoML

In SageMaker Autopilot, its process is internally divided into multiple steps. All the user has to do is provide the training data, and the data partitioning, preprocessing and model selection, hyperparameter tuning, etc. will be performed automatically. Note that 

- Autopilot is capable of handling datasets up to 5 GB.
- Autopilot can take a long time to explore the data and create the models, for example the process for this demo is 1-4 hours long. 
- Canvas is using AutoML under the hood, Canvas just provides a interface where less code is required. 

Amazon SageMaker Autopilot takes care of preprocessing your data for you. You do not need to perform conventional data preprocssing techniques such as handling missing values, converting categorical features to numeric features, scaling data, and handling more complicated data types.

Moreover, splitting the dataset into training and validation splits is not necessary. Autopilot takes care of this for you. You may, however, want to split out a test set. That's next, although you use it for batch inference at the end instead of testing the model.

```python
train_data = data.sample(frac=0.8,random_state=200)

test_data = data.drop(train_data.index)

test_data_no_target = test_data.drop(columns=['y'])
```

The dataset used in this example is [Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic), in which we need to build models to predict the survival rate of each passenger. The features include age, name, gender, etc. After uploading the dataset to Amazon S3, you can invoke Autopilot to find the best ML pipeline to train a model on this dataset. 

The required inputs for invoking a Autopilot job are:
* Amazon S3 location for input dataset and for all output artifacts
* Name of the column of the dataset you want to predict (`y` in this case) 
* An IAM role

Currently Autopilot supports only tabular datasets in CSV format. Either all files should have a header row, or the first file of the dataset, when sorted in alphabetical/lexical order, is expected to have a header row.

You can also specify the type of problem you want to solve with your dataset (`Regression, MulticlassClassification, BinaryClassification`). In case you are not sure, SageMaker Autopilot will infer the problem type based on statistics of the target column (the column you want to predict). You can launch the job via 

```python
from time import gmtime, strftime, sleep
timestamp_suffix = strftime('%d-%H-%M-%S', gmtime())

auto_ml_job_name = 'automl-banking-' + timestamp_suffix
print('AutoMLJobName: ' + auto_ml_job_name)

sm.create_auto_ml_job(AutoMLJobName=auto_ml_job_name,
                      InputDataConfig=input_data_config,
                      OutputDataConfig=output_data_config,
                      AutoMLJobConfig=autoMLJobConfig,
                      AutoMLJobObjective=autoMLJobObjective,
                      ProblemType="BinaryClassification",
                      RoleArn=role)
```

SageMaker Autopilot job consists of the following high-level steps : 
* Analyzing Data, where the dataset is analyzed and Autopilot comes up with a list of ML pipelines that should be tried out on the dataset. The dataset is also split into train and validation sets.
* Feature Engineering, where Autopilot performs feature transformation on individual features of the dataset as well as at an aggregate level.
* Model Tuning, where the top performing pipeline is selected along with the optimal hyperparameters for the training algorithm (the last stage of the pipeline). 

```python
escribe_response = sm.describe_auto_ml_job(AutoMLJobName=auto_ml_job_name)
print (describe_response['AutoMLJobStatus'] + " - " + describe_response['AutoMLJobSecondaryStatus'])
job_run_status = describe_response['AutoMLJobStatus']
    
while job_run_status not in ('Failed', 'Completed', 'Stopped'):
    describe_response = sm.describe_auto_ml_job(AutoMLJobName=auto_ml_job_name)
    job_run_status = describe_response['AutoMLJobStatus']
    
    print (describe_response['AutoMLJobStatus'] + " - " + describe_response['AutoMLJobSecondaryStatus'])
    sleep(30)
```

Now use the describe_auto_ml_job API to look up the best candidate selected by the SageMaker Autopilot job. 

```python
best_candidate = sm.describe_auto_ml_job(AutoMLJobName=auto_ml_job_name)['BestCandidate']
best_candidate_name = best_candidate['CandidateName']
print(best_candidate)
print('\n')
print("CandidateName: " + best_candidate_name)
print("FinalAutoMLJobObjectiveMetricName: " + best_candidate['FinalAutoMLJobObjectiveMetric']['MetricName'])
print("FinalAutoMLJobObjectiveMetricValue: " + str(best_candidate['FinalAutoMLJobObjectiveMetric']['Value']))
```

#### Debugger 

Enabling Amazon SageMaker Debugger in training job can be accomplished by adding its configuration into Estimator object constructor:

```python
from sagemaker.debugger import DebuggerHookConfig, CollectionConfig

estimator = Estimator(
    ...,
    debugger_hook_config = DebuggerHookConfig(
        s3_output_path="s3://{bucket_name}/{location_in_bucket}",  # Required
        collection_configs=[
            CollectionConfig(
                name="metrics",
                parameters={
                    "save_interval": "10"
                }
            )
        ]
    )
)
```

Here, the `DebuggerHookConfig` object instructs `Estimator` what data we are interested in.
Two parameters are provided in the example:

- `s3_output_path`: it points to S3 bucket/path where we intend to store our debugging tensors.
  Amount of data saved depends on multiple factors, major ones are: training job / data set / model / frequency of saving tensors.
- `collection_configs`: it enumerates named collections of tensors we want to save.In this particular example, you are instructing Amazon SageMaker Debugger that you are interested in a single collection named `metrics`. We also instructed Amazon SageMaker Debugger to save metrics every 10 iteration.

Enabling Rules in training job can be accomplished by adding the `rules` configuration into Estimator object constructor.

- `rules`: This new parameter will accept a list of rules you wish to evaluate against the tensors output by this training job. For rules, Amazon SageMaker Debugger supports two types:
  - SageMaker Rules: These are rules specially curated by the data science and engineering teams in Amazon SageMaker which you can opt to evaluate against your training job.
  - Custom Rules: You can optionally choose to write your own rule as a Python source file and have it evaluated against your training job.

In this example, you will use a Amazon SageMaker's LossNotDecreasing rule, which helps you identify if you are running into a situation where the training loss is not going down.

```python
from sagemaker.debugger import rule_configs, Rule

estimator = Estimator(
    ...,
    rules=[
        Rule.sagemaker(
            rule_configs.loss_not_decreasing(),
            rule_parameters={
                "collection_names": "metrics",
                "num_steps": "10",
            },
        ),
    ],
)
```

- `rule_parameters`: In this parameter, you provide the runtime values of the parameter in your constructor. In this example, you will use Amazon SageMaker's LossNotDecreasing rule to monitor the `metircs` collection. The rule will alert you if the tensors in `metrics` has not decreased for more than 10 steps.

Amazon SageMaker starts one training job and one rule job for you. The first one is the job that produces the tensors to be analyzed. The second one analyzes the tensors to check if `train-rmse` and `validation-rmse` are not decreasing at any point during training.

```python
from sagemaker.debugger import rule_configs, Rule, DebuggerHookConfig, CollectionConfig
from sagemaker.estimator import Estimator
sess = sagemaker.Session()

save_interval = 5 

xgboost_estimator = Estimator(
    role=role,
    base_job_name=base_job_name,
    instance_count=1,
    instance_type='ml.m5.4xlarge',
    image_uri=container,
    max_run=1800,
    sagemaker_session=sess,
    debugger_hook_config=DebuggerHookConfig(
        s3_output_path=bucket_path,  # Required
        collection_configs=[
            CollectionConfig(
                name="metrics",
                parameters={
                    "save_interval": str(save_interval)
                }
            ),
            CollectionConfig(
                name="predictions",
                parameters={
                    "save_interval": str(save_interval)
                }
            ),
            CollectionConfig(
                name="feature_importance",
                parameters={
                    "save_interval": str(save_interval)
                }
            ),
            CollectionConfig(
                name="average_shap",
                parameters={
                    "save_interval": str(save_interval)
                }
            )
        ],
    )
)

xgboost_estimator.set_hyperparameters(max_depth=5,
                        eta=0.2,
                        gamma=4,
                        min_child_weight=6,
                        subsample=0.8,
                        silent=0,
                        objective='binary:logistic',
                        num_round=100)

xgboost_estimator.fit(
    {"train": s3_input_train, "validation": s3_input_validation},
    # This is a fire and forget event. By setting wait=False, you submit the job to run in the background.
    # Amazon SageMaker starts one training job and release control to next cells in the notebook.
    # Follow this notebook to see status of the training job.
    wait=False
)
```

After your training job is started, Amazon SageMaker starts a rule-execution job to run the LossNotDecreasing rule.

```python
import time
from time import gmtime, strftime


# Below command will give the status of training job
job_name = xgboost_estimator.latest_training_job.name
client = xgboost_estimator.sagemaker_session.sagemaker_client
description = client.describe_training_job(TrainingJobName=job_name)
print('Training job name: ' + job_name)
print(description['TrainingJobStatus'])

if description['TrainingJobStatus'] != 'Completed':
    while description['SecondaryStatus'] not in ['Training', 'Completed']:
        description = client.describe_training_job(TrainingJobName=job_name)
        primary_status = description['TrainingJobStatus']
        secondary_status = description['SecondaryStatus']
        print("{}: {}, {}".format(strftime('%X', gmtime()), primary_status, secondary_status))
        time.sleep(15)
```

For the built-in rules, typically they are more standard ones that can be applied to any model, e.g. those metrics such as CPU/GPU usage that does not require thresholds at all. On the contrary, the customized rules can be defined for your specific model, eg whether it is overfitting, underfitting, etc. More information can be found [here](https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-custom-rules.html). 

#### Monitor 

To enable data capture for monitoring the model data quality, you specify the new capture option called `DataCaptureConfig`. You can capture the request payload, the response payload or both with this configuration. The capture config applies to all variants. Go ahead with the deployment.

```python
from sagemaker.model_monitor import DataCaptureConfig

endpoint_name = 'DEMO-xgb-churn-pred-model-monitor-' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
print("EndpointName={}".format(endpoint_name))

data_capture_config = DataCaptureConfig(
                        enable_capture=True,
                        sampling_percentage=100,
                        destination_s3_uri=s3_capture_upload_path)

predictor = model.deploy(initial_instance_count=1,
                instance_type='ml.m4.xlarge',
                endpoint_name=endpoint_name,
                data_capture_config=data_capture_config)
```

You can now send data to this endpoint to get inferences in real time. Because you enabled the data capture in the previous steps, the request and response payload, along with some additional metadata, is saved in the Amazon Simple Storage Service (Amazon S3) location you have specified in the DataCaptureConfig. This step invokes the endpoint with included sample data for about 2 minutes. Data is captured based on the sampling percentage specified and the capture continues until the data capture option is turned off.

```python
from sagemaker.predictor import Predictor
import sagemaker
import time

predictor = Predictor(endpoint_name=endpoint_name, serializer=sagemaker.serializers.CSVSerializer())

# get a subset of test data for a quick test
!head -120 test_data/test-dataset-input-cols.csv > test_data/test_sample.csv
print("Sending test traffic to the endpoint {}. \nPlease wait...".format(endpoint_name))

with open('test_data/test_sample.csv', 'r') as f:
    for row in f:
        payload = row.rstrip('\n')
        response = predictor.predict(data=payload)
        time.sleep(0.5)
        
print("Done!")     
```

Now list the data capture files stored in Amazon S3. You should expect to see different files from different time periods organized based on the hour in which the invocation occurred.

In addition to collecting the data, Amazon SageMaker provides the capability for you to monitor and evaluate the data observed by the endpoints. For example, you can create a baseline with which you compare the realtime traffic. 
```python
# copy over the training dataset to Amazon S3 (if you already have it in Amazon S3, you could reuse it)
baseline_prefix = prefix + '/baselining'
baseline_data_prefix = baseline_prefix + '/data'
baseline_results_prefix = baseline_prefix + '/results'

baseline_data_uri = 's3://{}/{}'.format(bucket,baseline_data_prefix)
baseline_results_uri = 's3://{}/{}'.format(bucket, baseline_results_prefix)
print('Baseline data uri: {}'.format(baseline_data_uri))
print('Baseline results uri: {}'.format(baseline_results_uri))

training_data_file = open("test_data/training-dataset-with-header.csv", 'rb')
s3_key = os.path.join(baseline_prefix, 'data', 'training-dataset-with-header.csv')
boto3.Session().resource('s3').Bucket(bucket).Object(s3_key).upload_fileobj(training_data_file)

from sagemaker.model_monitor import DefaultModelMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat

my_default_monitor = DefaultModelMonitor(
    role=role,
    instance_count=1,
    instance_type='ml.m5.xlarge',
    volume_size_in_gb=20,
    max_runtime_in_seconds=3600,
)

my_default_monitor_baseline = my_default_monitor.suggest_baseline(
    baseline_dataset=baseline_data_uri+'/training-dataset-with-header.csv',
    dataset_format=DatasetFormat.csv(header=True),
    output_s3_uri=baseline_results_uri,
    wait=True
)
```

The model monitoring could be used to check model drift and data drift. For instance, if you have developed a model to process postal codes in Singapore, but some users are located in UK, the model won't work well because the format for the postal codes are quite different. This type of issues can be captured via monitoring. 

## Conclusion 

The admin account created for this workshop only lasts for three days, and does not have GPU access. We can save the contents to another account, or export to a github repo, my repo is [here](https://github.com/shanmo/amazon-sagemaker-immersion-day). The instructor shared some fun facts as well:

- the user can create an instance that has M1 chip from Apple, since AWS has collaborated with Apple
- Netflix uses AWS to do the graphics rendering, so all the work is done in the cloud

That's the end of the workshop, folks! 





