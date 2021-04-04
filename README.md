# Distributed-Training-in-TensorFlow-2-with-AI-Platform

Accompanying blog post: [TBD]

This repository provides code to train an image classification model in a distributed manner with the [`tf.distribute.MirroredStrategy`](https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy) strategy (single host multiple GPUs) in TensorFlow 2.4.1. We make use of the MLOps stack to do this:


- Docker to create a custom container so that the code is reproducible. 
- AI Platform `training` jobs (by GCP) to manage running the custom Docker container using multiple GPUs. It also handles automatic provisioning and de-provisioning of resources.  

Advantages of training in this manner (as opposed to doing that in a Jupyter Notebook environment) are the following:


- Resources (GPUs, CPUs, memory, etc.) are fully managed by the custom service we are using to orchestrate our training workflow. In this case, it is AI Platform. 
- Resources are automatically provisioned and de-provisioned by the service. It helps to prevent any unnecessary costs. 

Consider the following scenario. You are operating in a Jupyter Notebook environment on Cloud that is configured with multiple V100 GPUs. You have started your training run and you then went out to meet an old friend. Now, what if because of a pesky bug in the code, the actual training did not start? Your development environment will likely continue to run and you will incur charges for practically nothing. Of course, this won’t happen if you configured alert metrics. 

With the workflow demonstrated in this repository, you can get away from the above situation way more easily. 

## Steps to run the code

**Note**: One needs to have a [billing-enabled GCP project](https://cloud.google.com/billing/docs/how-to/modify-project) to *fully* follow these steps. 

We will use a cheap [AI Platform Notebook](https://cloud.google.com/ai-platform-notebooks) instance as our staging machine which we will use to build our custom Docker image, push it to [Google Container Registry (GCR)](https://cloud.google.com/container-registry), and submit a training job to AI Platform. Additionally, we will use this instance to create TensorFlow Records (TFRecords) from the original dataset (Cats vs. Dogs) and upload them to a GCS Bucket. AI Platform notebooks come pre-configured with many useful Python libraries, Linux packages like `docker` and also the command-line GCP tools like `gcloud`. 

*(I used a* `*n1-standard-4*` *instance (with TensorFlow 2.4 as the base image) which costs $0.141 hourly.)*


1. Set the following environmental variables and set the shell scripts to be executables:

    ```shell
    $ export PROJECT_ID=your-gcp-project-id
    $ export BUCKET_NAME=unique-gcs-bucket-name
    $ chmod +x scripts/*.sh
    ```

2. Create a GCS Bucket:

    ```shell
    $ gsutil mb ${BUCKET_NAME}
    ```

   You can additionally pass in the zone where you want to create the bucket like the following: `$ gsutil mb -l asia-east1 ${BUCKET_NAME}`. If all of your resources will be provisioned from that same zone, then you will likely get slightly performance boost. 


3. Create TFRecords and upload them to the GCS Bucket.

    ```shell
    $ cd scripts
    $ source upload_tfr.sh
    ```

4. Build the custom Docker container and run it locally:

    ```shell
    $ cd ~/Distributed-Training-in-TensorFlow-2-with-AI-Platform
    $ source scripts/train_local.sh
    ```

5. If  everything is looking good, you can interrupt the training run with `Ctrl-C` and proceed to running on Cloud:

    ```shell
    $ source scripts/train_cloud.sh
    ```

... and done! 

## About the files

```shell
    ├── config.yaml: Specifies the type of machine to use to run training on Cloud.
    ├── scripts
    │   ├── train_cloud.sh: Trains on Cloud with the given specifications. 
    │   ├── train_local.sh: Trains locally. 
    │   └── upload_tfr.sh: Creates and uploaded TFRecords to a GCS Bucket. 
    └── trainer
        ├── config.py: Specifies hyperparameters and other constants. 
        ├── create_tfrecords.py: Driver code for creating TFRecords. It is called by `upload_tfr.sh`. 
        ├── data_loader.py: Contains utilities for the data loader. 
        ├── model_training.py: Contains the actually data loading and model training code.
        ├── model_utils.py: Contains model building utilities. 
        ├── task.py: Parses the command-line arguments given and starts an experiment.
        └── tfr_utils.py: Utilities for creating TFRecords. 
```

## References
https://github.com/GoogleCloudPlatform/ai-platform-samples

## Acknowledgements

I am thankful to the [ML-GDE program](https://developers.google.com/programs/experts/) for their generous GCP support.