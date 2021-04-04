# Note:
# `PROJECT_ID` and `BUCKET_NAME` need to be set externally.

echo "Submitting job to AI Platform."

DATE=$(date "+%Y%m%d_%H%M%S")

# JOB_NAME: the name of your job running on AI Platform.
JOB_NAME=cats_dogs_$(date +%Y%m%d_%H%M%S)

# Set up variables for Docker
IMAGE_REPO_NAME=tensorflow_gpu_cats_dogs
IMAGE_TAG=catsdogs_tf_gpu
IMAGE_URI=gcr.io/${PROJECT_ID}/${IMAGE_REPO_NAME}:${IMAGE_TAG}

# Build and push the docker image
docker build -f Dockerfile -t ${IMAGE_URI} ./
docker push ${IMAGE_URI}

# Set up variables for training
REGION=asia-east1
TRAIN_FILES=gs://${BUCKET_NAME}/train_tfr
VALIDATION_FILES=gs://${BUCKET_NAME}/validation_tfr

# Submit job
gcloud ai-platform jobs submit training ${JOB_NAME} \
    --region ${REGION} \
    --master-image-uri ${IMAGE_URI} \
    --config ./config.yaml \
    -- \
    trainer/task.py --bucket ${BUCKET_NAME} \
    --train-pattern ${TRAIN_FILES} \
    --valid-pattern ${VALIDATION_FILES}