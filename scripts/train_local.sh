# Note:
# `PROJECT_ID` and `BUCKET_NAME` need to be set externally.

# Set up variables for Docker
IMAGE_REPO_NAME=tensorflow_gpu_cats_dogs
IMAGE_TAG=catsdogs_tf_gpu
IMAGE_URI=gcr.io/${PROJECT_ID}/${IMAGE_REPO_NAME}:${IMAGE_TAG}

# Build the docker image
docker build -f Dockerfile -t ${IMAGE_URI} ./

# Data
TRAIN_FILES=gs://${BUCKET_NAME}/train_tfr
VALIDATION_FILES=gs://${BUCKET_NAME}/validation_tfr

# Test your docker image locally
echo "Running the Docker Image"
docker run ${IMAGE_URI} \
      --bucket ${BUCKET_NAME} \
      --train-pattern ${TRAIN_FILES} \
      --valid-pattern ${VALIDATION_FILES}