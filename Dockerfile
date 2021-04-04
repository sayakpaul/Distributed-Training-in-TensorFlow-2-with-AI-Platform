# Install TensorFlow
FROM gcr.io/deeplearning-platform-release/tf2-gpu.2-4

WORKDIR /root

# Update TensorFlow to 2.4.1.
RUN pip install -U tensorflow

# Copies the trainer code to the docker image.
COPY trainer/config.py ./trainer/config.py
COPY trainer/data_loader.py ./trainer/data_loader.py
COPY trainer/model_utils.py ./trainer/model_utils.py
COPY trainer/model_trainer.py ./trainer/model_trainer.py

# Set up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-u", "trainer/model_trainer.py"]