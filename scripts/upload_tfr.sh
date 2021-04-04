# Note:
# `BUCKET_NAME` needs to be set externally.

echo "Uploading TFRecords to Storage Bucket..."
echo gs://${BUCKET_NAME}

gsutil -m cp -r ../trainer/train_tfr gs://${BUCKET_NAME}
gsutil -m cp -r ../trainer/validation_tfr gs://${BUCKET_NAME}

gsutil ls -lh gs://${BUCKET_NAME}