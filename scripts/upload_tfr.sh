echo "Uploading TFRecords to Storage Bucket..."

gsutil cp -r train_tfr gs://$BUCKET_NAME
gsutil cp -r validation_tfr gs://$BUCKET_NAME

gsutil ls -lh gs://$BUCKET_NAME