#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <gs_path_to_prompts> <output_folder_name>"
    echo "Example: $0 gs://cohere-dev-central-2/huey/dev/eng_pool_252694.jsonl synthetic-translations"
    exit 1
fi

GS_PROMPT_PATH=$1
OUTPUT_FOLDER=$2
PROMPT_FILE=$(basename $GS_PROMPT_PATH)

echo "Downloading English prompt pool..."
gsutil cp $GS_PROMPT_PATH .

echo "Running translation job..."
python3 synthetic-generations.py "./$PROMPT_FILE" "./$OUTPUT_FOLDER"

echo "Uploading results to google storage..."
gsutil cp -r "./$OUTPUT_FOLDER" "$(dirname $GS_PROMPT_PATH)/"

echo "Cleaning up local files..."
rm $PROMPT_FILE
rm -r $OUTPUT_FOLDER

echo "Job Complete"