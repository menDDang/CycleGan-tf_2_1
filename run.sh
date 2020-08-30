#!/bin/bash

# Configuration
DATA_NAME="vangogh2photo"
version="v1"
start_step=0

# Set path
if [ ! -f config.sh ]; then
    echo "Error!!!!! There is no file : config.sh"
    exit 1
fi
. ./config.sh || exit 1


# Step 0. Data preparation
if [ $start_step -le 0 ]; then
    echo "Step 0 : Prepare data for $DATA_NAME"

    # Download data
    if [ ! -d $DATA_DIR/$DATA_NAME ]; then
        echo "Download data..."
        mkdir -p $DATA_DIR
        bash scripts/download_data.sh $DATA_NAME || exit 1
    fi

    # Prepare data
    # In this tutorial, data is already prepared.
    # So, just using it

    # Count number of files
    echo "[Info] Number of data :"
    for task in "train" "test"; do
        for type in "A" "B"; do
            file_num=$(ls $DATA_DIR/$DATA_NAME/${task}${type} | wc -l)
            echo "    Number of file for ${task}${type} : $file_num"
        done
    done
    
fi

# Step 1. Train data
if [ $start_step -le 1 ]; then
    echo "Step 1 : Train Cycle GAN for $DATA_NAME"

    python $SRC_DIR/train_model.py \
        --config $WORK_DIR/${DATA_NAME}_${version}.yaml \
        --chkptdir $OUT_DIR/chkpts/${DATA_NAME}_${version} \
        --logdir $OUT_DIR/logs/${DATA_NAME}_${version} \
        || exit 1
fi

# Step 2. Inference model
