#!/bin/bash

# Set path
if [ ! -f config.sh ]; then
    echo "There is no file : config.sh"
    exit 1
fi
. ./config.sh || exit 1

# Check number of arguments
if [ $# -ne 1 ]; then
    echo "Usage : download_data.sh DATA_NAME"
    exit 1
fi

# Get name of data
FILE=$1

echo "Specified [$FILE]"

if [[ $FILE != "ae_photos" && $FILE != "apple2orange" && $FILE != "summer2winter_yosemite" &&  $FILE != "horse2zebra" && $FILE != "monet2photo" && $FILE != "cezanne2photo" && $FILE != "ukiyoe2photo" && $FILE != "vangogh2photo" && $FILE != "maps" && $FILE != "cityscapes" && $FILE != "facades" && $FILE != "iphone2dslr_flower" && $FILE != "mini" && $FILE != "mini_pix2pix" && $FILE != "mini_colorization" ]]; then
    echo "Available datasets are: apple2orange, summer2winter_yosemite, horse2zebra, monet2photo, cezanne2photo, ukiyoe2photo, vangogh2photo, maps, cityscapes, facades, iphone2dslr_flower, ae_photos"
    exit 1
fi
if [[ $FILE == "cityscapes" ]]; then
    echo "Due to license issue, we cannot provide the Cityscapes dataset from our repository. Please download the Cityscapes dataset from https://cityscapes-dataset.com, and use the script ./datasets/prepare_cityscapes_dataset.py."
    echo "You need to download gtFine_trainvaltest.zip and leftImg8bit_trainvaltest.zip. For further instruction, please read ./datasets/prepare_cityscapes_dataset.py"
    exit 1
fi



URL=https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/$FILE.zip
ZIP_FILE=$DATA_DIR/$FILE.zip
TARGET_DIR=$DATA_DIR/$FILE/
wget -N $URL -O $ZIP_FILE
mkdir $TARGET_DIR
unzip -v $ZIP_FILE -d $DATA_DIR/
#rm $ZIP_FILE