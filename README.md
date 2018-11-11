# frcnn-fastcampus

# Dependencies
For python dependencies, a `requirements.txt` file is provided in the root directory. You can install the prerequisites by typing the command below.
> pip install -r requirements.txt

# Dataset
The model in this code use the `PennFudanPed` dataset by default. You can download the dataset at the link below.
https://www.cis.upenn.edu/~jshi/ped_html/
Just unzip and modify the `DATASET_DIR` in `datasets/pfp_dataset.py` with the location of your dataset!

# Step by step explanation
For simpler and stand-alone explanation, please read the `notebooks/simple_faster_rcnn.ipynb` file.

# Run
You can start the training process with .sh script file provided in the root directory.
>./train.sh

# Lisence
This code is a simplified version of the original repository which is written by Xinlei Chen and Zheqi He. You can refer to the original version here(https://github.com/endernewton/tf-faster-rcnn).
