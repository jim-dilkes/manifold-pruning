#!/bin/bash

## This script will set up the correct environment to run the bert-prune scripts
#  with GPU on an ubuntu 20 instance with nvidia gpu

## First clone the repo using the commands below, then run setup.sh
# user_dir=~/cloudfiles/code/Users/jimdilkes
# cd $user_dir
# sudo apt-get update 
# sudo apt-get install git
# git clone https://github.com/jim-dilkes/bert-prune.git
# cd bert-prune

## It will not install the wiki extract package - this must be done using a separate
#  python 3.8+ environment

sudo apt-get update 
sudo apt install unzip lbzip2

# Set up python 3.7 env
conda init bash
source ~/.bashrc
conda create --name bert-prune python=3.7
conda activate bert-prune
conda install conda
conda update conda
conda install cudatoolkit=10.0 cudnn=7.6
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# Install python packages
conda install -r requirements.txt

# Download BERT model - BERT-Base
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
unzip uncased_L-12_H-768_A-12.zip

# # Download GLUE
# python scripts/get_glue.py --data_dir data/glue_data

# # Download BookCorpus
# ./scripts/download_bookcorpus.sh

# # Download Wiki
# ./scripts/download_wiki.sh

# Download Sentencized
# Must create a new SAS token on azure storage account
azcopy copy "https://mlworkspace6882075176.blob.core.windows.net/bert-prune/sentencized.tar.gz?sp=r&st=2023-03-15T18:56:14Z&se=2023-03-16T02:56:14Z&spr=https&sv=2021-12-02&sr=b&sig=J53rFGU7o2oWRjoX44qHuUdH932MktwUoTqYpqHEZ2A%3D" "./data/"