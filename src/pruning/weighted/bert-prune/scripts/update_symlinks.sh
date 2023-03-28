#!/bin/bash

# set the base directory where the links are located
BASE_DIR="/home/azureuser/cloudfiles/code/Users/ucabmp4/bert-prune/data/pretraining/"

# set the new server's base directory
NEW_DIR="/home/azureuser/cloudfiles/code/Users/ucabmp4/bert-prune/data/pretraining/"
OLD_DIR="/home/azureuser/cloudfiles/code/Users/jimdilkes/bert-prune/data/pretraining/"


# loop through all symbolic links in the base directory
for link in $(find -L "$BASE_DIR" -type l); do
  # get the absolute path of the symbolic link
  absolute_path=$(readlink "$link")

  # replace the old server's base directory with the new one
  updated_path=${absolute_path/$OLD_DIR/$NEW_DIR}

  echo $updated_path
  # update the symbolic link with the new path
  ln -sf "$updated_path" "$link"

done