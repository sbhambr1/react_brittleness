#!/bin/bash

# Step 1: Define file path
FILE_PATH="/webshop/web_agent_site/utils.py"

# Step 2: Insert the new lines after line 12
sed -i '12a\
DEFAULT_ATTR_PATH = join(BASE_DIR, "../data/items_ins_v2.json")\
DEFAULT_FILE_PATH = join(BASE_DIR, "../data/items_shuffle.json")' $FILE_PATH

# DEFAULT_ATTR_PATH = join(BASE_DIR, '../data/items_ins_v2.json')
# DEFAULT_FILE_PATH = join(BASE_DIR, '../data/items_shuffle.json')

echo "Lines added to $FILE_PATH."

# Step 3: Source the virtual environment
source /webvenv/bin/activate
echo "Webshop webvenv Virtual environment activated."

cd /webshop/data/
# Step 4: Download the files using gdown
gdown https://drive.google.com/uc?id=1A2whVgOO0euk5O13n2iYDM0bQRkkRduB; # items_shuffle
gdown https://drive.google.com/uc?id=1s2j6NgHljiZzQNL3veZaAiyW_qDEgBNi; # items_ins_v2

echo "Files downloaded."