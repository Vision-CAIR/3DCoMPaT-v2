#!/bin/bash

# FINE
# Generate compositions (fine)
python3 code/gen_json.py --model-list ./data/lists/train_list.json \
	--output-folder ./json/train/ \
	--meta-folder ./metadata/ \
	--styles-count 1000 \
	--semantic-level fine

# Generate segmentation placeholders (fine)
python3 code/gen_json.py --model-list ./data/lists/train_list.json \
	--output-folder ./json/train/ \
	--meta-folder ./metadata/ \
	--styles-count 1 \
	--semantic-level fine \
	--seg-mode

# COARSE
# Generate compositions (coarse)
python3 code/gen_json.py --model-list ./data/lists/train_list.json \
	--output-folder ./json/train/ \
	--meta-folder ./metadata/ \
	--styles-count 1000 \
	--semantic-level coarse

# Generate segmentation placeholders (coarse)
python3 code/gen_json.py --model-list ./data/lists/train_list.json \
	--output-folder ./json/train/ \
	--meta-folder ./metadata/ \
	--styles-count 1 \
	--semantic-level coarse \
	--seg-mode
