#!/bin/bash

# File configuration
root_path=$(pwd)
data_path="$root_path/data"
blender_path="YOURBLENDERPATHHERE"
code_path="$root_path/code"

# JSON input files
json_file="$root_path/json/bake_mode/bake_shuffle.json"

# Semantic level
sem_level="fine"

# Input path
models_path=$data_path/models_debug/

# Starting index of the style files
style_index=-1

# Running the blender script
export PYTHONPATH=$PYTHONPATH:$code_path
$blender_path \
	--background --python "$code_path/main.py" \
	-- \
	--blender-config-dir "$code_path/config/" \
	--data-path $data_path \
	--json-file $json_file \
	--output-folder $data_path/output/model/ \
	--render-mode model \
	--models-path $models_path \
	--semantic-level $sem_level
