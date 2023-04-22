#!/bin/bash

# File configuration
root_path=$(pwd)
data_path="$root_path/data"
blender_path="YOURBLENDERPATHHERE"
code_path="$root_path/code"

# Semantic level
sem_level="fine"

# Input path
models_path=$data_path/models_debug/

# Output path
output_path=$data_path/output/${sem_level}/

# JSON input files
json_file="$root_path/json/debug/comp_${sem_level}_0.json"

# Running the blender script
export PYTHONPATH=$PYTHONPATH:$code_path
$blender_path \
	--background --python "$code_path/main.py" \
	-- \
	--blender-config-dir "$code_path/config/" \
	--data-path $data_path \
	--json-file $json_file \
	--models-path $models_path \
	--output-folder $output_path \
	--render-mode segmentation \
	--semantic-level $sem_level \
	--debug-mode
