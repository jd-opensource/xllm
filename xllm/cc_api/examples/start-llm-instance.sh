#!/bin/bash

clear

# export ASDOPS_LOG_LEVEL=DEBUG
# export ASDOPS_LOG_TO_STDOUT=1
export ASCEND_RT_VISIBLE_DEVICES=12
python3 -c "import torch; import torch_npu; torch_npu.npu.set_device('npu:0')"

# build/single_llm_instance
build/multiple_llm_instances
