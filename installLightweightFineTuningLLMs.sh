#!/bin/sh
pip3 -q install git+https://github.com/huggingface/transformers.git
pip3 install -q -U git+https://github.com/huggingface/trl
pip3 install -U accelerate
pip3 install wandb
pip3 install torch torchvision
pip3 install peft
pip3 install numpy
pip3 install datasets
pip3 install evaluate
pip3 install sklearn
pip3 install matplotlib
pip3 install ipywidgets
pip3 install bitsandbytes