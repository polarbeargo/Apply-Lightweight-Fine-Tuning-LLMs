#!/bin/bash
python3 -m venv .env
source .env/bin/activate
chmod +x installLightweightFineTuningLLMs.sh  
./installLightweightFineTuningLLMs.sh
pip freeze > requirements.txt