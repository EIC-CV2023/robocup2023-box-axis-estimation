# robocup2023-face-recognition
Person Pose Estimation using YoloV8 Pose for EIC RoboCup@Home 2023


### Pose Index
    0: person sit down
    1: person stand up
    2: person sit down and raise hand
    3: person stand up and raise hand


## Install
    [Clone and cd to this repo]
    conda create -n pose-estimation python=3.9.16
    conda activate pose-estimation
    pip install -r requirements.txt

## Run
### Server
    conda activate pose-estimation
    python3 main.py
### Live Client
    python3 client_live.py
