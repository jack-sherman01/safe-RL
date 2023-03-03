#!/bin/bash
# plz use sudo if probolems happended.
apt update
apt install -y python3-pip ffmpeg zip unzip libsm6 libxext6 libgl1-mesa-dev libosmesa6-dev libgl1-mesa-glx patchelf


# install Python dependencies in a virtualenv
virtualenv -p python3 venv
. venv/bin/activate
pip3 install numpy scipy gym dotmap matplotlib tqdm opencv-python tensorboardX moviepy plotly gdown
pip3 install torch==1.4.0
pip3 install torchvision==0.5.0
# if errors occured when run the following command, you can run commands in advance:
#sudo apt-get update
#sudo apt-get install libgl1-mesa-dev libgl1-mesa-glx libosmesa6-dev python3-pip python3-numpy python3-scipy

pip3 install mujoco_py==1.50.1.68

# download demonstration files for shelf, maze
gdown --id 10M7pzsvKP4DcDDqxctDzx_opGfoV6rVt --output recovery_rl_data.zip
unzip recovery_rl_data.zip
rm recovery_rl_data.zip
