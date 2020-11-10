#!/bin/bash

# TODO copyright

#remove all running cronjobs
echo "Starting setup"

#echo "Moving files"
cp -r ./src ./cgm-rg && rmdir --ignore-fail-on-non-empty ./src
cp -r ./deployment ./cgm-rg && rmdir --ignore-fail-on-non-empty ./deployment

echo "Removing cron jobs"
crontab -r &> /dev/null

# TODO check/setup permissions for folders


# TODO as root apt install -y python3-pip
echo "Installing pip"
sudo apt-get -qq install python3-pip --assume-yes

#installing requirements for cron jobs
echo "Installing jq"
sudo apt-get install -y jq



echo "Installing psycopg dependencies"
sudo apt-get -qq -y install libpq-dev python-dev

#rm -rf ~/.local

# to ignore already installed packages
export PIP_IGNORE_INSTALLED=0

#sudo apt-get -qq install python-commandnotfound

echo "Installing Python packages from requirements.txt"
pip3 install --user -r ~/deployment/requirements.txt

#upgrade azure-storage
#pip3 install azure-storage --upgrade
pip3 install azure-storage-queue==2.1.0
pip3 install tensorflow==1.13.1
pip3 install matplotlib==2.2.2
pip3 install pyntcloud==0.0.1
pip3 install tqdm==4.31.1
pip3 install Pillow==5.1.0
pip3 install opencv-python==4.1.2.30
pip3 install bunch==1.0.1
pip3 install azureml
pip3 install azureml-core
pip3 install scikit-image
pip3 install --upgrade setuptools
pip3 install cmake
#sudo apt-get install build-essential cmake pkg-config libx11-dev libatlas-base-dev libgtk-3-dev libboost-python-dev -y
pip3 install dlib
pip3 install face_recognition
pip3 install --upgrade git+https://github.com/klintan/pypcd.git

sudo apt-get install python3-tk --assume-yes
sudo apt-get install libsm6 libxrender1 libfontconfig1 --assume-yes

#make directories needed for processing messages and logging

env=$(jq -r '.Environment' ~/cgm-etl/PythonCode/dbconnection.json)
folder="cgminbmz$env"
echo "Creating ~/$folder/db and ~/$folder/log"
cd ~
mkdir -p "$folder"
cd "$folder"

mkdir -p db  
mkdir -p log  

#cd ~

#if [ ! -d "cgm-ml" ]; then
#    mkdir -p cgm-ml
#    git clone https://github.com/Welthungerhilfe/cgm-ml
#fi

#setup cgm tables
#python3 ~/deployment/init_db.py 

#dos2unix ~/deployment/get_artifact_list_message.sh

# make files executable
cd ~/deployment
chmod +x *.sh

# TODO setup permissions ~/cgm-etl/functioncode/PythonCode/dbconnection.json
echo "Setting up cron jobs"
crontab ~/deployment/crontab

echo "done"
