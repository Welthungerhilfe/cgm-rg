# base docker file
FROM python:3.6-slim-stretch

RUN apt-get -y update
RUN apt-get install -y --fix-missing \
    build-essential \
    cmake \
    gfortran \
    git \
    wget \
    curl \
    graphicsmagick \
    libgraphicsmagick1-dev \
    libatlas-base-dev \
    libavcodec-dev \
    libavformat-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    liblapack-dev \
    libswscale-dev \
    pkg-config \
    python3-dev \
    python3-numpy \
    software-properties-common \
    cron \
    zip \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*

RUN cd ~ && \
    mkdir -p dlib && \
    git clone -b 'v19.9' --single-branch https://github.com/davisking/dlib.git dlib/ && \
    cd  dlib/ && \
    python3 setup.py install --yes USE_AVX_INSTRUCTIONS


WORKDIR /app

ADD requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

ADD . /app
RUN chmod +x deployment/*.sh

#Environment name of the RG
ENV RUN_ENV local

RUN mkdir log

RUN crontab deployment/crontab


#Working
#ENTRYPOINT ["python", "src/result_gen.py", "--height_model_id", "q3_depthmap_height_run_01", "--height_service", "q3-depthmap-height-run-01-ci", "--pose_model_id", "posenet_1.0", "--pose_service", "aci-posenet-ind", "--face_blur_model_id", "face_recogntion"]

#Working
#ENTRYPOINT ["pytest", "tests/"]

#Working
ENTRYPOINT ["cron", "-f"]