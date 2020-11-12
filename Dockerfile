FROM ubuntu:16.04

RUN apt-get update \
    && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y python3.6 python3.6-dev python3-pip \
    && apt-get -y install cron

RUN ln -sfn /usr/bin/python3.6 /usr/bin/python3 \
    && ln -sfn /usr/bin/python3 /usr/bin/python \
    && ln -sfn /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

ADD requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

ADD . /app
RUN chmod +x deployment/*.sh

ENTRYPOINT ["crontab", "deployment/crontab"]