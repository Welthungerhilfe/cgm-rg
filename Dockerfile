FROM python:3.6

WORKDIR /app

RUN apt-get update
RUN apt-get -y install cron

ADD requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

ADD . /app
RUN chmod +x deployment/*.sh

ENTRYPOINT ["crontab", "deployment/crontab"]