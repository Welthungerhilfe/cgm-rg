FROM python:3.6

WORKDIR /app

ADD requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

ADD . /app

CMD ["python", "src/check_blob_access.py"]
