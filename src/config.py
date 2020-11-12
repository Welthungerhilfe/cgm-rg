import os
import json

ACC_NAME = os.environ['ACC_NAME']
ACC_KEY = os.environ['ACC_KEY']

ENV = os.environ["ENV"]

DB_NAME = "cgm"
DB_USER = os.environ["DB_USER"]
DB_HOST = os.environ["DB_HOST"]

DB_PASSWD = os.environ["DB_PASS"]
DB_PORT = 5432
DB_SSL_MODE = "require"

TENANT_ID = os.environ["TENANT_ID"]
SP_ID = os.environ["SP_ID"]
SP_PASSWD = os.environ["SP_PASS"]
SUB_ID = os.environ["SUB_ID"]
