import json
import os

connection_file = os.path.expanduser("~/src/dbconnection.json")

with open(connection_file) as json_file:
    json_data = json.load(json_file)


ACC_NAME = "cgminbmzci" + json_data["Environment"] + "sa"
ACC_KEY = json_data["account_key"]
ENV = json_data["Environment"]

DB_NAME = json_data["dbname"]
DB_USER = json_data["user"]
DB_HOST = json_data["host"]
DB_PASSWD = json_data["password"]
DB_PORT = json_data["port"]
DB_SSL_MODE = json_data["sslmode"]


TENANT_ID = json_data["tenant_id"]
SP_ID = json_data["sp_id"]
SP_PASSWD = json_data["sp_password"]
SUB_ID = json_data["sub_id"]
