import json
import os
import run_env

# RUN_ENV = os.environ['RUN_ENV']

RUN_ENV = run_env.RUN_ENV

print("Result Generation running in", RUN_ENV)
print(RUN_ENV)
print(len(RUN_ENV))


if RUN_ENV == 'local':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    connection_file = "dbconnection.json"
    db_connection_full_path = os.path.join(dir_path, connection_file)
    print(db_connection_full_path)

    with open(db_connection_full_path) as json_file:
        json_data = json.load(json_file)

    ACC_NAME = "cgminbmzci" + json_data["Environment"] + "sa"
    ACC_KEY = json_data["account_key"]
    # ENV = json_data["Environment"]

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

else:
    ACC_NAME = os.environ['ACC_NAME']
    ACC_KEY = os.environ['ACC_KEY']

    DB_NAME = "cgm"
    DB_USER = os.environ["DB_USER"]
    DB_HOST = os.environ["DB_HOST"]

    DB_PASSWD = os.environ["DB_PASSWD"]
    DB_PORT = 5432
    DB_SSL_MODE = "require"

    TENANT_ID = os.environ["TENANT_ID"]
    SP_ID = os.environ["SP_ID"]
    SP_PASSWD = os.environ["SP_PASSWD"]
    SUB_ID = os.environ["SUB_ID"]
