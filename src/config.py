import os

STORAGE_SECRET = os.environ['STORAGE_ACC']
ACC_NAME = STORAGE_SECRET.split(';')[0]
ACC_KEY = STORAGE_SECRET.split(';')[1]


DATABASE_CONFIG = os.environ['DB_CONFIG']
DB_NAME = DATABASE_CONFIG.split(';')[0]
DB_USER = DATABASE_CONFIG.split(';')[1]
DB_HOST = DATABASE_CONFIG.split(';')[2]
DB_PASSWD = DATABASE_CONFIG.split(';')[3]
DB_PORT = DATABASE_CONFIG.split(';')[4]
DB_SSL_MODE = DATABASE_CONFIG.split(';')[5]


AZURE_ML_CONFIG = os.environ['ML_CONFIG']
TENANT_ID = AZURE_ML_CONFIG.split(';')[0]
SP_ID = AZURE_ML_CONFIG.split(';')[1]
SP_PASSWD = AZURE_ML_CONFIG.split(';')[2]
SUB_ID = AZURE_ML_CONFIG.split(';')[3]