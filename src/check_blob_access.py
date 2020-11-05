import datetime
from azure.storage.blob import BlockBlobService, PublicAccess
import os

now = datetime.datetime.now()

# Setting up credentials to access Azure storage account
ACC_NAME = os.environ['ACC_NAME']
ACC_KEY = os.environ['ACC_KEY']

block_blob_service = BlockBlobService(account_name=ACC_NAME, account_key=ACC_KEY)
# Creating the timestamp for blob-container name
date_string = str(now.month) + str(now.day) + str(now.hour) + str(now.minute)
container_name = "test"
# Creating blob-container inside the storage account
block_blob_service.create_container(container_name)
block_blob_service.set_container_acl(container_name, public_access=PublicAccess.Container)

#base_page="https://www.docker.com/"
#html = urlopen(base_page).read()
file_name='SC'+ date_string +".txt"

with open(file_name, 'w') as fh:
    fh.write("well it works")
print('Uploading to blob')
block_blob_service.create_blob_from_path(container_name, file_name, file_name)