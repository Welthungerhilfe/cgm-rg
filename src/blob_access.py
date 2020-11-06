import os
import datetime
from azure.storage.blob import BlockBlobService, PublicAccess


def create_blob_storage(ACC_NAME, ACC_KEY, container_name):
    '''
    Create blob-container inside the storage account
    '''
    block_blob_service = BlockBlobService(account_name=ACC_NAME, account_key=ACC_KEY)
    block_blob_service.create_container(container_name)
    block_blob_service.set_container_acl(container_name, public_access=PublicAccess.Container)
    return block_blob_service


def download_text_blob(block_blob_service, container_name, file_name):
    '''
    Download the text blob file from storage
    '''
    print("Downloading from blob")
    block_blob_service.get_blob_to_path(container_name, file_name, file_name)
    text = open(file_name, 'r').read()
    
    return text


def upload_text_blob(block_blob_service, container_name, text, file_name):
    '''
    Upload text blob file to storage
    '''

    with open(file_name, 'w') as fh:
        fh.write(text)
    
    print('Uploading to blob')    
    block_blob_service.create_blob_from_path(container_name, file_name, file_name)


if __name__ == "__main__":

    # Setting up credentials to access Azure storage account
    ACC_NAME = os.environ['ACC_NAME']
    ACC_KEY = os.environ['ACC_KEY']

    container_name = "test"
    upload_text = 'Hello World'

    now = datetime.datetime.now()
    # Creating the timestamp for blob-container name
    date_string = str(now.month) + str(now.day) + str(now.hour) + str(now.minute)
    file_name='SC'+ date_string +".txt"

    block_blob_service = create_blob_storage(ACC_NAME, ACC_KEY, container_name)
    upload_text_blob(block_blob_service, container_name, upload_text, file_name)