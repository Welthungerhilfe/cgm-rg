import os
#import config
import datetime
from azure.storage.blob import BlockBlobService, PublicAccess

#ACC_NAME = config.ACC_NAME
#ACC_KEY = config.ACC_KEY

#block_blob_service = BlockBlobService(account_name=ACC_NAME, account_key=ACC_KEY)

def create_blob_storage(ACC_NAME, ACC_KEY, container_name):
    '''
    Create blob-container inside the storage account
    '''
    block_blob_service = BlockBlobService(
        account_name=ACC_NAME, account_key=ACC_KEY)
    block_blob_service.create_container(container_name)
    block_blob_service.set_container_acl(
        container_name, public_access=PublicAccess.Container)
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
    block_blob_service.create_blob_from_path(
        container_name, file_name, file_name)


def connect_blob_storage(ACC_NAME, ACC_KEY, container_name):
    '''
    Connect blob-container inside the storage account
    '''
    block_blob_service = BlockBlobService(
        account_name=ACC_NAME, account_key=ACC_KEY)
    return block_blob_service


def download_blobs(block_blob_service, container_name, file_list):
    """
    Downloads the artifacts specified in file_list from blob storage
    """
    for file in file_list:
        file_directory = os.path.dirname(file)
        if os.path.isdir(file_directory) == False:
            os.makedirs(file_directory)
        try:
            block_blob_service.get_blob_to_path(container_name, file, file)
        except Exception as error:
            print(error)


def upload_blobs(block_blob_service, container_name, file_list):
    """
    Uploads the artifacts specified in file_list to blob storage
    """
    for (local_file_name, blob_file_name) in file_list:
        try:
            block_blob_service.create_blob_from_path(container_name, blob_file_name, local_file_name)
        except Exception as error:
            print(error)


if __name__ == "__main__":

    # Setting up credentials to access Azure storage account
    ACC_NAME = os.environ['ACC_NAME']
    ACC_KEY = os.environ['ACC_KEY']

    container_name = "test"
    upload_text = 'Hello World'

    now = datetime.datetime.now()
    # Creating the timestamp for blob-container name
    date_string = str(now.month) + str(now.day) + \
        str(now.hour) + str(now.minute)
    file_name = 'SC' + date_string + ".txt"

    block_blob_service = create_blob_storage(ACC_NAME, ACC_KEY, container_name)
    upload_text_blob(
        block_blob_service,
        container_name,
        upload_text,
        file_name)