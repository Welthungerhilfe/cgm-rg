from azure.storage.blob import BlockBlobService, PublicAccess
import os


ACC_NAME = os.environ['ACC_NAME']
ACC_KEY = os.environ['ACC_KEY']

block_blob_service = BlockBlobService(account_name=ACC_NAME, account_key=ACC_KEY)



def download_blobs(file_list, container_name):
    """
        downloads the artifacts specified in file_list from blob storage
    """

    for file in file_list:
        try:
            block_blob_service.get_blob_to_path(container_name, file, file)
        except Exception as error:
            print(error)



def upload_blobs(file_list, container_name):
    """
        uploads the artifacts specified in file_list to blob storage
    """


    for (local_file_name, blob_file_name) in file_list:
        try:
            block_blob_service.create_blob_from_path(container_name, blob_file_name, local_file_name)
        except Exception as error:
            print(error)
