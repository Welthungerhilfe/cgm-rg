import os
import datetime
from azure.storage.blob import BlockBlobService, PublicAccess

import blob_access


def test_blob_access():
    '''
    Test case to check the blob access
    '''
    ACC_NAME = os.environ['ACC_NAME']
    ACC_KEY = os.environ['ACC_KEY']

    container_name = "test"
    upload_text = 'Hello World'


    now = datetime.datetime.now()
    # Creating the timestamp for blob-container name
    date_string = str(now.month) + str(now.day) + str(now.hour) + str(now.minute)
    file_name='SC'+ date_string +".txt"


    block_blob_service = blob_access.create_blob_storage(ACC_NAME, ACC_KEY, container_name)
    blob_access.upload_text_blob(block_blob_service, container_name, upload_text, file_name)
    download_text = blob_access.download_text_blob(block_blob_service, container_name, file_name)

    assert upload_text == download_text
    #assert 1 == 1