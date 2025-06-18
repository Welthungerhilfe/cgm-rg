import json
from os import getenv
from aiolimiter import AsyncLimiter
import anyio
import logging
from utils.retry_decorator import retry
import pickle
import numpy as np


pose_api_semaphore = anyio.Semaphore(4)
face_api_limiter = AsyncLimiter(3, 1)  # 3 requests per second


@retry(retries=3, delay=2)
async def call_sam_api(session, payload):
    pose_score_uri = 'https://sam-endpoint-2.centralindia.inference.ml.azure.com/score'
    api_key = getenv("SAM_API_KEY")
    headers_up = {'Content-Type':'application/octer-stream', 'Authorization':('Bearer '+ api_key), 'azureml-model-deployment': 'blue'}
    async with session.post(pose_score_uri, headers=headers_up, data=payload) as resp:
        if resp.status != 200:
            logging.error('call sam api failed')
        data = await resp.read()
        return pickle.loads(data)


@retry(retries=3, delay=2)
async def call_pose_api(session, payload):
    pose_score_uri = 'https://pose-endpoint-2.centralindia.inference.ml.azure.com/score'
    api_key = getenv("POSE_API_KEY")
    headers_up = {'Content-Type':'application/octer-stream', 'Authorization':('Bearer '+ api_key), 'azureml-model-deployment': 'blue'}
    async with pose_api_semaphore:
        async with session.post(pose_score_uri, headers=headers_up, data=payload) as resp:
            if resp.status != 200:
                logging.error('call pose api failed')
            data = await resp.read()
            return json.loads(data)


@retry(retries=3, delay=2)
async def call_face_api(session, image_data):
    """Call Microsoft Face API with rate limiting."""
    params = {
        'returnFaceId': 'false',  # Do not return face IDs
        'returnFaceLandmarks': 'false',  # Do not return face landmarks
        'returnFaceAttributes': 'headPose',  # Exclude attributes
        'detectionModel': 'detection_03',
    }
    face_api_url = f"https://cgm-be-ci-dev-face-api.cognitiveservices.azure.com//face/v1.0/detect"

    ms_face_api_headers = {
        'Content-Type': 'application/octet-stream',
        'Ocp-Apim-Subscription-Key': getenv('MS_FACE_API_KEY'),
    }
    async with face_api_limiter:
        async with session.post(face_api_url, headers=ms_face_api_headers, params=params, data=image_data) as resp:
            if resp.status != 200:
                logging.error('call face api failed')
            data = await resp.read()
            return json.loads(data)


@retry(retries=3, delay=2)
async def call_pcnn_height(session, depthmaps):
    scoring_uri = "https://height-plaincnn-endpoint-test.centralindia.inference.ml.azure.com/score"
    api_key = getenv('PCC_HEIGHT_KEY')
    headers = {'Content-Type':'application/octer-stream', 'Authorization':('Bearer '+ api_key), 'azureml-model-deployment': 'blue'}
    pickled_input_data = pickle.dumps(np.array(depthmaps))
    async with session.post(scoring_uri, data=pickled_input_data, headers=headers) as resp:
        print(resp.status)
        data = await resp.read()
        return json.loads(data)


@retry(retries=3, delay=2)
async def call_pcnn_weight(session, depthmaps):
    scoring_uri = "https://weight-plaincnn-endpoint.centralindia.inference.ml.azure.com/score"
    api_key = getenv('PCC_WEIGHT_KEY')
    headers = {'Content-Type':'application/octer-stream', 'Authorization':('Bearer '+ api_key), 'azureml-model-deployment': 'blue'}
    pickled_input_data = pickle.dumps(np.array(depthmaps))
    async with session.post(scoring_uri, data=pickled_input_data, headers=headers) as resp:
        print(resp.status)
        data = await resp.read()
        return json.loads(data)


@retry(retries=3, delay=2)
async def call_mn_height(session, depthmaps):
    scoring_uri = "https://mobilenet-v2-height-endpoint-2.centralindia.inference.ml.azure.com/score"
    api_key = getenv('MOBILENET_HEIGHT_KEY')
    headers = {'Content-Type':'application/octer-stream', 'Authorization':('Bearer '+ api_key), 'azureml-model-deployment': 'blue'}
    pickled_input_data = pickle.dumps(np.array(depthmaps))
    async with session.post(scoring_uri, data=pickled_input_data, headers=headers) as resp:
        print(resp.status)
        data = await resp.read()
        return json.loads(data)
