from azureml.core import Workspace, Webservice
from azureml.core.authentication import ServicePrincipalAuthentication
from os import getenv
import json
import logging
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import requests
import pickle
import cv2
from io import BytesIO
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials


sp = ServicePrincipalAuthentication(
    tenant_id=getenv('TENANT_ID'),
    service_principal_id=getenv('SP_ID'),
    service_principal_password=getenv('SP_PASSWD')
)

workspace = Workspace(
    subscription_id=getenv('SUB_ID'),
    resource_group=getenv('RESOURCE_GROUP'),
    workspace_name=getenv('WORKSPACE_NAME'),
    auth=sp
)

STANDING_SCAN_TYPE = [100, 101, 102]
LAYING_SCAN_TYPE = [200, 201, 202]

face_api_url = f"{getenv('MS_FACE_API_ENDPOINT')}/face/v1.0/detect"

ms_face_api_headers = {
    'Content-Type': 'application/octet-stream',
    'Ocp-Apim-Subscription-Key': getenv('MS_FACE_API_KEY'),
}

def requests_retry_session(
    retries=5,
    backoff_factor=2,
    status_forcelist=(500, 502, 503, 504),
    session=None,
):
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        method_whitelist=frozenset(['GET', 'POST'])
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


def get_json_prediction(pickled_input_data, service_name):
    service = Webservice(workspace=workspace, name=service_name)
    scoring_uri = service.scoring_uri
    
    # data = {
    #      "data":img.tolist()
    # }
    # data = json.dumps(data)
    # headers = {"Content-Type": "application/json"}
    headers = {"Content-Type": "application/octer-stream"}
    response = requests_retry_session().post(scoring_uri, data=pickled_input_data, headers=headers)
    logging.info(f"status code is {response.status_code}")
    predictions = response.json()

    return predictions


def get_raw_prediction(pickled_input_data, service_name):
    service = Webservice(workspace=workspace, name=service_name)
    scoring_uri = service.scoring_uri
    
    # data = {
    #      "data":img.tolist()
    # }
    # data = json.dumps(data)
    # headers = {"Content-Type": "application/json"}
    headers = {"Content-Type": "application/octer-stream"}
    response = requests_retry_session().post(scoring_uri, data=pickled_input_data, headers=headers)
    logging.info(f"status code is {response.status_code}")

    return response.content


def get_pose_prediction(pickled_data):
    pose_score_uri = 'https://pose-endpoint.centralindia.inference.ml.azure.com/score'
    api_key = getenv("POSE_ENDPOINT_KEY")
    headers_up = {'Content-Type':'application/octer-stream', 'Authorization':('Bearer '+ api_key), 'azureml-model-deployment': 'blue'}
    response = requests_retry_session().post(pose_score_uri, data=pickled_data, headers=headers_up)
    logging.info(f"status code is {response.status_code}")
    predictions = response.json()

    return predictions


def get_blur_prediction(pickled_data):
    blur_score_uri = 'https://blur-endpoint.centralindia.inference.ml.azure.com/score'
    api_key = getenv("BLUR_ENDPOINT_KEY")
    headers_up = {'Content-Type':'application/octer-stream', 'Authorization':('Bearer '+ api_key), 'azureml-model-deployment': 'blue'}
    response = requests_retry_session().post(blur_score_uri, data=pickled_data, headers=headers_up)
    logging.info(f"status code is {response.status_code}")
    # predictions = response.json()

    return pickle.loads(response.content)


def ms_face_api(input_image, scan_type):
    _, bin_file = cv2.imencode('.JPEG', input_image)
    bin_file = bin_file.tobytes()
    params = {
        'returnFaceId': 'false',  # Do not return face IDs
        'returnFaceLandmarks': 'false',  # Do not return face landmarks
        'returnFaceAttributes': '',  # Exclude attributes
        'detectionModel': 'detection_03',
    }
    response = requests.post(face_api_url, headers=ms_face_api_headers, params=params, data=bin_file)
    detected_faces = response.json()
    for face in detected_faces:
        fr = face['faceRectangle']
        origin = (fr['left'], fr['top'])
        input_image = blur_img(input_image,fr['height'],fr['width'],origin)
    # if len(detected_faces) > 0:
    if scan_type in STANDING_SCAN_TYPE:
        image = cv2.rotate(input_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif scan_type in LAYING_SCAN_TYPE:
        image = cv2.rotate(input_image, cv2.ROTATE_90_CLOCKWISE)

    return image, len(detected_faces)


def blur_img(im,height,width,origin):
    #Create ROI
    h,w = height,width
    x,y = origin[0],origin[1]
    roi = im[int(y):int(y)+int(h),int(x):int(x)+int(w)]
    #Blur image in ROI
    blurred_img = cv2.GaussianBlur(roi,(91,91),0)
    #Add blur to the overall img
    im[int(y):int(y)+int(h),int(x):int(x)+int(w)] = blurred_img
    return im
