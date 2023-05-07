from azureml.core import Workspace, Webservice
from azureml.core.authentication import ServicePrincipalAuthentication
from os import getenv
import json
import logging
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import requests
import pickle


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


def requests_retry_session(
    retries=9,
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
