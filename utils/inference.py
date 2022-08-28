from azureml.core import Workspace, Webservice
from azureml.core.authentication import ServicePrincipalAuthentication
from os import getenv
import json
import logging
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import requests
import numpy

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


def get_standing_laying_prediction(img, service_name):
    service = Webservice(workspace=workspace, name=service_name)
    scoring_uri = service.scoring_uri
    
    data = {
         "data":img.numpy().tolist()
    }
    data = json.dumps(data)
    headers = {"Content-Type": "application/json"}
    response = requests_retry_session().post(scoring_uri, data=data, headers=headers)
    logging.info(f"predictions {response.content} and status code is {response.status_code}")
    predictions = response.json()

    return predictions


def get_height_prediction(depthmaps, service_name):
    service = Webservice(workspace=workspace, name=service_name)
    scoring_uri = service.scoring_uri
    
    data = {
         "data":depthmaps.tolist()
    }
    data = json.dumps(data)
    headers = {"Content-Type": "application/json"}
    response = requests_retry_session().post(scoring_uri, data=data, headers=headers)
    logging.info(f"predictions {response.content} and status code is {response.status_code}")
    predictions = response.json()

    return predictions


def get_weight_prediction(depthmaps, service_name):
    service = Webservice(workspace=workspace, name=service_name)
    scoring_uri = service.scoring_uri

    data = {
         "data":depthmaps.tolist()
    }
    data = json.dumps(data)
    headers = {"Content-Type": "application/json"}
    response = requests_retry_session().post(scoring_uri, data=data, headers=headers)
    logging.info(f"predictions {response.content} and status code is {response.status_code}")
    predictions = response.json()

    return predictions