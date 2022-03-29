import logging
import json
import azure.functions as func
import requests
from os import getenv
from azure.mgmt.datafactory import DataFactoryManagementClient
from azure.common.credentials import ServicePrincipalCredentials
from utils.rest_api import MlApi

ml_api = MlApi()

def main(msg: func.QueueMessage,
        context: func.Context) -> None:
    message_received = json.loads(msg.get_body().decode('utf-8'))
    # response = requests.get(url + f"/api/scans?scan_id={message_received['scan_id']}", headers=headers)
    scan_metadata = ml_api.get_scan_metadata(message_received['scan_id'])

    # scan_metadata = { "scan_metadata": {"artifacts": [{"file": "330d8c60-72d1-11ec-95b8-bf53fe51e1e9", "format": "calibration", "id": "49b25cc0-72d1-11ec-95b8-4bc855daf97a", "order": 0, "timestamp": "2022-01-11T11:25:38Z"}, {"file": "467d7080-72d1-11ec-95b8-a7f0d090c085", "format": "rgb", "id": "49b25cc1-72d1-11ec-95b8-6f6f9ecf5f7c", "order": 1, "timestamp": "2022-01-11T11:26:11Z"}, {"file": "46775600-72d1-11ec-95b8-23cd00e57a1c", "format": "depth", "id": "49b25cc2-72d1-11ec-95b8-abd92e4e7399", "order": 1, "timestamp": "2022-01-11T11:26:11Z"}, {"file": "469a6e60-72d1-11ec-95b8-d37b7d4f11ce", "format": "rgb", "id": "49b25cc3-72d1-11ec-95b8-37b5a9196857", "order": 2, "timestamp": "2022-01-11T11:26:11Z"}, {"file": "468511a0-72d1-11ec-95b8-ff7b93e11a6f", "format": "depth", "id": "49b25cc4-72d1-11ec-95b8-97cef160ff40", "order": 2, "timestamp": "2022-01-11T11:26:11Z"}, {"file": "47bddfc0-72d1-11ec-95b8-dbd1eac94b41", "format": "rgb", "id": "49b25cc5-72d1-11ec-95b8-9b3e3e18710a", "order": 3, "timestamp": "2022-01-11T11:26:13Z"}, {"file": "47b63ea0-72d1-11ec-95b8-cf85f0f79190", "format": "depth", "id": "49b25cc6-72d1-11ec-95b8-7706d54dfb6e", "order": 3, "timestamp": "2022-01-11T11:26:13Z"}, {"file": "47ad16e0-72d1-11ec-95b8-2330918f1950", "format": "rgb", "id": "49b25cc7-72d1-11ec-95b8-ef2f6956ec01", "order": 4, "timestamp": "2022-01-11T11:26:13Z"}, {"file": "47518960-72d1-11ec-95b8-bba05d74e532", "format": "depth", "id": "49b3e360-72d1-11ec-95b8-074129d03cb2", "order": 4, "timestamp": "2022-01-11T11:26:13Z"}, {"file": "483c85a0-72d1-11ec-95b8-937b048d175f", "format": "rgb", "id": "49b3e361-72d1-11ec-95b8-ebf9c3306a00", "order": 5, "timestamp": "2022-01-11T11:26:14Z"}, {"file": "47592a80-72d1-11ec-95b8-0bded71367c4", "format": "depth", "id": "49b3e362-72d1-11ec-95b8-1775cc446c90", "order": 5, "timestamp": "2022-01-11T11:26:13Z"}, {"file": "48937f40-72d1-11ec-95b8-6b106feaf9eb", "format": "rgb", "id": "49b3e363-72d1-11ec-95b8-9facf4b1d566", "order": 6, "timestamp": "2022-01-11T11:26:14Z"}, {"file": "48411980-72d1-11ec-95b8-e771b84140b6", "format": "depth", "id": "49b3e364-72d1-11ec-95b8-93b11a2e594d", "order": 6, "timestamp": "2022-01-11T11:26:14Z"}, {"file": "48a2c180-72d1-11ec-95b8-e33640593ca0", "format": "rgb", "id": "49b3e365-72d1-11ec-95b8-830a02da52ec", "order": 7, "timestamp": "2022-01-11T11:26:15Z"}, {"file": "489b2060-72d1-11ec-95b8-174f65579cf4", "format": "depth", "id": "49b3e366-72d1-11ec-95b8-fb1ae2d83d1e", "order": 7, "timestamp": "2022-01-11T11:26:14Z"}, {"file": "492c15c0-72d1-11ec-95b8-2322119972d1", "format": "rgb", "id": "49b6f0a0-72d1-11ec-95b8-f7bca20d9d2e", "order": 8, "timestamp": "2022-01-11T11:26:15Z"}, {"file": "48f3a0a0-72d1-11ec-95b8-cf6a4e4b2c7e", "format": "depth", "id": "49b6f0a1-72d1-11ec-95b8-3b3728bf5c1d", "order": 8, "timestamp": "2022-01-11T11:26:15Z"}, {"file": "495e7060-72d1-11ec-95b8-771b70281f92", "format": "rgb", "id": "49b6f0a2-72d1-11ec-95b8-b7edd8fad077", "order": 9, "timestamp": "2022-01-11T11:26:16Z"}, {"file": "494913a0-72d1-11ec-95b8-6bd98110efef", "format": "depth", "id": "49b9fde0-72d1-11ec-95b8-cf765a83ea18", "order": 9, "timestamp": "2022-01-11T11:26:16Z"}], "device_info": {"model": "HUAWEI VOG-L29"}, "id": "49af4f80-72d1-11ec-95b8-7b82e3646e48", "location": {"lat": 21.4971267, "lng": 70.1431799}, "person": "266aaf60-72d1-11ec-95b8-a73e1e2bc319", "scan_end": "2022-01-11T11:25:36Z", "scan_start": "2022-01-11T11:25:18Z", "type": 102, "version": "v1.2.0"} }

    subscription_id = getenv("SUBSCRIPTION_ID")
    # credentials = ClientSecretCredential(client_id=getenv("CLIENT_ID"), client_secret=getenv("SECRET"), tenant_id=getenv("TENANT_ID"))
    credentials = ServicePrincipalCredentials(client_id=getenv("CLIENT_ID"), secret=getenv("SECRET"), tenant=getenv("TENANT_ID")) #To login with serv ppal
    adf_client = DataFactoryManagementClient(credentials, subscription_id)

    rg_name = getenv("RESOURCE_GROUP_NAME")
    df_name = getenv("DATA_FACTORY_NAME")
    p_name = getenv("PIPELINE_NAME")

    params = {
        "input":{
            "scan_metadata":scan_metadata,
            "workflow_name":message_received['workflow_name'],
            "workflow_version":message_received['workflow_version'],
            "service_name":message_received['service_name']
        }
    }
    logging.info(f"passed parameters are {params}")
    adf_client.pipelines.create_run(
        resource_group_name = rg_name, 
        factory_name = df_name, 
        pipeline_name = p_name, 
        parameters = params
    )
