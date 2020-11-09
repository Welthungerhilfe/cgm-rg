#
# Child Growth Monitor - Free Software for Zero Hunger
# Copyright (c) 2019 Tristan Behrens <tristan@ai-guru.de> for Welthungerhilfe
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

import os
import sys
import json
import numpy as np

from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication

import config

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"

#TODO generate the config file
#ws = Workspace.from_config('./ws_config.json')

#TODO load the weights of passed model and generate results for passed pointclouds

def get_predictions(pointcloud_numpy, model_id, service):
    #ws = Workspace.from_config('~/PythonCode/prod_ws_config.json')
    
    sp = ServicePrincipalAuthentication(tenant_id=config.TENANT_ID,
                                    service_principal_id=config.SP_ID,
                                    service_principal_password=config.SP_PASSWD)
    
    ws = Workspace.get(name="cgm-azureml-prod",
                   auth=sp,
                   subscription_id=config.SUB_ID)
    
    service = ws.webservices[service]
    pointcloud_json = json.dumps({'data': pointcloud_numpy.tolist()})
    predictions = service.run(input_data = pointcloud_json)
    return predictions

def get_pose_prediction(image, service):
    #ws = Workspace.from_config('~/PythonCode/prod_ws_config.json')
    
    sp = ServicePrincipalAuthentication(tenant_id=config.TENANT_ID,
                                    service_principal_id=config.SP_ID,
                                    service_principal_password=config.SP_PASSWD)
    
    ws = Workspace.get(name="cgm-azureml-prod",
                   auth=sp,
                   subscription_id=config.SUB_ID)
    
    service = ws.webservices[service]
    input_image_json = json.dumps({'input_image': image.tolist()})
    predictions = service.run(input_data = input_image_json)
    return predictions


def get_predictions_2(pointcloud_numpy, model_id, service_name):

    #ws = Workspace.from_config('~/PythonCode/prod_ws_config.json')
    
    sp = ServicePrincipalAuthentication(tenant_id=config.TENANT_ID,
                                    service_principal_id=config.SP_ID,
                                    service_principal_password=config.SP_PASSWD)
    
    ws = Workspace.get(name="cgm-azureml-prod",
                   auth=sp,
                   subscription_id=config.SUB_ID)
    
    service = ws.webservices[service_name]
    max_size = 20
    predictions = []
    for i in range(0, len(pointcloud_numpy), max_size):
        print(i, min(len(pointcloud_numpy), i+max_size))
        pcd_numpy = pointcloud_numpy[i:i+max_size]
        #service = ws.webservices[service_name]
        pointcloud_json = json.dumps({'data': pcd_numpy.tolist()})
        prediction = service.run(input_data = pointcloud_json)
        predictions += prediction

    return predictions