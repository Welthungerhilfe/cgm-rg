# This function is not intended to be invoked directly. Instead it will be
# triggered by an HTTP starter function.
# Before running this sample, please:
# - create a Durable activity function (default name is "Hello")
# - create a Durable HTTP starter function
# - add azure-functions-durable to requirements.txt
# - run pip install -r requirements.txt

import logging
import json

import azure.functions as func
import azure.durable_functions as df

from utils.rest_api import CgmApi
import height_plaincnn
import Weight_plaincnn
import height_efficient_former
import height_mobilenet


height_func_name = height_plaincnn.__name__
weight_func_name = Weight_plaincnn.__name__
eff_height_func_name = height_efficient_former.__name__
mobilenet_height_func_name = height_mobilenet.__name__


rgb_format = ["rgb", "image/jpeg"]
depth_format = ["depth", "application/zip"]

cgm_api = CgmApi()


def get_scan_by_format(artifacts, file_format):
    return [artifact for artifact in artifacts if artifact['format'] in file_format]


def orchestrator_function(context: df.DurableOrchestrationContext):
    scan_id = context.get_input()
    scan_metadata = cgm_api.get_scan_metadata(scan_id)
    # print(scan_metadata['type'])
    # artifacts = scan_metadata['artifacts']
    # version = scan_metadata['version']
    # scan_type = scan_metadata['type']
    workflows = cgm_api.get_workflows()

    payload = {
        "scan_metadata": scan_metadata,
        "workflows": workflows
    }

    # h_result = yield context.call_activity(height_func_name, scan_metadata)
    workflow_functions = [height_func_name, weight_func_name, eff_height_func_name, mobilenet_height_func_name]


    transfer_tasks = [ context.call_activity(func_name, payload) for func_name in workflow_functions ]
    # logging.info(f"Executing transfer tasks for {scan_type}")
    yield context.task_all(transfer_tasks)
    # for task in transfer_tasks:
    #     yield task

    return json.dumps({"success" : True})

    # result1 = yield context.call_activity('Hello', "Tokyo")
    # result2 = yield context.call_activity('Hello', "Seattle")
    # result3 = yield context.call_activity('Hello', "London")
    # return [result1, result2, result3]

main = df.Orchestrator.create(orchestrator_function)
