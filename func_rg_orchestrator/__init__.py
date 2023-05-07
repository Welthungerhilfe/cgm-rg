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

import func_rg_sub_orchestrator


sub_orchestration_func_name = func_rg_sub_orchestrator.__name__


def orchestrator_function(context: df.DurableOrchestrationContext):
    message_received = context.get_input()
    provisioning_tasks = []
    for id_, scan_id in enumerate(message_received['scan_ids']):
        child_id = f"{context.instance_id}:{id_}"
        provision_task = context.call_sub_orchestrator(sub_orchestration_func_name, scan_id, child_id)
        provisioning_tasks.append(provision_task)    
    yield context.task_all(provisioning_tasks)

main = df.Orchestrator.create(orchestrator_function)
