import logging
import json
import typing

import azure.functions as func
from azure.durable_functions import DurableOrchestrationClient
import func_rg_orchestrator


orchestration_func_name = func_rg_orchestrator.__name__


async def main(msg: func.QueueMessage, starter: str, msg2: func.Out[typing.List[str]]) -> None:
    try:
        logging.info('Python queue trigger function processed a queue item: %s',
                    msg.get_body().decode('utf-8'))
        message_received = json.loads(msg.get_body().decode('utf-8'))
        client = DurableOrchestrationClient(starter)
        instance_id = await client.start_new(orchestration_func_name, None, message_received)
        logging.info(f"Started {orchestration_func_name} with ID = '{instance_id}', "
                    f"Trigger: '{message_received}'")
        output_messages = [json.dumps({'scan_id': s_id}) for s_id in message_received['scan_ids']]
        msg2.set(output_messages)
        # return client.create_check_status_response(msg, instance_id)
    except Exception:
        logging.exception("Error starting ETL orchestration")
        raise
