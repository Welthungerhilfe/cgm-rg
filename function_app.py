import azure.functions as func
import datetime
import json
import logging
from typing import Any

from rg.entry import run_rg, run_mean_rg, generate_new_visualizations

app = func.FunctionApp()


@app.queue_trigger(arg_name="azqueue", queue_name="result-generation",
                               connection="QueueConnectionString")
@app.queue_output(arg_name="msg", 
                  queue_name="mean-result-generation", 
                  connection="QueueConnectionString")
async def rg_trigger(azqueue: func.QueueMessage, msg: func.Out[str]):
    logging.info('Python Queue trigger processed a message: %s',
                azqueue.get_body().decode('utf-8'))
    logging.info("test")
    message_received = json.loads(azqueue.get_body().decode('utf-8'))
    scan_ids = message_received['scan_ids']
    await run_rg(scan_ids)

    msg.set(message_received)


@app.queue_trigger(arg_name="azqueue", queue_name="mean-result-generation",
                               connection="QueueConnectionString")
async def calculate_mean_result(azqueue: func.QueueMessage):
    logging.info('Python Queue trigger processed a message: %s',
                azqueue.get_body().decode('utf-8'))
    message_received = json.loads(azqueue.get_body().decode('utf-8'))
    scan_ids = message_received['scan_ids']
    await run_mean_rg(scan_ids)
    output_message = [{"scan_id": scan_id} for scan_id in scan_ids]


# @app.queue_trigger(arg_name="azqueue", queue_name="mean-result-generation",
#                                connection="QueueConnectionString")
# @app.queue_output(arg_name="msg", 
#                   queue_name="new-visualization", 
#                   connection="QueueConnectionString")
# async def calculate_mean_result(azqueue: func.QueueMessage, msg: func.Out[list[dict[str, Any]]]):
#     logging.info('Python Queue trigger processed a message: %s',
#                 azqueue.get_body().decode('utf-8'))
#     message_received = json.loads(azqueue.get_body().decode('utf-8'))
#     scan_ids = message_received['scan_ids']
#     await run_mean_rg(scan_ids)
#     output_message = [{"scan_id": scan_id} for scan_id in scan_ids]
#     msg.set(output_message)


# @app.queue_trigger(arg_name="azqueue", queue_name="new_visualization",
#                                connection="QueueConnectionString") 
# async def new_visualizations(azqueue: func.QueueMessage):
#     logging.info('Python Queue trigger processed a message: %s',
#                 azqueue.get_body().decode('utf-8'))
#     message_received = json.loads(azqueue.get_body().decode('utf-8'))
#     scan_id = message_received['scan_id']
#     await generate_new_visualizations(scan_id)
