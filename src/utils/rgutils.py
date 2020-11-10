import os
import json
import dbutils
from datetime import datetime

def get_measure_insert(measure_id, model_id, scan_step, json_value, table, constant):
    '''
    Build a Insert query for measure result table
    '''
    measure_mapping = {}
    measure_mapping['measure_id'] = measure_id
    measure_mapping['model_id'] = model_id
    if 'height' in model_id:
        measure_mapping['key'] = 'height_' + scan_step
    elif 'weight' in model_id:
        measure_mapping['key'] = 'weight_' + scan_step
    for result_k in json_value.keys():
        json_value[result_k] = float(json_value[result_k])
    confidence_value = 1 - ((json_value["max"] - json_value["min"])/constant)
    if confidence_value < 0:
        confidence_value = 0
    measure_mapping['confidence_value'] = confidence_value
    measure_mapping['float_value'] = json_value['mean']
    json_value['timestamp'] = int(datetime.now().timestamp()*1000)
    json_value = json.dumps(json_value)
    measure_mapping['json_value'] = json_value
    keys = []
    values = []
    for key in measure_mapping.keys():
        keys.append(key)
        values.append(measure_mapping[key])
    insert_statement = dbutils.create_insert_statement(table, keys, values, True, True)
    #print(measure_mapping)

    return insert_statement

def get_artifact_insert(artifact_id, model_id, scan_step, prediction, table):
    '''
    Build a Insert query for Artifact result table
    '''
    artifact_mapping = {}
    artifact_mapping['model_id'] = model_id
    artifact_mapping['artifact_id'] = artifact_id
    if 'height' in model_id:
        artifact_mapping['key'] = 'height_' + scan_step
    elif 'weight' in model_id:
        artifact_mapping['key'] = 'weight_' + scan_step
    artifact_mapping['float_value'] = prediction
    keys = []
    values = []
    for key in artifact_mapping.keys():
        keys.append(key)
        values.append(artifact_mapping[key])
    insert_statement = dbutils.create_insert_statement(table, keys, values, True, True)
    #print(artifact_mapping)

    return insert_statement

def upload_result(filename, measure_id, db_connector):
    height_constant = 2.5
    weight_constant = 1.2
    with open(filename) as json_file:
        json_data = json.load(json_file)
        #for model_result in json_data["model_results"]:
        model_result = json_data['model_result']
        flag = 1
        table = "measure_result"
        if 'height' in model_result['model_id']:
            result_key = "height"
        elif 'weight' in model_result['model_id']:
            result_key = "weight"
        model_id = model_result['model_id']

        front_scan_results = model_result["measure_result"]["front_scan"]
        back_scan_results = model_result["measure_result"]["back_scan"]
        threesixty_scan_results = model_result["measure_result"]["threesixty_scan"]

        table = "measure_result"
        if result_key == 'height':
            front_insert_statement = get_measure_insert(measure_id, model_id, "front", front_scan_results, table, height_constant)
            back_insert_statement = get_measure_insert(measure_id, model_id, "back", back_scan_results, table, height_constant)
            threesixty_insert_statement = get_measure_insert(measure_id, model_id, "360", threesixty_scan_results, table, height_constant)
        elif result_key == 'weight':
            front_insert_statement = get_measure_insert(measure_id, model_id, "front", front_scan_results, table, weight_constant)
            back_insert_statement = get_measure_insert(measure_id, model_id, "back", back_scan_results, table, weight_constant)
            threesixty_insert_statement = get_measure_insert(measure_id, model_id, "360", threesixty_scan_results, table, weight_constant)
            #print(front_insert_statement)
            #print(back_insert_statement)
            #print(threesixty_insert_statement)

        try:
            db_connector.execute(front_insert_statement)
            db_connector.execute(back_insert_statement)
            db_connector.execute(threesixty_insert_statement)
            print('successfully inserted data to {0} table for measure_id {1}'.format(table, measure_id))
        except Exception as error:
            print(error)
            flag = 0

        table = "artifact_result"

        for artifact_result in model_result["artifact_results"]["front_scan"]:
            insert_statement = get_artifact_insert(artifact_result["artifact_id"], model_id, "front", artifact_result["prediction"], table)
            try:
                db_connector.execute(insert_statement)
                print('successfully inserted data to {0} table for artifact_id {1}'.format(table, artifact_result["artifact_id"]))
            except Exception as error:
                print(error)
                flag = 0
                #print(insert_statement)

        for artifact_result in model_result["artifact_results"]["back_scan"]:
            insert_statement = get_artifact_insert(artifact_result["artifact_id"], model_id, "back", artifact_result["prediction"], table)
            try:
                db_connector.execute(insert_statement)
                print('successfully inserted data to {0} table for artifact_id {1}'.format(table, artifact_result["artifact_id"]))
            except Exception as error:
                print(error)
                flag = 0
                #print(insert_statement)

        for artifact_result in model_result["artifact_results"]["threesixty_scan"]:
            insert_statement = get_artifact_insert(artifact_result["artifact_id"], model_id, "360", artifact_result["prediction"], table)
            try:
                db_connector.execute(insert_statement)
                print('successfully inserted data to {0} table for artifact_id {1}'.format(table, artifact_result["artifact_id"]))
            except Exception as error:
                print(error)
                flag = 0
                #print(insert_statement)
    return flag


def get_measure_insert_dict(measure_id, model_id, scan_step, json_value, constant):
    measure_mapping = {}
    measure_mapping['measure_id'] = measure_id
    measure_mapping['model_id'] = model_id
    if 'height' in model_id:
        measure_mapping['key'] = 'height_' + scan_step
    elif 'weight' in model_id:
        measure_mapping['key'] = 'weight_' + scan_step
    for result_k in json_value.keys():
        json_value[result_k] = float(json_value[result_k])
    confidence_value = 1 - ((json_value["max"] - json_value["min"])/constant)
    if confidence_value < 0:
        confidence_value = 0
    measure_mapping['confidence_value'] = confidence_value
    measure_mapping['float_value'] = json_value['mean']
    #json_value = json.dumps(json_value)
    json_value['timestamp'] = int(datetime.now().timestamp()*1000)
    measure_mapping['json_value'] = str(json_value)
    print(measure_mapping['json_value'])
    #print(measure_mapping)
    return measure_mapping

def create_queue(queue_name, queue_service):
    '''
    Create a Queue service of name query_name
    '''
    try:
        queue_service.create_queue(queue_name)
        print("Created {0} queue".format(queue_name))
    except:
        print("Queue: '{0}' already exists".format(queue_name))


def upload_message(queue_service, queue_name, message, flag = 0):
    '''
    Upload a message to a Queue Service
    '''
    try:
        queue_service.put_message(queue_name, message)
        print("successfully uploaded message")
        flag = 1
    except Exception as error:
        #move_processed_data(json_path, destination_folder, queue_name, 'unprocessed')
        print(error)
        flag = 0
    return flag


def upload_to_queue(storage_account_name, json_path, db_connector):
    try:
        dir = os.path.expanduser("~/PythonCode/dbconnection.json")
        with open(dir, "r") as json_file:
            json_data = json.load(json_file)
            account_key = json_data["account_key"]
    except IOError:
        #logger.error("file not found or empty")
        print("file not found or empty")

    queue_service = QueueService(storage_account_name, account_key)

    height_constant = 2.5
    weight_constant = 1.2

    with open(json_path) as f:
        json_data = json.load(f)
        print(type(json_data))
        #manual_measure_id = json_data["scan"]["manual_measure_id"]
        measure_id = json_data["scan"]["measure_id"]
        qr_code = json_data["scan"]["qrcode"]

        queue_name = measure_id.split('_')[0] + "-measure-result"
        create_queue(queue_name, queue_service)

        #for model_result in json_data["model_results"]:
        model_result = json_data["model_result"]
        model_id = model_result['model_id']
        #if model_result['model_id'] == model_id:
        front_scan_results = model_result["measure_result"]["front_scan"]
        back_scan_results = model_result["measure_result"]["back_scan"]
        threesixty_scan_results = model_result["measure_result"]["threesixty_scan"]

        if 'height' in model_result['model_id']:
            result_key = "height"
        elif 'weight' in model_result['model_id']:
            result_key = "weight"

        if result_key == 'height':
            front_scan_data = get_measure_insert_dict(measure_id, model_id, "front", front_scan_results, height_constant)
            #print(front_scan_data)
            encoded_fcd = str(json.dumps(front_scan_data))
            flag = upload_message(queue_service, queue_name, encoded_fcd, flag = 0)
            if flag == 0:
                return flag
            back_scan_data = get_measure_insert_dict(measure_id, model_id, "back", back_scan_results, height_constant)
            #print(back_scan_data)
            encoded_bcd = str(json.dumps(back_scan_data))
            flag = upload_message(queue_service, queue_name, encoded_bcd, flag = 0)
            if flag == 0:
                return flag
            threesixty_scan_data = get_measure_insert_dict(measure_id, model_id, "360", threesixty_scan_results, height_constant)
            #print(threesixty_scan_data)
            encoded_tcd = str(json.dumps(threesixty_scan_data))
            flag = upload_message(queue_service, queue_name, encoded_tcd, flag = 0)
            if flag == 0:
                return flag
        elif result_key == 'weight':
            front_scan_data = get_measure_insert_dict(measure_id, model_id, "front", front_scan_results, weight_constant)
            #print(front_scan_data)
            encoded_fcd = str(json.dumps(front_scan_data))
            flag = upload_message(queue_service, queue_name, encoded_fcd, flag = 0)
            if flag == 0:
                return flag
            back_scan_data = get_measure_insert_dict(measure_id, model_id, "back", back_scan_results, weight_constant)
            #print(back_scan_data)
            encoded_bcd = str(json.dumps(back_scan_data))
            flag = upload_message(queue_service, queue_name, encoded_bcd, flag = 0)
            if flag == 0:
                return flag
            threesixty_scan_data = get_measure_insert_dict(measure_id, model_id, "360", threesixty_scan_results, weight_constant)
            #print(threesixty_scan_data)
            encoded_tcd = str(json.dumps(threesixty_scan_data))
            flag = upload_message(queue_service, queue_name, encoded_tcd, flag = 0)
            if flag == 0:
                return flag

    return flag

def depth_exists(id, connector):
    
    check_dataformat = "SELECT EXISTS (SELECT id FROM artifact WHERE measure_id='{}'".format(id[0]) + " and dataformat = 'depth');"
    exists = connector.execute(check_dataformat, fetch_all=True)
    return exists[0][0]