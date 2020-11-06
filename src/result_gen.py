import os
import glob
import json
import shutil
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import time
import dbutils
import sys
#sys.path.append(os.path.expanduser('~/cgm-ml'))
#from cgmcore import modelutils, utils
#from azure.storage.queue import QueueService
from bunch import Bunch
import random
import numpy as np
import generate_result
from datetime import datetime
import preprocessing
from pyntcloud import PyntCloud
import download_and_upload_blobs

def load_pcd_as_ndarray(pcd_path):
    """
    Loads a PCD-file. Yields a numpy-array.
    """
    pointcloud = PyntCloud.from_file(pcd_path)
    values = pointcloud.points.values
    return values


def subsample_pointcloud(pointcloud, target_size, subsampling_method="random", dimensions=[0, 1, 2]):
    """
    Yields a subsampled pointcloud.
    These subsamplinge modes are available:
    - "random": Yields a random subset. Multiple occurrences of a single point are possible.
    - "first": Yields the first n points
    - "sequential_skip": Attempts to keep the order of the points intact, might skip some elements if the pointcloud is too big. E.g. every second point is skipped.
    Note: All methods ensure that the target_size is met. If necessary zeroes are appended.
    """

    # Check if the requested subsampling method is all right.
    possible_subsampling_methods = ["random", "first", "sequential_skip"]
    assert subsampling_method in possible_subsampling_methods, "Subsampling method {} not in {}".format(subsampling_method, possible_subsampling_methods)

    # Random subsampling.
    if subsampling_method == "random":
        indices = np.arange(0, pointcloud.shape[0])
        indices = np.random.choice(indices, target_size)
        result = pointcloud[indices]

    elif subsampling_method == "first":
        result = np.zeros((target_size, pointcloud.shape[1]), dtype="float32")
        result[:len(pointcloud),:] = pointcloud[:target_size]

    elif subsampling_method == "sequential_skip":
        result = np.zeros((target_size, pointcloud.shape[1]), dtype="float32")
        skip = max(1, round(len(pointcloud) / target_size))
        pointcloud_skipped = pointcloud[::skip,:]
        result = np.zeros((target_size, pointcloud.shape[1]), dtype="float32")
        result[:len(pointcloud_skipped),:] = pointcloud_skipped[:target_size]

    return result[:,dimensions]


def preprocess_pointcloud(pointcloud, subsample_size, channels):
    if subsample_size is not None:
        skip = max(1, round(len(pointcloud) / subsample_size))
        pointcloud_skipped = pointcloud[::skip,:]
        result = np.zeros((subsample_size, pointcloud.shape[1]), dtype="float32")
        result[:len(pointcloud_skipped),:] = pointcloud_skipped[:subsample_size]
        pointcloud = result
    if channels is not None:
        pointcloud = pointcloud[:,channels]
    return pointcloud.astype("float32")

def pcd_to_ndarray(pcd_paths, input_shape):
    pointclouds = []
    for pcd_path in pcd_paths:
        print(pcd_path)
        try:
            pointcloud = load_pcd_as_ndarray(pcd_path)
        except Exception as error:
            print(error)
            continue
        pointcloud = subsample_pointcloud(
            pointcloud,
            target_size=input_shape[0],
            subsampling_method="sequential_skip")
        pointclouds.append(pointcloud)
    pointclouds = np.array(pointclouds)
    #predictions = model.predict(pointclouds)
    #return predictions
    return pointclouds

def pcd_processing_gapnet(pcd_paths):
    pointclouds = []
    for pcd_path in pcd_paths:
        try:
            pointcloud = load_pcd_as_ndarray(pcd_path)
        except Exception as error:
            print(error)
            continue
        pointcloud = [preprocess_pointcloud(pointcloud, 1024, list(range(3)))]
        pointclouds.append(pointcloud)
    pointclouds = np.array(pointclouds)
    pointclouds = pointclouds.reshape((-1, 1024, 3))

    return pointclouds


def pcd_to_depthmap(paths, calibration):
    depthmaps = []
    for path in paths:
        depthmap = preprocessing.lenovo_pcd2depth(path, calibration)
        if depthmap is not None:
            depthmap = preprocessing.preprocess(depthmap)
            depthmaps.append(depthmap)
    
    depthmaps = np.array(depthmaps)
    
    return depthmaps

def depthmap_to_pcd(paths, calibration, preprocessing_type, input_shape = []):
    pcds = []
    for path in paths:
        data, width, height, depthScale, maxConfidence = preprocessing.load_depth(path)
        pcd = preprocessing.getPCD(path, calibration, data, maxConfidence, depthScale)
        
        if pcd.shape[0] == 0:
            continue
        
        if preprocessing_type == 'pointnet':
            pcd = subsample_pointcloud(
                pcd,
                target_size=input_shape[0],
                subsampling_method="sequential_skip")
        elif preprocessing_type == 'gapnet':
            pcd = [preprocess_pointcloud(pcd, 1024, list(range(3)))]
        pcds.append(pcd)
    
    pcds = np.array(pcds)

    if preprocessing_type == 'gapnet':
        pcds = pcds.reshape((-1, 1024, 3))

    return pcds


def get_depthmaps(paths):
    depthmaps = []
    for path in paths:
        data, width, height, depthScale, maxConfidence = preprocessing.load_depth(path)
        depthmap,height, width = preprocessing.prepare_depthmap(data, width, height, depthScale)
        #print(height, width)
        depthmap = preprocessing.preprocess(depthmap)
        #print(depthmap.shape)
        depthmaps.append(depthmap)
    
    depthmaps = np.array(depthmaps)
    
    return depthmaps

def get_stats(predictions):
    res = Bunch()
    res.mean = str(np.mean(predictions))
    res.min = str(np.min(predictions))
    res.max = str(np.max(predictions))
    res.std = str(np.std(predictions))
    return res

def get_artifact_result(artifacts, predictions):
    results = []
    for artifact, prediction in zip(artifacts, predictions):
        artifact_result = Bunch()
        artifact_result.artifact_id = artifact[0]
        artifact_result.path = '/'.join(artifact[3].split('/')[4:])
        artifact_result.prediction = str(prediction[0])
        results.append(artifact_result)
    return results

def move_processed_data(json_path, table, state):
    #destination_folder1 = destination_folder + "{0}_{1}".format(state, table)
    destination_folder = '/'.join(json_path.split('/')[:-3])
    destination_folder = destination_folder + "/" + "{0}_{1}".format(state, table)
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    try:
        filename = json_path.split('/')[-1]
        shutil.move(json_path, os.path.join(destination_folder, filename))
    except Exception as error:
        print(error)

def get_measure_insert(measure_id, model_id, scan_step, json_value, table, constant):
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

def upload_message(queue_service, queue_name, message, flag = 0):

    try:
        queue_service.put_message(queue_name, message)
        print("successfully uploaded message")
        flag = 1
    except Exception as error:
        #move_processed_data(json_path, destination_folder, queue_name, 'unprocessed')
        print(error)
        flag = 0
    return flag

def create_queue(queue_name, queue_service):
    try:
        queue_service.create_queue(queue_name)
        print("Created {0} queue".format(queue_name))
    except:
        print("Queue: '{0}' already exists".format(queue_name))

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

def get_file_name(destination_folder, model_id, qr_code):
    filename = '{0}-{1}-{2}-{3}.json'.format(model_id, qr_code,random.randint(10000, 99999))
    folder = '{0}/{1}/'.format(destination_folder, model_id)
    return filename, folder


def depth_exists(id, connector):
    
    check_dataformat = "SELECT EXISTS (SELECT id FROM artifact WHERE measure_id='{}'".format(id) + " and dataformat = 'depth');"
    exists = connector.execute(check_dataformat, fetch_all=True)
    
    return exists[0][0]

def main():
    #if len(sys.argv) != 7:
    #    print("Please provide destination folder and database connection file and storage account name and model_id.")
    #    exit(1)

    #destination_folder = str(sys.argv[1])
    #db_connection_file = str(sys.argv[2])
    #storage_account_name = str(sys.argv[3])
    #model_id = str(sys.argv[4])
    #service = str(sys.argv[5])
    #calibration_file = str(sys.argv[6])
    #container_name = str(sys.argv[7])

    model_id = "q3_depthmap_height_run_01"
    service = "q3-depthmap-height-run-01-ci"
    container_name = "scans"
    destination_folder = '.'

    #calibration = preprocessing.parseCalibration(calibration_file)
    preprocessing.setWidth(int(240 * 0.75))
    preprocessing.setHeight(int(180 * 0.75))

    main_connector = dbutils.connect_to_main_database()

    #TODO check if model_id is in active state
    check_model = "select (json_metadata->>'active')::BOOLEAN from model where id = '{}';".format(model_id)
    active = main_connector.execute(check_model, fetch_all=True)

    if not active[0][0]:
        print("model {0} is not active.... exiting".format(model_id))
        exit(1)

    select_measures = "select measure_id from artifact where not exists (SELECT measure_id from measure_result WHERE measure_id=artifact.measure_id and model_id = '{}')".format(model_id) + " and dataformat in ('pcd', 'depth') group by measure_id having count(case when substring(substring(storage_path from '_[0-9]\d\d_') from '[0-9]\d\d') in ('100', '104', '200') then 1 end) > 4 and count(case when substring(substring(storage_path from '_[0-9]\d\d_') from '[0-9]\d\d') in ('102', '110', '202') then 1 end) >4 and count(case when substring(substring(storage_path from '_[0-9]\d\d_') from '[0-9]\d\d') in ('101', '107', '201') then 1 end) > 4;"
    measure_id = main_connector.execute(select_measures, fetch_all=True)

    replace_path = 'qrcode/'

    measure_id = [('24675719efe635df_measure_1604424965938_dw2WjWUdv6JnQn37',), ('1e7a0527a2e148ec_measure_1603966156661_H8OJeVlOykvIkvXT',)]

    print(len(measure_id))

    for id in measure_id:

        #if id[0] not in ['c99eac07f31db1e7_measure_1574697616130_aZ9NT80CxjnRKcIR', 'c99eac07f31db1e7_measure_1579610075909_icDn7feQYj9NE7ng', 'c99eac07f31db1e7_measure_1579610268433_V2zLDTwmTr2Qfdp2', 'c99eac07f31db1e7_measure_1579619585972_CAPpdAg2ysyVK7fW', 'c99eac07f31db1e7_measure_1579620308013_mHdUzpwpI9xW5g2t']:
            #print("nope")
            #continue
        
        if depth_exists(id[0], main_connector):
            dataformat = 'depth'
        else:
            dataformat = 'pcd'

        get_artifacts = "SELECT id, qr_code, create_timestamp, replace('{}'".format(replace_path)
        get_artifacts += " || split_part(storage_path,'/storage/emulated/0/Child Growth Monitor Scanner App/', 2), 'measurements', 'measure')"
        get_artifacts += "from artifact" # where dataformat = '{}'".format(dataformat)
        get_artifacts += " where measure_id = '{}';".format(id[0])
        artifacts = main_connector.execute(get_artifacts, fetch_all=True)
        print(len(artifacts))
        
        if len(artifacts) < 15:
            print("not enough data to calculate results measure id {0}".format(id[0]))
            continue
        print("enough data to calculate results measure id {0}".format(id[0]))
        artifact_list = [list(artifact) for artifact in artifacts]
        
        files = [artifact[3] for artifact in artifact_list]
        download_and_upload_blobs.download_blobs(files, container_name)

        for file in files:
            if "camera_calibration.txt" in file:
                calibration_file = file
        
        calibration = preprocessing.parseCalibration(calibration_file)

        artifact_present = []
        for artifact in artifact_list:
            if os.path.isfile(artifact[3]):
                artifact_present.append(artifact)

        print("no of artifacts present ", len(artifact_present))

        qr_codes = [artifact[1] for artifact in artifact_present]
        qr_code = set(qr_codes)
        qr_code = list(qr_code)

        timestamp = [artifact[2] for artifact in artifact_present]
        timestamp = set(timestamp)
        timestamp = list(timestamp)

        #pcd_path for pcd_path in pcd_paths if pcd_path.split('/')[-1].split('_')[-2] in ['100','104','200']
        artifact_front = [artifact for artifact in artifact_present if artifact[3].split('/')[-1].split('_')[-2] in ['100','104','200'] or artifact[3].split('/')[-1].split('_')[-3] in ['100','104','200']]
        front_pcd_paths = [artifact[3] for artifact in artifact_front]
        #pcd_path for pcd_path in pcd_paths if pcd_path.split('/')[-1].split('_')[-2] in ['102','110','202']
        artifact_back = [artifact for artifact in artifact_present if artifact[3].split('/')[-1].split('_')[-2] in ['102','110','202'] or artifact[3].split('/')[-1].split('_')[-3] in ['102','110','202']]
        back_pcd_paths = [artifact[3] for artifact in artifact_back]
        #pcd_path for pcd_path in pcd_paths if pcd_path.split('/')[-1].split('_')[-2] in ['101','107','201']
        artifact_360 = [artifact for artifact in artifact_present if artifact[3].split('/')[-1].split('_')[-2] in ['101','107','201'] or artifact[3].split('/')[-1].split('_')[-3] in ['101','107','201']]
        threesixty_pcd_paths = [artifact[3] for artifact in artifact_360]

        print(len(artifact_front), len(artifact_back), len(artifact_360))

        if len(artifact_front)<4 or len(artifact_back)<4 or len(artifact_360)<4:
            print("not enough scan data for each scan step")
            continue
        print("enough scan data for each scan step")
        get_json_metadata = "select json_metadata from model where id = '{}';".format(model_id)
        json_metadata = main_connector.execute(get_json_metadata,fetch_all=True)

        if model_id == 'GAPNet_height_s1':
            if dataformat == 'depth':
                front_pointcloud_numpy = depthmap_to_pcd(front_pcd_paths, calibration, 'gapnet')
                back_pointcloud_numpy = depthmap_to_pcd(back_pcd_paths, calibration, 'gapnet')
                threesixty_pointcloud_numpy = depthmap_to_pcd(threesixty_pcd_paths, calibration, 'gapnet')
            else:
                front_pointcloud_numpy = pcd_processing_gapnet(front_pcd_paths)
                back_pointcloud_numpy = pcd_processing_gapnet(back_pcd_paths)
                threesixty_pointcloud_numpy = pcd_processing_gapnet(threesixty_pcd_paths)
        #TODO check which file is used depth/pcd
        elif 'depthmap' in model_id:
            if dataformat == 'depth':
                front_pointcloud_numpy = get_depthmaps(front_pcd_paths)
                back_pointcloud_numpy = get_depthmaps(back_pcd_paths)
                threesixty_pointcloud_numpy = get_depthmaps(threesixty_pcd_paths)
            else:
                front_pointcloud_numpy = pcd_to_depthmap(front_pcd_paths, calibration)
                back_pointcloud_numpy = pcd_to_depthmap(back_pcd_paths, calibration)
                threesixty_pointcloud_numpy = pcd_to_depthmap(threesixty_pcd_paths, calibration)
        else:
            input_shape = json_metadata[0][0]['input_shape']
            if dataformat == 'depth':
                front_pointcloud_numpy = depthmap_to_pcd(front_pcd_paths, calibration, 'pointnet', input_shape)
                back_pointcloud_numpy = depthmap_to_pcd(back_pcd_paths, calibration, 'pointnet', input_shape)
                threesixty_pointcloud_numpy = depthmap_to_pcd(threesixty_pcd_paths, calibration, 'pointnet', input_shape)
            else:
                front_pointcloud_numpy = pcd_to_ndarray(front_pcd_paths, input_shape)
                back_pointcloud_numpy = pcd_to_ndarray(back_pcd_paths, input_shape)
                threesixty_pointcloud_numpy = pcd_to_ndarray(threesixty_pcd_paths, input_shape)

        front_predictions = generate_result.get_predictions_2(front_pointcloud_numpy, model_id, service)
        back_predictions = generate_result.get_predictions_2(back_pointcloud_numpy, model_id, service)
        threesixty_predictions = generate_result.get_predictions_2(threesixty_pointcloud_numpy, model_id, service)

        #TODO alert when this happens
        if isinstance(front_predictions, str) or isinstance(back_predictions, str) or isinstance(threesixty_predictions, str):
            print("result is type string skipping it")
            print(id)
            continue

        if len(front_predictions)<2 or len(back_predictions)<2 or len(threesixty_predictions)<2:
            print("not enough predictions")
            if len(front_predictions) < 2:
                front_predictions = [[0], [10]]
            if len(back_predictions) < 2:
                back_predictions = [[0], [10]]
            if len(threesixty_predictions) < 2:
                threesixty_predictions = [[0], [10]]
            #continue

        #TODO get measure result bunch object and artifact result bunch object
        results = Bunch()
        results.scan = Bunch()

        if len(qr_code) == 1:
            results.scan.qrcode = qr_code[0]
        results.scan.timestamp = timestamp

        results.scan.measure_id = id[0]

        results.model_result = Bunch()
        results.model_result.model_id = model_id
        results.model_result.measure_result = Bunch()

        results.model_result.measure_result.front_scan = Bunch()
        results.model_result.measure_result.front_scan = get_stats(np.array(front_predictions, dtype = 'float32'))
        results.model_result.measure_result.back_scan = Bunch()
        results.model_result.measure_result.back_scan = get_stats(np.array(back_predictions, dtype = 'float32'))
        results.model_result.measure_result.threesixty_scan = Bunch()
        results.model_result.measure_result.threesixty_scan = get_stats(np.array(threesixty_predictions, dtype = 'float32'))

        #TODO get bunch object of artifact results for each type of scan

        results.model_result.artifact_results = Bunch()

        results.model_result.artifact_results.front_scan = get_artifact_result(artifact_front, front_predictions)

        results.model_result.artifact_results.back_scan = get_artifact_result(artifact_back, back_predictions)

        results.model_result.artifact_results.threesixty_scan = get_artifact_result(artifact_360, threesixty_predictions)

        results_json_string = json.dumps(results)
        results_json_object = json.loads(results_json_string)


        filename, folder = get_file_name(destination_folder, model_id, qr_code[0], 'test')

        if not os.path.exists(folder):
            os.makedirs(folder)

        try:
            with open(r'{0}{1}'.format(folder,filename), 'w') as json_file:
                json.dump(results_json_object, json_file, indent=2)
        except Exception as error:
            print(error)

        result_file = '{0}{1}'.format(folder,filename)

        #TODO update results to database

        flag = upload_result(result_file, id[0], main_connector)

        if flag == 0:
            print("could not update to database")
            continue
        #TODO send results to storage queue

        #flag = upload_to_queue(storage_account_name, result_file, main_connector)

        #if flag != 0:
        #    move_processed_data(result_file, 'measure_result', 'processed')
        #    print(id[0])
            #print(scan_measure_id[0][0])

    main_connector.cursor.close()
    main_connector.connection.close()


if __name__ == "__main__":
    main()