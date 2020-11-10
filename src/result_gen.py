import os
import sys
import time
import glob
import json
import shutil
import random
import numpy as np
from bunch import Bunch
from datetime import datetime

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

#sys.path.append(os.path.expanduser('~/cgm-ml'))
#from cgmcore import modelutils, utils
#from azure.storage.queue import QueueService

import config
import utils.inference as inference
import utils.dbutils as dbutils
import utils.blob_access as blob_access
import utils.rgutils as rgutils
import utils.preprocessing as preprocessing


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


def get_file_name(destination_folder, model_id, qr_code, environment):
    filename = '{0}-{1}-{2}-{3}.json'.format(environment, model_id, qr_code,random.randint(10000, 99999))
    folder = '{0}/{1}/'.format(destination_folder, model_id)
    return filename, folder



def extract_status_code_one(path):
    '''
    Extract the status code from the artifact path
    '''
    status_code = path.split('/')[-1].split('_')[-2]
    return status_code

def extract_status_code_two(path):
    '''
    Extract the status code from the artifact path
    '''
    status_code = path.split('/')[-1].split('_')[-3]
    return status_code

def check_status_code(path, extract_status_code, status_list):
    '''
    Check the artifact belong to one of the status code in status list
    '''

    if extract_status_code(path) in status_list:
        return True
    else:
        return False


class Measure_Result_Generation:

    def __init__(self, measure_id, main_connector, replace_path, container_name):
        self.measure_id = measure_id
        self.main_connector = main_connector
        
        if rgutils.depth_exists(measure_id, self.main_connector):
            self.dataformat = 'depth'
        else:
            self.dataformat = 'pcd'

        self.replace_path = replace_path
        self.container_name = container_name
        self.front_status_list = ['100','104','200']
        self.back_status_list = ['100','104','200']
        self.threesixty_status_list = ['100','104','200']
        self.ACC_NAME = config.ACC_NAME 
        self.ACC_KEY = config.ACC_KEY


    def get_artifact_list_per_measure(self):
        '''
        Get the list of artifact per measure
        '''
        get_artifacts = "SELECT id, qr_code, create_timestamp, replace('{}'".format(self.replace_path)
        get_artifacts += " || split_part(storage_path,'/storage/emulated/0/Child Growth Monitor Scanner App/', 2), 'measurements', 'measure')"
        get_artifacts += "from artifact" # where dataformat = '{}'".format(dataformat)
        get_artifacts += " where measure_id = '{}';".format(self.measure_id[0])
        artifacts = self.main_connector.execute(get_artifacts, fetch_all=True)
        print(len(artifacts))
        
        if len(artifacts) < 15:
            print("not enough data to calculate results measure id {0}".format(self.measure_id[0]))
            continue
        print("enough data to calculate results measure id {0}".format(self.measure_id[0]))
        
        # TODO put it in Dataframe
        self.artifact_list = [list(artifact) for artifact in artifacts]

        self.download_measure()

        self.artifact_present = []
        for artifact in artifact_list:
            if os.path.isfile(artifact[3]) and 'depth' in artifact[3]:
                self.artifact_present.append(artifact)

        print("no of artifacts present ", len(self.artifact_present))


    def download_measure_and_set_calibration(self):
        '''
        Download a scan measure and sets calibration parameter
        '''
        files = [artifact[3] for artifact in self.artifact_list]
        block_blob_service = connect_blob_storage(self.ACC_NAME, self.ACC_KEY, self.container_name)
        blob_access.download_blobs(block_blob_service, container_name, file_list)

        # Not able to understand the code. This does not make sense
        for _file in files:
            if "camera_calibration.txt" in _file:
                calibration_file = _file
        
        try:
            self.calibration = preprocessing.parseCalibration(calibration_file)
        except Exception as error:
            print(error)


    def get_qrcodes_per_measure(self):
        '''
        Get the list of qrcodes from the list of artifact of a measure
        '''
        qr_codes = [artifact[1] for artifact in self.artifact_present]
        qr_code = set(qr_codes)
        self.qr_code = list(qr_code)


    def get_timestamp_per_measure(self):
        '''
        Get the list of timestamp from the list of artifact of a measure
        '''
        timestamp = [artifact[2] for artifact in self.artifact_present]
        timestamp = set(timestamp)
        self.timestamp = list(timestamp)


    def get_artifact_paths(self):
        '''
        Prepare the artifact paths for each scantype
        '''
        #pcd_path for pcd_path in pcd_paths if pcd_path.split('/')[-1].split('_')[-2] in ['100','104','200']
        self.artifact_front = []
        for artifact in self.artifact_present:
            if check_status_code(artifact[3], self.front_status_list, extract_status_code_one, self.front_status_list) or if check_status_code(artifact[3], extract_status_code_two, self.front_status_list):
                self.artifact_front.append(artifact)

        self.artifact_front_pcd_paths = [artifact[3] for artifact in self.artifact_front]

        #pcd_path for pcd_path in pcd_paths if pcd_path.split('/')[-1].split('_')[-2] in ['102','110','202']
        self.artifact_back = []
        for artifact in self.artifact_present:
            if check_status_code(artifact[3], self.front_status_list, extract_status_code_one, self.back_status_list) or if check_status_code(artifact[3], extract_status_code_two, self.back_status_list):
                self.artifact_back.append(artifact)
        self.artifact_back_pcd_paths = [artifact[3] for artifact in self.artifact_back]

        #pcd_path for pcd_path in pcd_paths if pcd_path.split('/')[-1].split('_')[-2] in ['101','107','201']
        self.artifact_threesixty = []
        for artifact in self.artifact_present:
            if check_status_code(artifact[3], self.front_status_list, extract_status_code_one, self.threesixty_status_list) or if check_status_code(artifact[3], extract_status_code_two, self.threesixty_status_list):
                self.artifact_threesixty.append(artifact)
        self.artifact_threesixty_pcd_paths = [artifact[3] for artifact in self.artifact_threesixty]

        print(len(self.artifact_front), len(self.artifact_back), len(self.artifact_threesixty))


    def check_enough_artifact(self, MIN_ARTIFACT_REQUIRED = 4):
        '''
        Checks if there are enought artifact in each scantype to perform the prediction
        '''
        if len(self.artifact_front)<MIN_ARTIFACT_REQUIRED or len(self.artifact_back)<MIN_ARTIFACT_REQUIRED or len(self.artifact_threesixty)<MIN_ARTIFACT_REQUIRED:
            print("not enough scan data for each scan step")
            continue
        print("enough scan data for each scan step")


    def preprocess_artifact(self, model_id):
        '''
        Preprocess each artifact based on its data format
        '''
        get_json_metadata = "select json_metadata from model where id = '{}';".format(model_id)
        json_metadata = self.main_connector.execute(get_json_metadata,fetch_all=True)

        if model_id == 'GAPNet_height_s1':
            if dataformat == 'depth':
                front_pcd_numpy = depthmap_to_pcd(self.artifact_front_pcd_paths, self.calibration, 'gapnet')
                back_pcd_numpy = depthmap_to_pcd(self.artifact_back_pcd_paths, self.calibration, 'gapnet')
                threesixty_pcd_numpy = depthmap_to_pcd(self.artifact_threesixty_pcd_paths, self.calibration, 'gapnet')
            else:
                front_pcd_numpy = preprocessing.pcd_processing_gapnet(self.artifact_front_pcd_paths)
                back_pcd_numpy = preprocessing.pcd_processing_gapnet(self.artifact_back_pcd_paths)
                threesixty_pcd_numpy = preprocessing.pcd_processing_gapnet(self.artifact_threesixty_pcd_paths)
        #TODO check which file is used depth/pcd
        elif 'depthmap' in model_id:
            if dataformat == 'depth':
                front_pcd_numpy = get_depthmaps(self.artifact_front_pcd_paths)
                back_pcd_numpy = get_depthmaps(self.artifact_back_pcd_paths)
                threesixty_pcd_numpy = get_depthmaps(self.artifact_threesixty_pcd_paths)
            else:
                front_pcd_numpy = pcd_to_depthmap(self.artifact_front_pcd_paths, self.calibration)
                back_pcd_numpy = pcd_to_depthmap(self.artifact_back_pcd_paths, self.calibration)
                threesixty_pcd_numpy = pcd_to_depthmap(self.artifact_threesixty_pcd_paths, self.calibration)
        else:
            input_shape = json_metadata[0][0]['input_shape']
            if dataformat == 'depth':
                front_pcd_numpy = depthmap_to_pcd(self.artifact_front_pcd_paths, self.calibration, 'pointnet', input_shape)
                back_pcd_numpy = depthmap_to_pcd(self.artifact_back_pcd_paths, self.calibration, 'pointnet', input_shape)
                threesixty_pcd_numpy = depthmap_to_pcd(self.artifact_threesixty_pcd_paths, self.calibration, 'pointnet', input_shape)
            else:
                front_pcd_numpy = preprocessing.pcd_to_ndarray(self.artifact_front_pcd_paths, input_shape)
                back_pcd_numpy = preprocessing.pcd_to_ndarray(self.artifact_back_pcd_paths, input_shape)
                threesixty_pcd_numpy = preprocessing.pcd_to_ndarray(self.artifact_threesixty_pcd_paths, input_shape)

        self.artifact_front_pcd_numpy = front_pcd_numpy
        self.artifact_back_pcd_numpy = back_pcd_numpy
        self.artifact_threesixty_pcd_numpy = threesixty_pcd_numpy


    def get_height_per_artifact(self, model_id, service):
        '''
        For a list of artifacts, Get the Height prediction for each artifact in a list
        '''
        self.front_predictions = inference.get_predictions_2(self.artifact_front_pcd_numpy, model_id, service)
        self.back_predictions = inference.get_predictions_2(self.artifact_back_pcd_numpy, model_id, service)
        self.threesixty_predictions = inference.get_predictions_2(self.artifact_threesixty_pcd_numpy, model_id, service)
    

    def get_weight_per_artifact(self):
        '''
        Get the Weight prediction per artifact
        '''
        pass


    def check_enough_height_prediction(self):
        '''
        '''
        #TODO alert when this happens
        # May need to different function
        if isinstance(front_predictions, str) or isinstance(back_predictions, str) or isinstance(threesixty_predictions, str):
            # Is this modification correct
            print("Result is type string. Skipping it")
            print(id)
            continue

        if len(self.front_predictions)<2 or len(self.back_predictions)<2 or len(self.threesixty_predictions)<2:
            print("not enough predictions")
            # Not able to understand logic
            if len(self.front_predictions) < 2:
                self.front_predictions = [[0], [10]]
            if len(self.back_predictions) < 2:
                self.back_predictions = [[0], [10]]
            if len(self.threesixty_predictions) < 2:
                self.threesixty_predictions = [[0], [10]]
            #continue


    def get_height_per_measure(self):
        '''
        Get the heigth prediction per scan measure
        '''
        pass


    def get_weight_per_measure(self):
        '''
        Get the Weight prediction per scan measure
        '''
        pass


    def get_height_result(self):
        '''

        '''
        pass


    def get_weight_result(self):
        pass


    def get_blur_result(self):
        pass


    def create_result_in_json_format(self, model_id):
        '''
        Prepare results in Json format
        '''
        #TODO get measure result bunch object and artifact result bunch object
        results = Bunch()
        results.scan = Bunch()

        if len(self.qr_code) == 1:
            results.scan.qrcode = self.qr_code[0]
        results.scan.timestamp = self.timestamp

        results.scan.measure_id = self.measure_id[0]

        results.model_result = Bunch()
        results.model_result.model_id = model_id
        results.model_result.measure_result = Bunch()

        results.model_result.measure_result.front_scan = Bunch()
        results.model_result.measure_result.front_scan = get_stats(np.array(self.front_predictions, dtype = 'float32'))
        results.model_result.measure_result.back_scan = Bunch()
        results.model_result.measure_result.back_scan = get_stats(np.array(self.back_predictions, dtype = 'float32'))
        results.model_result.measure_result.threesixty_scan = Bunch()
        results.model_result.measure_result.threesixty_scan = get_stats(np.array(self.threesixty_predictions, dtype = 'float32'))

        #TODO get bunch object of artifact results for each type of scan

        results.model_result.artifact_results = Bunch()
        results.model_result.artifact_results.front_scan = get_artifact_result(self.artifact_front, self.front_predictions)
        results.model_result.artifact_results.back_scan = get_artifact_result(self.artifact_back, self.back_predictions)
        results.model_result.artifact_results.threesixty_scan = get_artifact_result(self.artifact_360, self.threesixty_predictions)

        results_json_string = json.dumps(results)
        self.results_json_object = json.loads(results_json_string)


    def update_measure_table_and_blob(self, model_id, destination_folder):
        '''
        Update the result to measure_result table
        '''
        filename, folder = get_file_name(destination_folder, model_id, self.qr_code[0], 'test')

        if not os.path.exists(folder):
            os.makedirs(folder)

        try:
            with open(r'{0}{1}'.format(folder, filename), 'w') as json_file:
                json.dump(self.results_json_object, json_file, indent=2)
        except Exception as error:
            print(error)

        result_file = '{0}{1}'.format(folder, filename)

        #TODO update results to database
        flag = rgutils.upload_result(result_file, measure_id[0], self.main_connector)

        if flag == 0:
            print("could not update to database")
            continue
        #TODO send results to storage queue

        #flag = rgutils.upload_to_queue(storage_account_name, result_file, main_connector)

        #if flag != 0:
        #    move_processed_data(result_file, 'measure_result', 'processed')
        #    print(id[0])
            #print(scan_measure_id[0][0])


def main():
    if len(sys.argv) != 3:
        print("Please provide model_id and endpoint name.")
        exit(1)

    #destination_folder = str(sys.argv[1])
    #db_connection_file = str(sys.argv[2])
    #storage_account_name = str(sys.argv[3])
    model_id = str(sys.argv[1])
    service = str(sys.argv[2])
    #calibration_file = str(sys.argv[6])
    #container_name = str(sys.argv[7])

    container_name = "scans"
    destination_folder = '~'

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
    measure_ids = main_connector.execute(select_measures, fetch_all=True)

    replace_path = "~/" + config.ACC_NAME + '/qrcode/'

    #measure_ids = [('c66050300c1ab684_measure_1601356048051_vj7fOLrU2dYwWDOT',), ('c66050300c1ab684_measure_1601356093034_CFIfgb2SFufC7Pe9',)]

    print(len(measure_ids))

    for measure_id in measure_ids:
        measure_rg = Measure_Result_Generation(measure_id, main_connector, replace_path, container_name)
        measure_rg.get_artifact_list_per_measure()
        measure_rg.download_measure_and_set_calibration()
        measure_rg.get_qrcodes_per_measure()
        measure_rg.get_timestamp_per_measure()
        measure_rg.get_artifact_paths()
        measure_rg.check_enough_artifact()
        measure_rg.preprocess_artifact(model_id)
        measure_rg.get_height_per_artifact(model_id, service)
        measure_rg.check_enough_height_prediction()
        measure_rg.create_result_in_json_format(model_id)
        measure_rg.update_measure_table_and_blob(self, model_id, destination_folder)

    main_connector.cursor.close()
    main_connector.connection.close()


if __name__ == "__main__":
    main()