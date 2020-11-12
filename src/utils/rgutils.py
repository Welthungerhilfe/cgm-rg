import json
import dbutils
from datetime import datetime
import numpy as np


def get_measure_insert(
        measure_id,
        model_id,
        scan_step,
        json_value,
        table,
        constant):
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
    confidence_value = 1 - ((json_value["max"] - json_value["min"]) / constant)
    if confidence_value < 0:
        confidence_value = 0
    measure_mapping['confidence_value'] = confidence_value
    measure_mapping['float_value'] = json_value['mean']
    json_value['timestamp'] = int(datetime.now().timestamp() * 1000)
    json_value = json.dumps(json_value)
    measure_mapping['json_value'] = json_value
    keys = []
    values = []
    for key in measure_mapping.keys():
        keys.append(key)
        values.append(measure_mapping[key])
    insert_statement = dbutils.create_insert_statement(
        table, keys, values, True, True)
    # print(measure_mapping)

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
    insert_statement = dbutils.create_insert_statement(
        table, keys, values, True, True)
    # print(artifact_mapping)

    return insert_statement


def upload_result(filename, measure_id, db_connector):
    height_constant = 2.5
    weight_constant = 1.2
    with open(filename) as json_file:
        json_data = json.load(json_file)
        # for model_result in json_data["model_results"]:
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
            front_insert_statement = get_measure_insert(
                measure_id, model_id, "front", front_scan_results, table, height_constant)
            back_insert_statement = get_measure_insert(
                measure_id, model_id, "back", back_scan_results, table, height_constant)
            threesixty_insert_statement = get_measure_insert(
                measure_id, model_id, "360", threesixty_scan_results, table, height_constant)
        elif result_key == 'weight':
            front_insert_statement = get_measure_insert(
                measure_id, model_id, "front", front_scan_results, table, weight_constant)
            back_insert_statement = get_measure_insert(
                measure_id, model_id, "back", back_scan_results, table, weight_constant)
            threesixty_insert_statement = get_measure_insert(
                measure_id, model_id, "360", threesixty_scan_results, table, weight_constant)
            # print(front_insert_statement)
            # print(back_insert_statement)
            # print(threesixty_insert_statement)

        try:
            db_connector.execute(front_insert_statement)
            db_connector.execute(back_insert_statement)
            db_connector.execute(threesixty_insert_statement)
            print(
                'successfully inserted data to {0} table for measure_id {1}'.format(
                    table, measure_id))
        except Exception as error:
            print(error)
            flag = 0

        table = "artifact_result"

        for artifact_result in model_result["artifact_results"]["front_scan"]:
            insert_statement = get_artifact_insert(
                artifact_result["artifact_id"],
                model_id,
                "front",
                artifact_result["prediction"],
                table)
            try:
                db_connector.execute(insert_statement)
                print(
                    'successfully inserted data to {0} table for artifact_id {1}'.format(
                        table, artifact_result["artifact_id"]))
            except Exception as error:
                print(error)
                flag = 0
                # print(insert_statement)

        for artifact_result in model_result["artifact_results"]["back_scan"]:
            insert_statement = get_artifact_insert(
                artifact_result["artifact_id"],
                model_id,
                "back",
                artifact_result["prediction"],
                table)
            try:
                db_connector.execute(insert_statement)
                print(
                    'successfully inserted data to {0} table for artifact_id {1}'.format(
                        table, artifact_result["artifact_id"]))
            except Exception as error:
                print(error)
                flag = 0
                # print(insert_statement)

        for artifact_result in model_result["artifact_results"]["threesixty_scan"]:
            insert_statement = get_artifact_insert(
                artifact_result["artifact_id"],
                model_id,
                "360",
                artifact_result["prediction"],
                table)
            try:
                db_connector.execute(insert_statement)
                print(
                    'successfully inserted data to {0} table for artifact_id {1}'.format(
                        table, artifact_result["artifact_id"]))
            except Exception as error:
                print(error)
                flag = 0
                # print(insert_statement)
    return flag


def get_measure_insert_dict(
        measure_id,
        model_id,
        scan_step,
        json_value,
        constant):
    measure_mapping = {}
    measure_mapping['measure_id'] = measure_id
    measure_mapping['model_id'] = model_id
    if 'height' in model_id:
        measure_mapping['key'] = 'height_' + scan_step
    elif 'weight' in model_id:
        measure_mapping['key'] = 'weight_' + scan_step
    for result_k in json_value.keys():
        json_value[result_k] = float(json_value[result_k])
    confidence_value = 1 - ((json_value["max"] - json_value["min"]) / constant)
    if confidence_value < 0:
        confidence_value = 0
    measure_mapping['confidence_value'] = confidence_value
    measure_mapping['float_value'] = json_value['mean']
    # json_value = json.dumps(json_value)
    json_value['timestamp'] = int(datetime.now().timestamp() * 1000)
    measure_mapping['json_value'] = str(json_value)
    print(measure_mapping['json_value'])
    # print(measure_mapping)
    return measure_mapping


def depth_exists(id, connector):

    check_dataformat = "SELECT EXISTS (SELECT id FROM artifact WHERE measure_id='{}'".format(
        id[0]) + " and dataformat = 'depth');"
    exists = connector.execute(check_dataformat, fetch_all=True)
    return exists[0][0]


def process_posenet_result(
        pose_prediction,
        model_id,
        artifact_id,
        db_connector):
    table = "artifact_result"
    PART_NAMES = [
        "nose",
        "rightShoulder",
        "rightElbow",
        "rightWrist",
        "leftShoulder",
        "leftElbow",
        "leftWrist",
        "rightHip",
        "rightKnee",
        "rightAnkle",
        "leftHip",
        "leftKnee",
        "leftAnkle",
        "rightEye",
        "leftEye",
        "rightEar",
        "leftEar"]
    pose_scores = np.array(pose_prediction['pose_scores'])
    keypoint_scores = np.array(pose_prediction['keypoint_scores'])
    keypoint_coords = np.array(pose_prediction['keypoint_coords'])

    float_value = 0.0
    confidence_value = 0.0
    pose_list = []

    for pi in range(len(pose_scores)):
        if pose_scores[pi] == 0.:
            break
        float_value += 1
        confidence_value = pose_scores[pi]
        print('Pose #%d, score = %f' % (pi, pose_scores[pi]))
        # pose_dict['pose_number'] = pi
        pose_dict = {}
        for ki, (s, c) in enumerate(
                zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
            pose_result = {}
            pose_result['score'] = s.tolist()
            pose_result['coordinates'] = c.tolist()
            pose_dict[PART_NAMES[ki]] = pose_result
            # print('Keypoint %s, score = %f, coord = %s' % (PART_NAMES[ki], s, c))
        pose_list.append(pose_dict)

    pose_dict = {}
    for num, pose in enumerate(pose_list, start=1):
        key = 'pose_' + str(num)
        pose_dict[key] = pose

    pose_json = json.dumps(pose_dict)

    artifact_mapping = {}
    artifact_mapping['model_id'] = model_id
    artifact_mapping['artifact_id'] = artifact_id
    artifact_mapping['key'] = 'pose_prediction'
    artifact_mapping['float_value'] = float_value
    artifact_mapping['confidence_value'] = confidence_value
    artifact_mapping['json_value'] = pose_json

    keys = []
    values = []
    for key in artifact_mapping.keys():
        keys.append(key)
        values.append(artifact_mapping[key])
    insert_statement = dbutils.create_insert_statement(
        table, keys, values, True, True)
    try:
        db_connector.execute(insert_statement)
        print(
            'successfully inserted data to {0} table for artifact_id {1}'.format(
                table, artifact_id))
    except Exception as error:
        print(error)


def process_face_blur_results(model_id, artifact_id, db_connector):
    '''
    Makes entry for artifact which are face blurred in db
    '''

    table = "artifact_result"

    artifact_mapping = {}
    artifact_mapping['model_id'] = model_id
    artifact_mapping['artifact_id'] = artifact_id
    artifact_mapping['key'] = 'face_blur'

    insert_statement = dbutils.create_insert_statement(
        table, list(artifact_mapping.keys()), list(artifact_mapping.values()), True, True)

    try:
        db_connector.execute(insert_statement)
        print(
            'successfully inserted data to {0} table for artifact_id {1}'.format(
                table, artifact_id))
    except Exception as error:
        print(error)
