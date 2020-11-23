from pathlib import Path
import os
import sys

REPO_DIR = Path(__file__).parents[1].absolute()
SRC_DIR = str(REPO_DIR / "src")
sys.path.append(SRC_DIR)

import result_gen
import utils.dbutils as dbutils
import utils.preprocessing as preprocessing
import random


container_name = "scans"
destination_container_name = "processed-images"
destination_folder = '~'
height_model_id = 'q3_depthmap_height_run_01'
height_service = 'q3-depthmap-height-run-01-ci'
pose_model_id = 'posenet_1.0'
pose_service = 'aci-posenet-ind'
face_blur_model_id = 'face_recogntion'
measure_id = ('c66050300c1ab684_measure_1601356048051_vj7fOLrU2dYwWDOT',)
preprocessing.setWidth(int(240 * 0.75))
preprocessing.setHeight(int(180 * 0.75))
replace_path = 'qrcode/'

main_connector = dbutils.connect_to_main_database()

id_split = measure_id[0].split('_')
query_delete_measure_result = "delete from measure_result where measure_id = '{}'".format(
    measure_id[0]) + " and model_id = '{}';".format(height_model_id)
try:
    main_connector.execute(query_delete_measure_result)
except Exception as error:
    print(error)
tmp_str = id_split[0] + "%" + id_split[2][:-1] + "%"
query_delete_artifact_result = f"delete from artifact_result where model_id in ('{height_model_id}', '{pose_model_id}', '{face_blur_model_id}') and artifact_id like '{tmp_str}';"
try:
    main_connector.execute(query_delete_artifact_result)
except Exception as error:
    print(error)

test_measure_rg = result_gen.MeasureResultGeneration(
    measure_id, main_connector, replace_path, container_name, destination_container_name)


def test_get_artifact_list_per_measure():

    result = test_measure_rg.get_artifact_list_per_measure()

    assert result == True


def test_get_qrcodes_per_measure():

    test_measure_rg.get_qrcodes_per_measure()

    result = test_measure_rg.qr_code

    assert len(result) > 0


def test_get_timestamp_per_measure():

    test_measure_rg.get_timestamp_per_measure()

    result = test_measure_rg.timestamp

    assert len(result) > 0


def test_get_artifact_paths():

    test_measure_rg.get_artifact_paths()

    artifacts_front = test_measure_rg.artifact_front

    artifacts_back = test_measure_rg.artifact_back

    artifact_threesixty = test_measure_rg.artifact_threesixty

    assert len(artifacts_front) > 0 and len(artifacts_back) > 0 and len(artifact_threesixty) > 0


def test_check_enough_artifact():

    result = test_measure_rg.check_enough_artifact()

    assert result == True


def test_preprocess_artifact():

    test_measure_rg.preprocess_artifact(height_model_id)

    assert len(test_measure_rg.artifact_front_pcd_numpy) > 0 and len(test_measure_rg.artifact_back_pcd_numpy) > 0 and len(test_measure_rg.artifact_threesixty_pcd_numpy) > 0


def test_get_height_per_artifact():

    test_measure_rg.get_height_per_artifact(height_model_id, height_service)

    assert len(test_measure_rg.front_predictions) > 0 and len(test_measure_rg.back_predictions) > 0 and len(test_measure_rg.threesixty_predictions) > 0


def test_check_enough_height_prediction():

    result = test_measure_rg.check_enough_height_prediction()

    assert result == True


def test_measure_result_deleted():

    check_exists = f"select exists(select 1 from measure_result where measure_id='{measure_id[0]}' and model_id = '{height_model_id}');"
    result = main_connector.execute(check_exists, fetch_all=True)
    result = result[0][0]

    assert result != True


def test_artifact_result_deleted():

    random_depth_artifact = random.choice(test_measure_rg.depth_artifact_present)
    artifact_id = random_depth_artifact[0]

    check_exists = f"select exists(select 1 from artifact_result where artifact_id='{artifact_id}' and model_id = '{height_model_id}');"
    result = main_connector.execute(check_exists, fetch_all=True)
    result = result[0][0]

    assert result != True


def test_pose_results_deleted():

    random_rgb_artifact = random.choice(test_measure_rg.rgb_artifact_present)

    artifact_id = random_rgb_artifact[0]

    check_exists = f"select exists(select 1 from artifact_result where artifact_id='{artifact_id}' and model_id = '{pose_model_id}');"
    result = main_connector.execute(check_exists, fetch_all=True)
    result = result[0][0]

    assert result != True


def test_blur_result_deleted():

    random_rgb_artifact = random.choice(test_measure_rg.rgb_artifact_present)

    artifact_id = random_rgb_artifact[0]

    check_exists = f"select exists(select 1 from artifact_result where artifact_id='{artifact_id}' and model_id = '{face_blur_model_id}');"
    result = main_connector.execute(check_exists, fetch_all=True)
    result = result[0][0]

    assert result != True


def test_create_result_in_json_format():

    test_measure_rg.create_result_in_json_format(height_model_id)

    assert len(test_measure_rg.results_json_object) > 0


def test_update_measure_table_and_blob():

    result = test_measure_rg.update_measure_table_and_blob(height_model_id, destination_folder)

    assert result == True


def test_measure_result_inserted():

    check_exists = f"select exists(select 1 from measure_result where measure_id='{measure_id[0]}' and model_id = '{height_model_id}');"
    result = main_connector.execute(check_exists, fetch_all=True)
    result = result[0][0]

    assert result == True


def test_artifact_result_inserted():

    random_depth_artifact = random.choice(test_measure_rg.depth_artifact_present)
    artifact_id = random_depth_artifact[0]

    check_exists = f"select exists(select 1 from artifact_result where artifact_id='{artifact_id}' and model_id = '{height_model_id}');"
    result = main_connector.execute(check_exists, fetch_all=True)
    result = result[0][0]

    assert result == True


def test_get_pose_results():

    test_measure_rg.get_pose_results(pose_model_id, pose_service)
    random_rgb_artifact = random.choice(test_measure_rg.rgb_artifact_present)

    artifact_id = random_rgb_artifact[0]

    check_exists = f"select exists(select 1 from artifact_result where artifact_id='{artifact_id}' and model_id = '{pose_model_id}');"
    result = main_connector.execute(check_exists, fetch_all=True)
    result = result[0][0]

    assert result == True


def test_get_blur_result():

    test_measure_rg.get_blur_result(face_blur_model_id)
    random_rgb_artifact = random.choice(test_measure_rg.rgb_artifact_present)

    artifact_id = random_rgb_artifact[0]

    check_exists = f"select exists(select 1 from artifact_result where artifact_id='{artifact_id}' and model_id = '{face_blur_model_id}');"
    result = main_connector.execute(check_exists, fetch_all=True)
    result = result[0][0]

    assert result == True


def test_delete_downloaded_artifacts():

    test_measure_rg.delete_downloaded_artifacts()
    random_artifact = random.choice(test_measure_rg.artifact_list)

    artifact_path = random_artifact[3]

    result = os.path.isfile(artifact_path)

    assert result != True
