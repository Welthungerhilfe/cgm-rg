from uuid import uuid4
import anyio
import asyncer
from asyncer import asyncify
import numpy as np


async def post_result_files(cgm_api, files, file_format='rgb'):
    try:
        async with asyncer.create_task_group() as task_group:
            soon_values = [task_group.soonify(cgm_api.post_files)(file, file_format) for k, file in files.items()]
        post_file_ids = {}
        # 🔹 Ensure results are properly accessed after task_group exits
        results = [soon.value for soon in soon_values]
        for result, k in zip(results, files.keys()):
            post_file_ids[k] = result
        return post_file_ids
    except Exception as e:
        print(e)


def get_result_dict(scan_id, workflow_id, source_artifacts=[], data=None, source_results=[], file=None):
    result_dict =  {}
    result_dict["id"] = str(uuid4())
    result_dict["scan"] = scan_id
    result_dict["workflow"] = workflow_id
    result_dict["generated"] = "2025-02-20T18:33:18.444Z"
    result_dict["start_time"] = "2025-02-20T18:33:18.444Z"
    result_dict["end_time"] = "2025-02-20T18:33:18.444Z"
    result_dict["source_artifacts"] = source_artifacts
    result_dict["source_results"] = source_results
    if data:
        result_dict["data"] = data
    if file:
        result_dict["file"] = file

    return result_dict


def get_files_results(file_id_dict, workflow_id, results):
    result_dicts = []
    for k, v in file_id_dict.items():
        if k[1] in results:
            continue
        rd = get_result_dict(k[0], workflow_id, [k[1]], file=v)
        result_dicts.append(rd)
    return result_dicts


def get_pose_json_results(artifacts, pose_predictions, workflow_id, results):
    pose_results = []
    for (artifact, pose_prediction) in zip(artifacts, pose_predictions):
        if artifact['id'] in results:
            continue
        no_of_pose_detected, pose_score, pose_result = pose_prediction[0]
        no_of_pose_result = get_result_dict(artifact['scan_id'], workflow_id, [artifact['id']], data={'no of person using pose': str(no_of_pose_detected)})
        pose_results.append(no_of_pose_result)
        for i in range(0, no_of_pose_detected):
            data={'Pose Scores': str(pose_score[i]),
                'Pose Results': str(pose_result[i])}
            pose_score_results = get_result_dict(artifact['scan_id'], workflow_id, [artifact['id']], data=data)
            pose_results.append(pose_score_results)
    return pose_results


def get_rgb_depth_allignment_result(scan_id, artifacts, workflow_id, similarity_index, alignment_status):
    data = {"similarity_index": similarity_index, "alignment_status": alignment_status}
    rd = get_result_dict(scan_id, workflow_id, artifacts, data=data)
    return rd


def get_blur_json_results(artifacts, blur_results, workflow_id, results):
    blur_results = []
    for artifact, blur_result in zip(artifacts, blur_results):
        if artifact['id'] in results:
            continue
        data = {'faces_detected': str(len(blur_result['faces_detected'])), 'face_attributes': blur_result['faces_detected']},
        blur_results.append(get_result_dict(artifact['scan_id'], workflow_id, [artifact['id']], data=data))
    return blur_results


def get_depth_feature_json_results(artifacts, no_of_zeroes, angles, workflow_id, results):
    no_of_zeroes_results = []
    for (artifact, no_of_zero, angle) in zip(artifacts, no_of_zeroes, angles):
        if artifact['id'] in results:
            continue
        data = {'percentage_of_zeroes': no_of_zero, 'angle_between_camera_and_floor': angle}
        no_of_zeroes_results.append(get_result_dict(artifact['scan_id'], workflow_id, [artifact['id']], data=data))
    return no_of_zeroes_results


def get_json_results(artifacts, predictions, workflow_id, existing_results, data_key):
    results = []
    for artifact, prediction in zip(artifacts, predictions):
        if artifact['id'] in existing_results:
            continue
        data = {data_key: str(prediction[0])}
        results.append(get_result_dict(artifact['scan_id'], workflow_id, [artifact['id']], data=data))
    return results


def get_mean_result(predictions):
    return np.mean(predictions)


def get_mean_results(scan_id, filtered_data, results_workflow_dict, source_workflow_id, workflow_id, key, artifact_ids):
    if len(filtered_data) < 9 and results_workflow_dict.get(source_workflow_id, []):
        workflow_results = results_workflow_dict.get(source_workflow_id, [])
        filtered_results = [r for r in workflow_results if r[0] not in filtered_data]
        predictions = [float(r[2][key]) for r in filtered_results]
        predictions_wo_outliers = remove_outliers(predictions)
        preds = np.mean(predictions_wo_outliers)
        data={
            f'mean_{key}': np.mean(predictions),
            f'iqr_{key}': preds,
        }
    else:
        data = {
            f'mean_{key}': None
        }
        preds = None
    rd = get_result_dict(scan_id, workflow_id, data=data, source_artifacts=artifact_ids)
    return rd, preds


def remove_outliers(arr):
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    result = [nums for nums in arr if lower <= nums <= upper]
    arr_new = np.array(result).reshape(len(result), 1)
    return arr_new


# Convert the list to an array of subarrays
def LOF(predictions):
    # data = rows['prediction_heights']
    if len(predictions)>8:
        # Apply the Local Outlier Factor algorithm
        lof = LocalOutlierFactor(n_neighbors=5)
        outlier_labels = lof.fit_predict(predictions)
        # Filter out the outliers
        inliers = predictions[outlier_labels == 1]
        inliers_data = list(itertools.chain.from_iterable(inliers))
        return inliers_data
    return []


def filter_results(results, artifact_order_mapping, filter_functions_dict):
    filtered_data = set()
    results_workflow_dict = {}
    for r in results:
        k = r['workflow']
        if k in filter_functions_dict:
            if filter_functions_dict[k](r['data']):
                filtered_data.add(artifact_order_mapping[r['source_artifacts'][0]])
        if k not in results_workflow_dict:
            results_workflow_dict[k] = []
        results_workflow_dict[k].append((artifact_order_mapping[r['source_artifacts'][0]], r['source_artifacts'][0], r.get('data')))
    return filtered_data, results_workflow_dict


def pose_score_filter(data):
    if 'no of person using pose' in data:
        if int(data['no of person using pose']) != 1:
            return True
        else:
            False
    elif 'Pose Scores' in data:
        if float(data['Pose Scores']) < 0.83:
            return True
        return False
    else:
        raise Exception("No required fields found in pose result")


def depth_feature_filter(data):
    if data['angle_between_camera_and_floor'] == 'NA':
        return False
    if 10 < float(data['angle_between_camera_and_floor']) or -20 > float(data['angle_between_camera_and_floor']) or float(data['percentage_of_zeroes']) > 15:
        return True
    return False


def distance_to_child_filter(data):
    if float(data['child_distance']) < 0.5 or float(data['child_distance']) > 1.5:
        return True
    return False
