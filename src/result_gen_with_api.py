import os
import cv2
import glob
import json
import uuid
import copy
import pprint
import requests
from datetime import datetime
import jsonschema
import numpy as np
import face_recognition
from bunch import Bunch
from shutil import copyfile
#import utils.inference as inference
#import utils.preprocessing as preprocessing

ARTIFACT_FOR_BLUR_FLOW_STATUS = False
ARTIFACT_FOR_POSE_FLOW_STATUS = False
ARTIFACT_FOR_HEIGHT_FLOW_STATUS = False
RESIZE_FACTOR = 4

def download_blur_flow_artifact(format_wise_artifact, format_name, scan_dir):
    global ARTIFACT_FOR_BLUR_FLOW_STATUS

    blur_format_wise_artifact = []
    print('scan_dir:: ', scan_dir, '\n')
    print("format_name: ", format_name, '\n')
    format_dir = os.path.join(scan_dir, format_name)
    print('format_dir: ', format_dir, '\n')
    for i, artifact in enumerate(format_wise_artifact[format_name]):
        mod_artifact = copy.deepcopy(artifact)
        
        
        print("Artifact Name: ", mod_artifact["file"], '\n')
        status_code = get_files(mod_artifact["file"], format_dir)
        #status_code = get_files_mockup(mod_artifact["file"], format_dir)
        if status_code == 200:
            mod_artifact['status'] = '200'
            blur_format_wise_artifact.append(mod_artifact)

    ARTIFACT_FOR_BLUR_FLOW_STATUS = True

    return blur_format_wise_artifact


def download_pose_flow_artifact(format_name):
    if ARTIFACT_FOR_POSE_FLOW_STATUS:
        return
    for artifact in format_wise_artifact[format_name]:
        get_files(artifact["file"], format_dir)
    ARTIFACT_FOR_POSE_FLOW_STATUS = True


def download_height_flow_artifact(format_name):
    if ARTIFACT_FOR_HEIGHT_FLOW_STATUS:
        return
    for artifact in format_wise_artifact[format_name]:
        get_files(artifact["file"], format_dir)
    ARTIFACT_FOR_HEIGHT_FLOW_STATUS = True

def get_files(file_id, format_dir):
    '''
    Get request for /files/id
    '''
    url = "http://localhost:5001/"
    file_endpoint = 'api/scan/files/'

    response = requests.get(url+file_endpoint+ file_id)

    print("\nGet Status code of download file: ", response.status_code)

    file_path = os.path.join(format_dir, file_id)

    print("\nPath of dowloaded file: ", file_path)

    with open(file_path, 'wb') as f:
        f.write(response.content)

    return response.status_code


def get_files_mockup(file_id, format_dir):
    '''
    Mock up for get request for /files/id
    '''

    get_dir = "/home/nikhil/cgm-all/cgm-rg/get-mockup/"
    input_path = os.path.join(get_dir, file_id)
    output_path = os.path.join(format_dir, file_id)
    copyfile(input_path, output_path)

    return 200


def blur_faces_in_file(source_path: str, target_path: str) -> bool:
    """Blur image
    Returns:
        bool: True if blurred otherwise False
    """

    # Read the image.
    assert os.path.exists(source_path), f"{source_path} does not exist"
    rgb_image = cv2.imread(source_path)
    image = rgb_image[:, :, ::-1]  # RGB -> BGR for OpenCV

    # The images are provided in 90degrees turned. Here we rotate 90degress to the right.
    image = np.swapaxes(image, 0, 1)

    # Scale image down for faster prediction.
    small_image = cv2.resize(image, (0, 0), fx=1. / RESIZE_FACTOR, fy=1. / RESIZE_FACTOR)

    # Find face locations.
    face_locations = face_recognition.face_locations(small_image, model="cnn")

    # Check if image should be used.
    '''if not should_image_be_used(source_path, number_of_faces=len(face_locations)):
        # logging.warn(f"{len(face_locations)} face locations found and not blurred for path: {source_path}")
        print(f"{len(face_locations)} face locations found and not blurred for path: {source_path}")
        return False
    '''

    file_directory = os.path.dirname(target_path)
    if not os.path.isdir(file_directory):
        os.makedirs(file_directory)

    # Blur the image.
    for top, right, bottom, left in face_locations:
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= RESIZE_FACTOR
        right *= RESIZE_FACTOR
        bottom *= RESIZE_FACTOR
        left *= RESIZE_FACTOR

        # Extract the region of the image that contains the face.
        face_image = image[top:bottom, left:right]

        # Blur the face image.
        face_image = cv2.GaussianBlur(face_image, ksize=(99, 99), sigmaX=30)

        # Put the blurred face region back into the frame image.
        image[top:bottom, left:right] = face_image

    # Rotate image back.
    image = np.swapaxes(image, 0, 1)

    # Write image to hard drive.
    rgb_image = image[:, :, ::-1]  # BGR -> RGB for OpenCV
    print("\ntarget_path : ", target_path)
    cv2.imwrite(target_path, rgb_image)

    # logging.info(f"{len(face_locations)} face locations found and blurred for path: {source_path}")
    print(f"{len(face_locations)} face locations found and blurred for path: {source_path}")
    return True

def blur_face(source_path: str):
    """Blur image
    Returns:
        bool: True if blurred otherwise False
    """

    # Read the image.
    assert os.path.exists(source_path), f"{source_path} does not exist"
    rgb_image = cv2.imread(source_path)
    image = rgb_image[:, :, ::-1]  # RGB -> BGR for OpenCV

    # The images are provided in 90degrees turned. Here we rotate 90degress to the right.
    image = np.swapaxes(image, 0, 1)

    # Scale image down for faster prediction.
    small_image = cv2.resize(image, (0, 0), fx=1. / RESIZE_FACTOR, fy=1. / RESIZE_FACTOR)

    # Find face locations.
    face_locations = face_recognition.face_locations(small_image, model="cnn")

    # Check if image should be used.
    #if not should_image_be_used(source_path, number_of_faces=len(face_locations)):
    #    # logging.warn(f"{len(face_locations)} face locations found and not blurred for path: {source_path}")
    #    print(f"{len(face_locations)} face locations found and not blurred for path: {source_path}")
    #    return _, False

    #file_directory = os.path.dirname(target_path)
    #if not os.path.isdir(file_directory):
    #    os.makedirs(file_directory)

    # Blur the image.
    for top, right, bottom, left in face_locations:
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= RESIZE_FACTOR
        right *= RESIZE_FACTOR
        bottom *= RESIZE_FACTOR
        left *= RESIZE_FACTOR

        # Extract the region of the image that contains the face.
        face_image = image[top:bottom, left:right]

        # Blur the face image.
        face_image = cv2.GaussianBlur(face_image, ksize=(99, 99), sigmaX=30)

        # Put the blurred face region back into the frame image.
        image[top:bottom, left:right] = face_image

    # Rotate image back.
    image = np.swapaxes(image, 0, 1)

    # Write image to hard drive.
    rgb_image = image[:, :, ::-1]  # BGR -> RGB for OpenCV

    # logging.info(f"{len(face_locations)} face locations found and blurred for path: {source_path}")
    print(f"{len(face_locations)} face locations found and blurred for path: {source_path}\n")
    return rgb_image, True


def post_files_using_path(path):
    '''
    Post the file results produced while Result Generation
    using POST /files
    '''
    url = "http://localhost:5001"
    file_endpoint = '/api/scan/files'

    type_ = 'image/jpeg'
    files = {}
    files['file'] = (open(path, 'rb'), type_)
    files['filename'] = path.split('/')[-1]
    print('\nFile name to post : ', files['filename'])
    
    response = requests.post(url + file_endpoint, files=files, headers={'content_type':'multipart/form-data'})

    '''request_body1 = {
        'filename': "test.jpg",
        'file': (open(path, 'rb'), 'image/jpeg')
    }
    

    response = requests.post(url + file_endpoint, data=request_body1, content_type='multipart/form-data')
    '''

    #response = requests.post(url+file_endpoint, files=files,  headers={'content_type':'multipart/form-data'})
    file_id  = response.content.decode('utf-8')
    print("\nFile Id from post of test.jpg: ", file_id, '\n')

    return file_id, response.status_code

def post_files(bin_file):
    '''
    Post the file results produced while Result Generation
    using POST /files
    '''
    url = "http://localhost:5001"
    file_endpoint = '/api/scan/files'

    
    _, bin_file = cv2.imencode('.JPEG', bin_file)
    bin_file = bin_file.tostring()

    files = {
            'file': bin_file,
            'filename': 'test.jpg'
            }
    response = requests.post(url+file_endpoint, files=files,  headers={'content_type':'multipart/form-data'})
    file_id  = response.content.decode('utf-8')
    print("File Id from post of test.jpg: ", file_id, '\n')

    return file_id, response.status_code

def post_workflow(worflow_path):
    '''
    Post the worflows using POST /files
    '''
    return str(uuid.uuid4()), 200


def post_results(json_obj):
    '''
    Post the result object produced while Result Generation
    using POST /results
    '''
    url = "http://localhost:5001"
    endpoint = '/api/scan/results'

    response = requests.post(url+endpoint, json=json_obj)

    print("Status of post result response: ", response.status_code, '\n')

    return response.status_code


def prepare_blur_result_object(scan_id, workflow_id, source_artifacts_list, blur_id_from_post_request, generated_timestamp):
    '''
    Prepare the result object in the results format
    '''

    blur_result = Bunch()
    # Need to clarify
    #blur_result.id = str(uuid.uuid4())
    blur_result.id = f"{uuid.uuid4()}"
    blur_result.scan = scan_id
    blur_result.workflow = workflow_id
    blur_result.source_artifacts = source_artifacts_list
    blur_result.source_results = []
    #blur_result.data = Bunch()
    blur_result.file = blur_id_from_post_request
    blur_result.generated = generated_timestamp
    blur_result.meta = []

    res = Bunch()
    res.results = []
    res.results.append(blur_result)

    return res


def process_scan_metadata(scan_metadata_obj):
    '''
    process the scan object to get the list of jpeg id 
    and artifact id return a dict of format as key and 
    list of file id as values
    '''
    format_wise_artifact = {}
    
    scan_id = scan_metadata_obj['id']
    artifact_list = scan_metadata_obj['artifacts']
    #pprint.pprint(artifact_list)
    
    for artifact in artifact_list:

        mod_artifact = copy.deepcopy(artifact)
        mod_artifact['status'] = ''
        pprint.pprint(mod_artifact)

        if mod_artifact['format'] in format_wise_artifact:
            format_wise_artifact[mod_artifact['format']].append(mod_artifact)
        else:
            format_wise_artifact[mod_artifact['format']] = [mod_artifact]

    return format_wise_artifact


def get_blur_worflow(path):
    with open(path, 'r') as f:
        worflow_data = f.read()
    blur_workflow = json.loads(worflow_data)
    return blur_workflow


def run_blur_flow(scan_id, blur_format_wise_artifact, blur_workflow, blur_dir):
    
    model_id = blur_workflow["name"]
    workflow_id = blur_workflow["id"]
    blur_input_format = blur_workflow["meta"]["input_format"]
    
    print("blur_dir : ", blur_dir,'\n')

    #blur_paths = glob.glob( os.path.join(blur_dir, '*') )
    #print("Blur files found : ")
    #print(blur_paths)

    print(blur_format_wise_artifact)

    for i, artifact in enumerate(blur_format_wise_artifact):
        input_path = os.path.join(blur_dir, artifact['file'])
        print("input_path to blur: ", input_path, '\n')
        
        source_artifacts_list = [artifact['id']]

        
        target_path = input_path + '_blur.jpg'
        #blur_status = blur_faces_in_file(input_path, target_path)
        blur_img_binary, blur_status = blur_face(input_path)

        if blur_status:
            # post the blur files /file
            # TODO Need to change

            blur_id_from_post_request, post_status = post_files(blur_img_binary)
            #blur_id_from_post_request, post_status = post_files_using_path(target_path)
            #blur_id_from_post_request, post_status = '5850e04c-33e1-11eb-af63-4f5622046249' ,True
            
            print("Post status of uploading the blur file: ", post_status)
            
            if post_status == 201:
                generated_timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
                
                # prepare results
                blur_result = prepare_blur_result_object(scan_id, workflow_id, source_artifacts_list, 
                    blur_id_from_post_request, generated_timestamp)
                print("----------------------------------------------")
                print("Blur Result:")
                #print(blur_result)
                blur_result_string = json.dumps(blur_result, indent=2, separators=(',', ':'))
                blur_result_object = json.loads(blur_result_string)
                pprint.pprint(blur_result_object)
                print("----------------------------------------------")

                # post the blur results using /results
                if post_results(blur_result_object) == 201:
                    print("Result posted successfully for Blur Result No. : ", i)


def run_pose_flow(paths, flow):
    model_id = "posenet_1.0"
    service = "aci-posenet-ind"

    for path in paths:

        # Preprocessing files
        image = preprocessing.posenet_processing(path)

        # Get the results
        results = inference.get_pose_prediction(image, service)
        pose_prediction = json.loads(results)
        
        #prepare the results
        #validate the prepare results with schema
        #post the results


def run_height_flow(paths, flow):

    model_id = 'q3_depthmap_height_run_01' 
    service = 'q3-depthmap-height-run-01-ci'

    # Nik This is dependency
    input_shape = json_metadata[0][0]['input_shape']
    
    # Working on assumption that we are using pcd based height model
    # Prepare the data for prediction
    if flow.format == 'depthmap':
        pcd_numpy = preprocessing.depthmap_to_pcd(paths, self.calibration, 'pointnet', input_shape)
    elif format == 'pcd':
        pcd_numpy = preprocessing.pcd_to_ndarray(paths, input_shape)        
    else:
        print("Incorrect format\n")
        
    cgm_height = inference.get_predictions_2(pcd_numpy, model_id, service)

    height_result_object = {}

    # Prepare Height Results object per scan
    # TODO

    # Validate the Height Result Object
    with open('./schema/height_result_schema.json', 'r') as f:
        schema_data = f.read()
    schema = json.loads(schema_data)

    try:
        jsonschema.validate(height_result_object, schema)
    except jsonschema.exceptions.ValidationError as e:
        print(e, "Result Validation Error\n")
    except jsonschema.exceptions.SchemaError as e:
        print(e, "Schema Error\n")

    # post the height results
    return height_result_object


def run_scan_flow(scan_metadata_obj, blur_workflow_response_path, scan_dir):
    global ARTIFACT_FOR_BLUR_FLOW_STATUS

    # make dir to store scan files in scan folder
    format_wise_artifact = process_scan_metadata(scan_metadata_obj)
    
    print("\nFormat wise list of artifacts\n")
    pprint.pprint(format_wise_artifact)
    
    for artifact_format in format_wise_artifact:
        if not os.path.exists(os.path.join(scan_dir, artifact_format)):
            os.mkdir(os.path.join(scan_dir, artifact_format))

    blur_workflow = get_blur_worflow(blur_workflow_response_path)
    blur_workflow_id = blur_workflow["id"]
    blur_input_format = blur_workflow["meta"]["input_format"]


    if not ARTIFACT_FOR_BLUR_FLOW_STATUS:
        blur_format_wise_artifact = download_blur_flow_artifact(format_wise_artifact, blur_input_format, scan_dir)

    if ARTIFACT_FOR_BLUR_FLOW_STATUS:
        blur_dir = os.path.join(scan_dir, blur_input_format)
        run_blur_flow(scan_metadata_obj["id"], blur_format_wise_artifact, blur_workflow, blur_dir)
    else:
        print("Data not available for Blur Flow\n")

    
    '''First priority is to deliver blur worflow
    
    download_pose_flow_artifact(pose_format, scan_dir)
    if ARTIFACT_FOR_POSE_FLOW_STATUS:
        run_pose_flow(os.path.join(scan_dir, pose_format))
    else:
        print("Data not available for Pose Flow")

    height_format = 'depthmap'
    download_height_flow_artifact(height_format, scan_dir)
    if ARTIFACT_FOR_HEIGHT_FLOW_STATUS:
        run_pose_flow(os.path.join(scan_dir, height_format))
    else:
        print("Data not available for Pose Flow")
    '''


def get_blur_worflow_id(path):
    with open(path, 'r') as f:
        worflow = json.loads(f)

    return workflow['id']


if __name__ == "__main__":
    
    scan_json_path = './schema/scan_with_blur_artifact.json'
    blur_workflow_path = './schema/blur-workflow.json'
    blur_workflow_response_path = './schema/blur-worflow-post.json'

    # This is done in using another script
    # blur_workflow_id, status = post_workflow(blur_workflow_path)

    #blur_workflow = get_blur_worflow(blur_workflow_response_path)

    with open(scan_json_path, 'r') as f:
        scan_metadata = f.read()
    scan_metadata_obj = json.loads(scan_metadata)

    scan_parent_dir = "/home/nikhil/cgm-all/cgm-rg/data/scans/"
    scan_id = scan_metadata_obj["id"]
    scan_dir = os.path.join(scan_parent_dir, scan_id)

    if not os.path.isdir(scan_dir):
        os.mkdir(scan_dir)

    print("scan_dir: ", scan_dir, '\n')

    run_scan_flow(scan_metadata_obj, blur_workflow_response_path, scan_dir)
