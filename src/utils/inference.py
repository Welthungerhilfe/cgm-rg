import os
import sys
import pathlib
# from azureml.core import Workspace
# from azureml.core.authentication import ServicePrincipalAuthentication
from tensorflow.keras.models import load_model

# To include the config file
sys.path.append(
    os.path.join(
        os.path.dirname(
            os.path.realpath(__file__)),
        os.pardir))

current_working_directory = pathlib.Path.cwd()
models_path = current_working_directory.joinpath('models')

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"

# TODO generate the config file
# ws = Workspace.from_config('./ws_config.json')

# TODO load the weights of passed model and generate results for passed
# pointclouds

try:
    height_model = load_model(
        '/app/models/height/outputs/best_model.ckpt/', compile=False)
except OSError as error:
    print(error)
    print("Not able to load the Height model")
except Exception as e:
    print(e)

try:
    depthmapmultiartifactlatefusion_height_model = load_model(
        '/app/models/depthmapmultiartifactlatefusion/outputs/best_model.ckpt', compile=False)
except OSError as error:
    print(error)
    print("Not able to load the depthmapmultiartifactlatefusion Height model")
except Exception as e:
    print(e)

try:
    weight_model = load_model(
        '/app/models/weight/outputs/best_model.ckpt/', compile=False)
except OSError as error:
    print(error)
    print("Not able to load the Weight model")
except Exception as e:
    print(e)

try:
    standing_laying = load_model('/app/models/Standing_laying/best_model.h5')
except OSError as error:
    print(error)
    print("Not able to load the Standind Laying model")
except Exception as e:
    print(e)


def get_height_predictions_local(numpy_array):
    return height_model.predict(numpy_array)


def get_weight_predictions_local(numpy_array):
    return weight_model.predict(numpy_array)


def get_standing_laying_prediction_local(numpy_array):
    return standing_laying.predict(numpy_array)


def get_depthmapmultiartifactlatefusion_height_predictions_local(numpy_array):
    return depthmapmultiartifactlatefusion_height_model.predict(numpy_array)


'''
def get_predictions(numpy_array, service_name):

    sp = ServicePrincipalAuthentication(
        tenant_id=os.environ['TENANT_ID'],
        service_principal_id=os.environ['SP_ID'],
        service_principal_password=os.environ['SP_PASSWD'])

    ws = Workspace.get(name="cgm-azureml-prod",
                       auth=sp,
                       subscription_id=os.environ['SUB_ID'])

    service = ws.webservices[service_name]
    max_size = 20
    predictions = []
    for i in range(0, len(numpy_array), max_size):
        print(i, min(len(numpy_array), i + max_size))
        pcd_numpy = numpy_array[i:i + max_size]
        # service = ws.webservices[service_name]
        pointcloud_json = json.dumps({'data': pcd_numpy.tolist()})
        prediction = service.run(input_data=pointcloud_json)
        predictions += prediction

    return predictions
'''
