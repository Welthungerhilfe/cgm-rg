import os
import sys
from pathlib import Path

# from azureml.core import Workspace
# from azureml.core.authentication import ServicePrincipalAuthentication
from tensorflow.keras.models import load_model

# To include the config file
sys.path.append(
    os.path.join(
        os.path.dirname(
            os.path.realpath(__file__)),
        os.pardir))

REPO_DIR = Path(os.getenv('APP_DIR', '/app'))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"

# TODO generate the config file
# ws = Workspace.from_config('./ws_config.json')

# TODO load the weights of passed model and generate results for passed
# pointclouds

try:
    height_model = load_model(str(REPO_DIR / 'models/height/outputs/best_model.ckpt'), compile=False)
except OSError as error:
    print(error)
    print("Not able to load the Height model")
except Exception as e:
    print(e)

try:
    weight_model = load_model(
        '/app/models/weight/best_model.ckpt/', compile=False)
except OSError as error:
    print(error)
    print("Not able to load the Weight model")
except Exception as e:
    print(e)

try:
    standing_laying = load_model(str(REPO_DIR / 'models/Standing_laying/best_model.h5'))
except OSError as error:
    print(error)
    print("Not able to load the Standind Laying model")
except Exception as e:
    print(e)

try:
    height_rgbd_model = load_model(str(REPO_DIR / 'models/height_rgbd/best_model.ckpt'), compile=False)
except OSError as error:
    print(error)
    print("Not able to load the rgbd model")
except Exception as e:
    print(e)


def get_height_predictions_local(numpy_array):
    return height_model.predict(numpy_array)

def get_weight_predictions_local(numpy_array):
    return weight_model.predict(numpy_array)

def get_standing_laying_prediction_local(numpy_array):
    return standing_laying.predict(numpy_array)


def get_height_rgbd_prediction_local(numpy_array):
    return height_rgbd_model.predict(numpy_array)


def get_ensemble_height_predictions_local(model_path, numpy_array):
    model_path += '/outputs/best_model.ckpt/'
    model = load_model(model_path, compile=False)
    return model.predict(numpy_array)
