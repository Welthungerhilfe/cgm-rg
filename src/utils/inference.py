import os
import sys
import pathlib
# from azureml.core import Workspace
# from azureml.core.authentication import ServicePrincipalAuthentication
from tensorflow.keras.models import load_model, Model
import tensorflow as tf 
import numpy as np
import cv2

import preprocessing as preprocessing  # noqa: E402


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
        '/app/models/height/best_model.ckpt/', compile=False)
except OSError as error:
    print(error)
    print("Not able to load the Height model")
except Exception as e:
    print(e)

try:
    standing_laying = load_model('/app/models/Standing_laying/best_model.h5')
except OSError as error:
    print(error)
    print("Not able to load the Standind Laying model")
except Exception as e:
    print(e)

try:
    height_rgbd_model = load_model(
        '/app/models/height_rgbd/best_model.ckpt', compile=False)
except OSError as error:
    print(error)
    print("Not able to load the rgbd model")
except Exception as e:
    print(e)


# where should this function be stored?
def extract_last_conv_layer_name(model, substring='conv'):
    # 1. save all layer names in a list
    layer_names = [layer.name for layer in model.layers]
    # 2. search for substring in all layers and put them in list
    conv_layer_names = []
    for layer in layer_names:
        if substring in layer:
            conv_layer_names.append(layer)
    # 3. return last one
    return conv_layer_names[-1]


# should this all be global?
last_conv_layer_name = extract_last_conv_layer_name(height_model)
grad_model = Model(inputs=[height_model.inputs], 
                        outputs=[height_model.get_layer(last_conv_layer_name).output, height_model.output])


def get_height_predictions_local(numpy_array):
    return height_model.predict(numpy_array)


def get_standing_laying_prediction_local(numpy_array):
    return standing_laying.predict(numpy_array)


def get_height_rgbd_prediction_local(numpy_array):
    return height_rgbd_model.predict(numpy_array)


def get_ensemble_height_predictions_local(model_path, numpy_array):
    model_path += '/outputs/best_model.ckpt/'
    model = load_model(model_path, compile=False)
    return model.predict(numpy_array)


def get_height_prediction_and_heatmap_local(numpy_array):
    # GET the score for target prediction
    with tf.GradientTape() as tape:
        numpy_array = tf.cast(numpy_array, tf.float32)
        (conv_outputs, height_prediction) = grad_model(np.array(numpy_array))
    # EXTRACT filters and gradients
    output = conv_outputs[0]
    grads = tape.gradient(height_prediction, conv_outputs)[0]
    # GUIDED backpropagation - eliminating elements that act negatively towards the decision - zeroing-out negative gradients
    guided_grads = tf.cast(output > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads
    # AVERAGE gradients spatially
    weights = tf.reduce_mean(guided_grads, axis=(0, 1))
    # BUILD a weighted map of filters according to gradients importance
    cam = np.ones(output.shape[0: 2], dtype = np.float32)
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]
    # HEATMAP visualization
    cam = cv2.resize(cam.numpy(), (preprocessing.IMAGE_TARGET_HEIGHT, preprocessing.IMAGE_TARGET_WIDTH))
    cam = np.maximum(cam, 0)
    heatmap = (cam - cam.min()) / (cam.max() - cam.min())
    return heatmap, height_prediction