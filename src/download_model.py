import os
from pathlib import Path

from azureml.core import Experiment, Run, Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.model import Model

REPO_DIR = Path(__file__).parents[1].absolute()

ENSEMBLE_RUN_IDS = ['q1-ensemble-warmup_1610544610_eb44bfe2', 'q1-ensemble-warmup_1610547587_7ca932c3',
                    'q1-ensemble-warmup_1610547669_5b789bd1', 'q1-ensemble-warmup_1610547705_f2141d0f']
'''
'q1-ensemble-warmup_1610547744_d2b42ce5', 'q1-ensemble-warmup_1610547780_2f000a25',
'q1-ensemble-warmup_1610547816_c3f815df', 'q1-ensemble-warmup_1610547892_8ee6ff49',
'q1-ensemble-warmup_1610547928_b9519b6a', 'q1-ensemble-warmup_1610547986_ad0186b8',
'q1-ensemble-warmup_1610548023_99ac6060', 'q1-ensemble-warmup_1610548064_afefd4e4',
'q1-ensemble-warmup_1610548106_69993d24', 'q1-ensemble-warmup_1610548137_a8c52d63',
'q1-ensemble-warmup_1610548168_914ce1f6', 'q1-ensemble-warmup_1610548209_9692a253'
'''


def download_model(ws, experiment_name, run_id, input_location, output_location):  # noqa :E501
    """Download the pretrained model

    Args:
         ws: workspace to access the experiment
         experiment_name: Name of the experiment in which model is saved
         run_id: Run Id of the experiment in which model is pre-trained
         input_location: Input location in a RUN Id
         output_location: Location for saving the model
    """
    experiment = Experiment(workspace=ws, name=experiment_name)
    # Download the model on which evaluation need to be done
    run = Run(experiment, run_id=run_id)
    if input_location.endswith(".h5"):
        run.download_file(input_location, output_location)
    elif input_location.endswith(".ckpt"):
        run.download_files(prefix=input_location,
                           output_directory=output_location)
    else:
        raise NameError(f"{input_location}'s path extension not supported")
    print("Successfully downloaded model")


def main():
    sp = ServicePrincipalAuthentication(
        tenant_id=os.environ['TENANT_ID'],
        service_principal_id=os.environ['SP_ID'],
        service_principal_password=os.environ['SP_PASSWD']
    )

    ws = Workspace(
        subscription_id=os.environ['SUB_ID'],
        resource_group="cgm-ml-prod-we-rg",
        workspace_name="cgm-ml-prod-we-azml",
        auth=sp
    )

    # Downlaod model for standing/laying
    standing_laying = Model(ws, name='standing_laying_classifier')
    standing_laying.download(target_dir=REPO_DIR / 'models')
    print("Model Succesfully downloaded")

    # Downlaod model for height
    download_model(ws=ws,
                   experiment_name='q3-depthmap-plaincnn-height-95k',
                   run_id='q3-depthmap-plaincnn-height-95k_1610709896_ef7f755d',  # noqa :E501
                   input_location=os.path.join('outputs', 'best_model.ckpt'),
                   output_location=REPO_DIR / 'models/height')

    # Downlaod model for  weight
    download_model(ws=ws,
                   experiment_name='q4-depthmap-plaincnn-weight-95k',
                   run_id='q4-depthmap-plaincnn-weight-95k_1611336518_642a9c58',  # noqa :E501
                   input_location=os.path.join('outputs', 'best_model.ckpt'),
                   output_location=REPO_DIR / 'models/weight')

    # Download M-CNN Model for height
    download_model(ws=ws,
                   experiment_name='q3-depthmapmultiartifactlatefusion-plaincnn-height-95',  # noqa :E501
                   run_id='q3-depthmapmultiartifactlatefusion-plaincnn-height-95k_1614177517_ecd7b6e2',  # noqa :E501
                   input_location=os.path.join('outputs', 'best_model.ckpt'),
                   output_location=REPO_DIR / 'models/depthmapmultiartifactlatefusion')  # noqa :E501

    # Downlaod model for RGBD
    download_model(ws=ws,
                   experiment_name='2021q2-rgbd-plaincnn-height-5kscans',
                   run_id='2021q2-rgbd-plaincnn-height-5kscans_1616835920_c469620e',  # noqa :E501
                   input_location=os.path.join('outputs', 'best_model.ckpt'),
                   output_location=REPO_DIR / 'models/height_rgbd')

    for id in ENSEMBLE_RUN_IDS:
        print(f"Downloading run {id}")
        download_model(
            ws=ws,
            experiment_name='q1-ensemble-warmup',
            run_id=id,
            input_location=os.path.join('outputs', 'best_model.ckpt'),
            output_location=REPO_DIR / 'models/deepensemble' / id)


if __name__ == "__main__":
    main()
