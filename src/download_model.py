import os
from pathlib import Path

from azureml.core import Experiment, Run, Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.model import Model

REPO_DIR = Path(__file__).parents[1].absolute()


def download_model(ws, experiment_name, run_id, input_location, output_location):
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
                   run_id='q3-depthmap-plaincnn-height-95k_1610709896_ef7f755d',
                   input_location=os.path.join('outputs', 'best_model.ckpt'),
                   output_location=REPO_DIR / 'models/height')

    # Downlaod model for  weight
    download_model(ws=ws,
                   experiment_name='q4-depthmap-plaincnn-weight-95k',
                   run_id='q4-depthmap-plaincnn-weight-95k_1611336518_642a9c58',
                   input_location=os.path.join('outputs', 'best_model.ckpt'),
                   output_location=REPO_DIR / 'models/weight')


if __name__ == "__main__":
    main()
