from pathlib import Path
import os

from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.model import Model

REPO_DIR = Path(__file__).parents[1].absolute()

sp = ServicePrincipalAuthentication(
    tenant_id=os.environ['TENANT_ID'],
    service_principal_id=os.environ['SP_ID'],
    service_principal_password=os.environ['SP_PASSWD'])

ws = Workspace(
    subscription_id=os.environ['SUB_ID'],
    resource_group="cgm-ml-prod-we-rg",
    workspace_name="cgm-ml-prod-we-azml",
    auth=sp
)

# ws = Workspace.from_config()

standing_laying = Model(ws, name='standing_laying_classifier')
print(standing_laying)
standing_laying.download(target_dir=REPO_DIR / 'models')
print("Model Succesfully downloaded")
