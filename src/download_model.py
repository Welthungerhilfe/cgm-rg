import os
from pathlib import Path

from azureml.core import Workspace
#from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.model import Model

REPO_DIR = Path(__file__).parents[1].absolute()

'''
sp = ServicePrincipalAuthentication(
        tenant_id=os.environ['TENANT_ID'],
        service_principal_id=os.environ['SP_ID'],
        service_principal_password=os.environ['SP_PASSWD'])

ws = Workspace.get(name="cgm-azureml-prod",
                    auth=sp,
                    subscription_id=os.environ['SUB_ID'])
'''
ws = Workspace.from_config()

standing_laying = Model(ws, name='standing_laying_classifier')
print(standing_laying)
standing_laying.download(
    target_dir=REPO_DIR / 'models', exist_ok=True, exists_ok=None)
