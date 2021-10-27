[![codecov](https://codecov.io/gh/Welthungerhilfe/cgm-rg/branch/develop/graph/badge.svg?token=W9UYS0I78M)](https://codecov.io/gh/Welthungerhilfe/cgm-rg)

# Child Growth Monitor Result Generation

[Child Growth Monitor (CGM)](https://childgrowthmonitor.org) is a
game-changing app to detect malnutrition. If you have questions about the project, reach out to `info@childgrowthmonitor.org`.

## Introduction

Result Generation is performed when the Scanner App takes a scan and uploads it using cgm-api.
Result Generation plays a central role in generation of the accurate prediction of height,
weight and MUAC for the scans collected from the scanner app.
These generated results will be used by the tagging tool
to inspect the scans and return the results to users which are accuracte.

For detailed explaination of data flow, one may look into [wiki introduction](https://github.com/Welthungerhilfe/cgm-rg/wiki)

Supported Python versions:
- `3.6`
- maybe `3.7`

Branch Policy: See [branch policy document](https://github.com/Welthungerhilfe/cgm-rg/wiki/Branch-Policy) and this [internal documentation](https://dev.azure.com/cgmorg/ChildGrowthMonitor/_wiki/wikis/ChildGrowthMonitor.wiki/115/-InProgress-Branching-strategy)

## Local Setup

Follow the steps in this wiki page: [detailed setup of local development](https://github.com/Welthungerhilfe/cgm-rg/wiki/Setup-local-development).

### Models

For some parts of the code to work, you need ML models.
You can either build them yourself using [cgm-ml](https://github.com/Welthungerhilfe/cgm-ml).
However, for internal contributors we recommend to download existing models as follows:
* Note that `src/download_model.py` helps to download models and uses environment variables
* Internal contributors can get the environment variables, e.g. for the sandbox environment:
    * Go to <portal.azure.com>
    * Find the ACI (Azure container instance) for your environment, e.g. `cgmbecidevrgaci` for sandbox
    * On the left (in the `Settings` section), click `Containers`
    * Go to the `Properties` tab and copy the needed environment variables from here
    * Put the enviromentment into a file called `env.list`. Example (make sure to replace with actual variable values)

    ```
    SUB_ID=00000000-1111-2222-3333-444444444444
    TENANT_ID=00000000-1111-2222-3333-444444444444
    SP_ID=00000000-1111-2222-3333-444444444444
    SP_PASSWD=00000000-1111-2222-3333-444444444444
    ```

It can be helpful to run the local development using this command
```bash
docker run -it --rm --env-file ./env.list \
    -v $(pwd)/src:/app/src \
    -v $(pwd)/models:/app/models \
    --add-host="localhost:<YOUR IP>" \
    --name rg_service_1_0 rgservice:1.0
```

## Versioning

This project is versioned according to CGM
[versioning rules](https://dev.azure.com/cgmorg/ChildGrowthMonitor/_wiki/wikis/ChildGrowthMonitor.wiki/185/Versioning-Release-management).
The version of the project can be found in a file called `VERSION`.
The CI pipeline tags and pushes a patch version update automatically after a merge to the `main` branch.

The following utility scripts can be used to manipulate the version
(requires [bump2version](https://pypi.org/project/bump2version/)):
```shell
# install bump2version
pip install bump2version

# Get the current version
head -n 1 VERSION

# Bump minor version (use major or patch to bump respective parts)
bumpversion minor

# Set a specific version. The `patch` argument does not affect the verions but is required.
bumpversion --new-version <version> patch
```
These utility command do not commit or tag the repository.
