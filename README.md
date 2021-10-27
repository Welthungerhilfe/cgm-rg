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

## Local Setup

Follow the steps in this wiki page: [detailed setup of local development](https://github.com/Welthungerhilfe/cgm-rg/wiki/Setup-local-development).

### Models

For some parts of the code to work, you need ML models.
You can either build them yourself using [cgm-ml](https://github.com/Welthungerhilfe/cgm-ml).
However, for internal contributors we recommend to download existing models as follows:
* Note that `src/download_model.py` helps to download models and uses environment variables
*

## Branch Policy

| Name    | Description                                                                                                                                                                                                                                                                                         |
| ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| main  | The master branch tracks released code only. The only commits to master are merges from release branches and hotfix branches.                                                                                                                                                                       |
| release | The code in the release branch is deployed onto a suitable test environment, tested, and any problems are fixed                                                                                                                                                                                     |
| develop | As new development is completed, it gets merged back into the develop branch, which is a staging area for all completed features that haven’t yet been released. So when the next release is branched off of develop, it will automatically contain all of the new features that has been finished. |
| feature | The master branch tracks released code only. The only commits to master are merges from release branches and hotfix branches.                                                                                                                                                                       |

For more detailed explaination, please go through [branch policy document](https://github.com/Welthungerhilfe/cgm-rg/wiki/Branch-Policy) and this [internal documentation](https://dev.azure.com/cgmorg/ChildGrowthMonitor/_wiki/wikis/ChildGrowthMonitor.wiki/115/-InProgress-Branching-strategy)

## Versioning

This project is versioned according to CGM [versioning rules](https://dev.azure.com/cgmorg/ChildGrowthMonitor/_wiki/wikis/ChildGrowthMonitor.wiki/185/Versioning-Release-management). The version of the project
can be found in `VERSION`. The CI pipeline tags and pushes a patch version update automatically
after a merge to the `main` branch.

The following utility scripts can be used to manipulate the version (requires [bump2version](https://pypi.org/project/bump2version/)):
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
