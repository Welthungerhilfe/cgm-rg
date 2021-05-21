[![codecov](https://codecov.io/gh/Welthungerhilfe/cgm-rg/branch/develop/graph/badge.svg?token=W9UYS0I78M)](https://codecov.io/gh/Welthungerhilfe/cgm-rg)

# Child Growth Monitor Result Generation

[Child Growth Monitor (CGM)](https://childgrowthmonitor.org) is a
game-changing app to detect malnutrition. If you have questions about the project, reach out to `info@childgrowthmonitor.org`.

## Introduction

Result Generation is performed when the Scanner App take an scan and upload using cgm-api. Result Generation plays an central role in generation of the accurate prediction of Height, Weight and MUAC for the scans collected from the scanner app. These generated results will be used by tagging tool to inspect the scans and return the results to users which are accuracte.

For detailed explaination of data flow, one may look into [wiki introduction](https://github.com/Welthungerhilfe/cgm-rg/wiki)

## Guide to setup RG

Please go through wiki pages for [detailed setup of local development](https://github.com/Welthungerhilfe/cgm-rg/wiki/Setup-local-development).

## Branch Policy

| Name    | Description                                                                                                                                                                                                                                                                                         |
| ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| master  | The master branch tracks released code only. The only commits to master are merges from release branches and hotfix branches.                                                                                                                                                                       |
| release | The code in the release branch is deployed onto a suitable test environment, tested, and any problems are fixed                                                                                                                                                                                     |
| develop | As new development is completed, it gets merged back into the develop branch, which is a staging area for all completed features that haven’t yet been released. So when the next release is branched off of develop, it will automatically contain all of the new features that has been finished. |
| feature | The master branch tracks released code only. The only commits to master are merges from release branches and hotfix branches.                                                                                                                                                                       |

For more detailed explaination, please go through [branch policy document](https://github.com/Welthungerhilfe/cgm-rg/wiki/Branch-Policy)

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
