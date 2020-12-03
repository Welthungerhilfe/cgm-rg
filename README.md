# Child Growth Monitor Result Generation

[Child Growth Monitor (CGM)](https://childgrowthmonitor.org) is a
game-changing app to detect malnutrition. If you have questions about the project, reach out to `info@childgrowthmonitor.org`.

## Introduction
Result Generation is performed when the Scanner App take an scan and upload it to datastore. Result Generation will produce results and update the results to datastore and these generated results will be used by tagging tool to inspect the scans and return the results to users which are accuracte.


## Steps to start RG



### Below are the command to run using the azure vm
# get the ip address of the local machine

```sh
docker build --tag etlservice:1.0 --file ./Dockerfile .

docker run -it --add-host="localhost:'YOUR IP ADDRESS'" --name rgservice_1_0 rgservice:1.0
```

### Steps to run RG in the container instance in Azure.

```
Step 1: Build and push the RG image with latest code in azure container registry. Use the existing 'cgm-rg-release' pipeline which is already setup in the Azure Devops Releases for build and push the RG image
Step 2: Use the latest image in azure container registry cgmrgtest.azurecr.io/welthungerhilfe and run a container instance.
```


**Branch Policy**

| Name | Description |
| --- | --- |
| master | The master branch tracks released code only. The only commits to master are merges from release branches and hotfix branches. |
| release | The code in the release branch is deployed onto a suitable test environment, tested, and any problems are fixed |
| develop | As new development is completed, it gets merged back into the develop branch, which is a staging area for all completed features that haven’t yet been released. So when the next release is branched off of develop, it will automatically contain all of the new features that has been finished. |
| feature | The master branch tracks released code only. The only commits to master are merges from release branches and hotfix branches. |

For more detailed explaination, please go through [branch policy document](BRANCH_POLICY.md)