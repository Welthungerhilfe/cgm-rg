# Child Growth Monitor Result Generation

[Child Growth Monitor (CGM)](https://childgrowthmonitor.org) is a
game-changing app to detect malnutrition. If you have questions about the project, reach out to `info@childgrowthmonitor.org`.

## Introduction
Result Generation is performed when the Scanner App take an scan and upload it to datastore. Result Generation will produce results and update the results to datastore and these generated results will be used by tagging tool to inspect the scans and return the results to users which are accuracte.


**Branch Policy**

| Name | Description |
| --- | --- |
| master | The master branch tracks released code only. The only commits to master are merges from release branches and hotfix branches. |
| release | The code in the release branch is deployed onto a suitable test environment, tested, and any problems are fixed |
| develop | As new development is completed, it gets merged back into the develop branch, which is a staging area for all completed features that haven’t yet been released. So when the next release is branched off of develop, it will automatically contain all of the new features that has been finished. |
| feature | The master branch tracks released code only. The only commits to master are merges from release branches and hotfix branches. |

For more detailed explaination, please go through [branch policy document](BRANCH_POLICY.md)