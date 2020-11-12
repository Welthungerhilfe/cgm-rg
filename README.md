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


**Master branch**
* Contains production code for CGM-ML
* Release branch will be merged into master branch after passing all checks and tests

**Develop branch**
* Reflects a code base with the latest delivered development changes.
* Feature branch will be branch off using this branch.
* Features branch will be merged into develop branch
* Release branch will be merged into develop branch

**Feature branches**
* Used to develop features
* Branched off from the develop branch
* PR should be created to merge Features to Develop branch and code should be reviewed

**Release branch**
* Used for prepare releases specific to PBI in the current release/sprint.
* Branch off from develop branch
* Merge to master and develop

**Hotfix branch**
* Hotfix branches are very much like release branches but will be used for creating hotfixes, if bugs are encountered in master branch.
* This branch will be be explicitly used for fixes which need be addressed immediately.
* Hotfix branches will be merged to master branch.
* Other bug fixes which are found while development doesn't need to create a hotfix branch and can be fixed in feature branch itself.git