# cgm-rg
Code for result generation


**Branching Policy**

**Master branch**
    Contains production code for CGM-ML
    Release branch will be merged into master branch after passing all checks and tests

**Develop branch**
    Reflects a code base with the latest delivered development changes.
	Feature branch will be branch off using this branch.
	Features branch will be merged into develop branch
	Release branch will be merged into develop branch

**Feature branches**
    Used to develop features
    Branched off from the develop branch
    PR should be created to merge Features to Develop branch and code should be reviewed

**Release branch**
    Used for prepare releases specific to PBI in the current release/sprint.
	Branch off from develop branch
	Merge to master and develop

**Hotfix branch**
    Hotfix branches are very much like release branches but will be used for creating hotfixes, if bugs are encountered in master branch.
	This branch will be be explicitly used for fixes which need be addressed immediately.
	Hotfix branches will be merged to master branch.
	Other bug fixes which are found while development doesn't need to create a hotfix branch and can be fixed in feature branch itself.