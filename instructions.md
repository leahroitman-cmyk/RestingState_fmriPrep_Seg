#get files 
git clone https://github.com/OpenNeuroDatasets/ds004564.git

# install large files (user files) (have to cd into ds004564 directory "cd ds004564)
datalad get -r sub-17017

# install large files (user files) (have to cd into ds004564 directory "cd ds004564) for all users
datalad get -r .

# for one user (17017)
docker compose run --rm fmriprep --participant-label 17017 --output-spaces MNI152NLin2009cAsym:res-2

# for all users:
docker compose run --rm fmriprep --output-spaces MNI152NLin2009cAsym:res-2

