# HDSI Faculty Exploration Tool

This repository contains project code for experimenting with LDA for Faculty Information Retrieval System.

## Running the Project
* All of the below lines should be run within a terminal:

* Before running any of the below commands, launch the docker image by running `launch.sh -i duxiang/dsc180a:latest`

* To get the preprocessed data file, run `python run.py process_data`
* To get the fitted sklearn.lda model, run `python run.py model`
* To prepare/update the dashboard, run `python run.py prepare_sankey`
* To run the live dashboard, run `python run.py run_dashboard`

## Using the Dashboard
* When executing `run_dashboard`, it will launch dash with a locally hosted port.
* It would require port-forwarding on a remote server.

# Web Link
Website: https://marthay01.github.io/hdsi_faculty_tool/
