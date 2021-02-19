Overview
========

Thank you for agreeing to help me generate data for my work. Your cooperation is greatly appreciated.
  
In these experiments I am gathering data about driving behaviours in highway environments. My experiments are focused on the execution of lane change manoeuvres
in a scenario with one other car on the road. 
All the experiments will take place on a straight road 2-lane highway environment (both lanes going the same direction).
At different points in the experiments you will be required to assume different roles in the lane change; either performing the lane change, or keeping your lane while the other car changes lanes.


Installation
============

This repository uses submodules, so you will need to clone it with the following command:

```
git clone --recursive https://github.com/Justme21/Conflict_Data_Collection_Experiments.git
```

Python 3 is required to run the experiments. To ensure you are able to run all the experiments, please look at the `requirements.txt` file, to make sure you have all the relevant libraries installed. You can also use the following commands in a UNIX-like terminal to install the required packages, provided you already have python 3 installed:

```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Getting Started with the Sandbox
================================
In order to familiarise yourself with the simulator controls, feel free to use the sandbox environment provided in sandbox.py (run using the terminal command `python3 sandbox.py`)
This is a 2-lane highway environment with a stationary car to navigate around.
No data is stored from executions on the sandbox. The scenario will restart until you close the window, so take as long as you need to get comfortable with the controls.
Once you are ready to continue, close the sandbox window. 


The Experiments
===============
In total you are participating in 2 experiments, labelled "exp1" and "exp2".
Experiment 1 (exp1) examines different ways a lane change can be executed. This experiment has two parts:

* exp1a: where you are performing a lane change, while the other car stays in their lane.
* exp2a: where you are keeping your lane, while the other car performs a lane change.

Each sub-experiment is in a different directory, and details regarding each sub-experiment can be found in `.txt` files found in the relevant directories.
This experiment will involve repeated executions of each scenario, subject to changing sub-objectives (see the relevant `.txt` files for more information).

To access these experiments move to the `exp1` directory, and then into the `a` and `b` sub-directories. In each sub-directory there is a `.txt` file providing instructions on what the experiment entails,
and a `.py` file that contains the experiment code. After reading the `.txt` file, run the `.py` file using the command `python3 <filename>.py`.

Experiment 2 (exp2) examines how sub-objectives affect manoeuvre execution, and details regarding this experiment can be found in a `.txt` file in the exp2 directory.
This experiment will involve a single execution of each scenario, subject to changing sub-objecives (see the relevant `.txt` file for more information).

To access this experiment move to the `exp2` directory. In this directory there is a `.txt` file providing instructions on what the experiment entails,
and a `.py` file that contains the experiment code. After reading the `.txt` file, run the `.py` file using the command `python3 <filename>.py`.

You MUST complete Experiment 1 BEFORE doing Experiment 2.
You do not need to do exp1a,exp1b and exp2 in the same sitting, and a break to recharge between experiment sections is permitted.
It is also possible to pause experiments during their execution, if a break is required.

Each experiment in Experiment 1 is expected to take about 20 minutes.
Experiment 2 is expected to take only a few minutes.

Submitting the Results
======================
Once you have completed both Experiment 1 and Experiment 2, please rename the folder from "results" to a different title (which can be whatever you like) and upload the file to:
https://drive.google.com/drive/folders/1x87aWfRLnZFmCvl4Cbs0-3fyky3nCKQ2?usp=sharing

Thank you again for your participation.


