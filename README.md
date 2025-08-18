# README

## Overview
This repository contains the Python implementation of the algorithms and experiments presented in our paper "**Harnessing Individual Motivation for Collective Efficiency: A Mechanism-Driven Distributed Optimization Method**".
The code is designed to demonstrate various functionalities, including example creation, comparison of mechanism design, evaluation of distributed algorithm performance, and assessment of code acceleration techniques.

## Requirements

This code was tested under Python 3.11.7. To run this code, you need to install some packages.

	numpy==1.26.4
	matplotlib==3.8.0
	networkx==3.1
	gurobipy==11.0.3
	scipy==1.11.4
	pandas==2.1.4

## Run

The code provides the following main functionalities: mechanism design comparison, distributed algorithms comparison and algorithm acceleration comparison.

### Mechanism design comparison
We compare the performance between two machanisms： shadow price mechanism and VCG mechanism. We achieve the two mechanisms and run some tasks in `Mech_related.py`, and plot the figure in `Mech_draw.py`.
The complete pipeline is implemented in `mechanism_design_comparison(examplenow)` function in `Main.py`.
To test this component, you can run the following code in `main.py`:

	# examplenow = 1 for mechanism_design_comparison, contains preconfigured Parameters.
	mechanism_design_comparison(examplenow = 1)


### Distributed algorithms comparison
Compare our distributed algorithm's performance with other algorithm. We choose four algorithms: Tracking-ADMM, DPMM, IPLUX, DC-ADMM. We achieve these algorithms in `TrackingADMM.py`, `DPMM.py`, `IPLUX.py`, `DCADMM.py` respectively, and acheive ours in `Ours.py`.
The complete pipeline is implemented in `distributed_alg_comparison(examplenow)` function in `Main.py`.
To test this component, you can run the following code in `main.py`:

	# examplenow = 1/2/3 for distributed_alg_comparison, contains preconfigured Parameters, representing small-scale, midium-scale, large-scale scenarios respectively.
	distributed_alg_comparison(examplenow = k)

### Algorithm acceleration comparison
Compare the performance between Algorithm 1 (Consensus-Tracking-ADMM) and Algorithm 2 (Improved-Consensus-Tracking-ADMM).
The complete pipeline is implemented in `acceleration_comparison(examplenow)` function in `Main.py`.
To test this component, you can run the following code in `main.py`:

	# examplenow = 4/5/6 for acceleration comparison, contains preconfigured Parameters, representing small-scale, midium-scale, large-scale scenarios respectively.
