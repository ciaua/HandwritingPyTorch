(Handwriting) Unconditional generation, Conditional generation, and Recognition with PyTorch
============================================================================================

The unconditoinal and conditional generation are based on "Generating Sequences With Recurrent Neural Networks" by Alex Graves (https://arxiv.org/pdf/1308.0850.pdf). The models are implemented with GRUs instead of LSTMs.

This repository contains the trained models of the three tasks.


(Task 1) Unconditional generation (Goal: Nothing => hand writing)
-----------------------------------------------------------------




(Task 2) Conditional generation (Goal: text => hand writing)
------------------------------------------------------------



(Task 3) Recognition (Goal: hand writing => text)
-------------------------------------------------
I devised a method where the conditional generation (Task 2) provides assistance to the recognition model.


Dependencies
------------
Python==3.7
PyTorch==0.4
Numpy
Scipy
