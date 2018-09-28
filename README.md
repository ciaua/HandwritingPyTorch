(Handwriting) Unconditional generation, Conditional generation, and Recognition with PyTorch
============================================================================================

![Generated](/images/Hello, this is Jen-Yu Liu.png)

The unconditoinal and conditional generation are based on "Generating Sequences With Recurrent Neural Networks" by Alex Graves (https://arxiv.org/pdf/1308.0850.pdf). The models are implemented with GRUs instead of LSTMs.

The recogniton model is trained with the assistance from the conditional model. See below for more details.

This repository contains the trained models of the three tasks. The main code and trained models are in `models`. Main code: handwriting.py


(Task 1) Unconditional generation (Goal: Nothing => hand writing)
-----------------------------------------------------------------
In `models/handwriting.py`:
UncondStackedGRUCell: a module processing information at a given timestep in a sequence
UncondNet: a modeule processing a sequence, using UncondStackedGRUCell as a submodule
setup_unconditional_model: load the model
generate_unconditionally: generate


(Task 2) Conditional generation (Goal: text => hand writing)
------------------------------------------------------------
In `models/handwriting.py`:
CondStackedGRUCell: a module processing information at a given timestep in a sequence
CondNet: a modeule processing a sequence, using CondStackedGRUCell as a submodule
setup_conditional_model: load the model
generate_conditionally: generate


(Task 3) Recognition (Goal: hand writing => text)
-------------------------------------------------
In `models/handwriting.py`:
RecogNet: the recognition module 
setup_recognition_mode: load the model
recognize_stroke: recognize

I devised a method where the conditional generation (Task 2) provides assistance to the recognition model. 

I use the `phi` produced by the conditional model to provide training targets. In Alex Graves's paper, the `phi` is used as weights for the soft window that measures the importance of the chars at a given stroke. This means that the `phi` can tell us what char we are processing at a given stroke. By picking the char with the maximum `phi` value at each stroke, a training target with the same length as the strokes can be derived. In this way, we do not have to align the text and the strokes, and we can train the model pretty straightforward.

For example, a derived target may look like this:
Text: 'I am fine'
Char target: IIIII  aaammmmm    ffffffiiiiiiinnnnnneeeee

In addition to the vector specifying the characters (char target), we can also derive a vector specifying where to cut the chars (cut target).
Char target: IIIII  aaammmmm    ffffffiiiiiiinnnnnneeeee
Cut  target: 0000010100100001000100000100000010000010000

The recognition model will have two heads that predict both of the chars and cuts.
 The cuts will also be used to estimate the number of chars in the text. See recognize_stroke and RecogNet in models/handwriting.py for more details.

I want the model to also know where to cut for cases like "zoo" or "happen" that contains chars that appear twice consecutively. However, it turns out that these consecutive chars are still difficult for the model to recognize.


Dependencies
------------
Python==3.7
PyTorch==0.4
Numpy
Scipy
