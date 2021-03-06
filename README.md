(Handwriting) Unconditional generation, Conditional generation, and Recognition with PyTorch
============================================================================================

![Generated](images/sample.png)

The unconditoinal and conditional generation are based on "Generating Sequences With Recurrent Neural Networks" by Alex Graves (https://arxiv.org/pdf/1308.0850.pdf). The models are implemented with GRUs instead of LSTMs.

The recogniton model is trained with the help from the conditional model. See below for more details.

This repository contains the trained models of the three tasks. The main code (handwriting.py) and trained models are in `models`. 


(Task 1) Unconditional generation (Goal: Nothing => hand writing)
-----------------------------------------------------------------
In `models/handwriting.py`:
* UncondStackedGRUCell: a module processing information at a given timestep in a sequence
* UncondNet: a modeule processing a sequence, using UncondStackedGRUCell as a submodule
* setup_unconditional_model: load the model
* generate_unconditionally: generate


(Task 2) Conditional generation (Goal: text => hand writing)
------------------------------------------------------------
In `models/handwriting.py`:
* CondStackedGRUCell: a module processing information at a given timestep in a sequence
* CondNet: a modeule processing a sequence, using CondStackedGRUCell as a submodule
* setup_conditional_model: load the model
* generate_conditionally: generate

This model is similar to the first one. It predicts an additional soft window to condition the strokes.


(Task 3) Recognition (Goal: hand writing => text)
-------------------------------------------------
In `models/handwriting.py`:
* RecogNet: the recognition module 
* setup_recognition_mode: load the model
* recognize_stroke: recognize

I devised a method where the conditional generation (Task 2) provides assistance to the recognition model. 

I use the `phi` produced by the conditional model to provide training targets. In Alex Graves's paper, the `phi` is used as weights for the soft window that measures the importance of the chars at a given stroke. This means that the `phi` can tell us what char we are processing at a given stroke. By picking the char with the maximum `phi` value at each stroke, a training target with the same length as the strokes can be derived. In this way, we do not have to align the text and the strokes, and we can train the model pretty straightforward.

For example, a derived target may look like this:

```
Text: 'I am fine'  
Char target: IIIII  aaammmmm    ffffffiiiiiiinnnnnneeeee
```

In addition to the vector specifying the characters (char target), we can also derive a vector specifying where to cut the chars (cut target).

```
Char target: IIIII  aaammmmm    ffffffiiiiiiinnnnnneeeee  
Cut  target: 0000010100100001000100000100000010000010000
```

The recognition model will have two heads that predict both of the chars and cuts.
 The cuts will also be used to estimate the number of chars in the text. See recognize_stroke and RecogNet in models/handwriting.py for more details.

I want the model to also know where to cut for cases like "zoo" and "happen" where the same char appearing consecutively ("oo" or "pp" in the examples). However, it turns out that these consecutive chars are still difficult for the model to recognize.

In the evaluation phase, first the cut prediction is used to determined several segments. Note that the raw cut predictions will be in the range [0, 1], while we need binary decisions. A location is selected as a cut if 1) its cut prediction is higher than a given threshold (`cut_threshold`) and 2) it is a local maximum. These selected locations are set to 1, and other locations are set to 0.  Let's see an example:
```
Char  prediction: IIII   aaaaaammmmm    ffffffiiiiiinnnnnneeeeee  
Cut   prediction: 0001000100100001000100000100000010000010000100
Text  prediction:  I |   |a | a  | m |  f  |   i  |  n  | e  | e
Final prediction: "I aamfinee"
```
There are 9 1s in the cut prediction, so the strokes are divided by the cut prediction into 10 segments. This also means that there will be 10 chars in the predicted text. 

The next step is to determine which char will represent a given segment. We simply let the chars in a segment vote for their representative: The winner takes all.


Dependencies
------------
* Python==3.7   
* PyTorch==0.4   
* Numpy   
* Scipy
