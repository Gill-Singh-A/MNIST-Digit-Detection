# MNIST Digits using PyTorch
A Python Program that uses PyTorch to train a Neural Network for identifying Handwritten Digits (0-9) displayed on 28x28 GrayScale Image

## Requirements
Language Used = Python3<br />
Modules/Packages used:
* torch
* torchvision
* pickle
* pathlib
* datetime
* optparse
* matplotlib
* time
* colorama
* numpy
* pygame
<!-- -->
Install the dependencies:
```bash
pip install -r requirements.txt
```

## Neural Network
Input Layer Dimensions = **28x28**<br />
Input Layer Depth = **1**<br />
Inner Layers:
1. Convolutional Neural Network
    * Input Depth  = 1
    * Output Depth = 32
    * Kernel Size  = 3
    * Activation Function = Rectified Linear Function (ReLu)
2. Convolutional Neural Network
    * Input Depth  = 32
    * Output Depth = 64
    * Kernel Size  = 3
    * Activation Function = Rectified Linear Function (ReLu)
3. Max Pool
4. Flatten
5. Fully Connected
6. Softmax
<!-- -->
Algorithm for Minimizing Loss = **Scholastic Gradient Descent**

## Inputs
* '-d', "--device" : Device to use for training the Neural Network (cpu/gpu)
* '-s', "--save" : Name for the Model File to be saved (Default=Current Date and Time) in 'models' Folder
* '-l', "--load" : Load an existing Model File (stored in 'models' folder)
* '-b', "--batch" : Batch Size for the Training
* '-r', "--learning-rate" : Learning Rate for Loss Function
* '-m', "--momentum" : Momentum for Loss Function
* '-e', "--epoches" : Number of Epoches for Training

## Outputs
A Graph of Accuracy & Loss vs Epochs.<br />
Then a Screen with 28x28 boxes in which you can play, by drawing numbers and predicting them using the Network.<br />
The Screen has these controls:
* Press D: Sets Mode to Drawing (Click on any box, it would become White)
* Press E: Sets Mode to Erasing (Click on any box, it would become Black)
* Press C: Clears the Whole Screen (All Boxes Black)
* Press =: Runs the Whole Network and displays the Predicted Number