Assignment - Final - Task 4 - Classification
=================================

Please refer to ReadMe.txt for general rules for this final assignment.

In this task, you shall train and compare classification models for various versions of data.
For each of these you shall use the provided code to generate your train and test data.
Keep the images small - 32 pixels - with a single object of one of the four categories with max size 24 pixels.
You can generate as many examples for the training set as you wish (10000 may be a good number).
You shall always generate 1000 examples for the test set.

Part 1:
* Generate train a test data in the original colors (with --task4 commented out in config.ini).
* Train and test a classification network over these images to classify which type of sportsball there is in the image.
* The architecture can be fairly simple, do not go for too complicated models.

Part 2: 
* Generate train a test data with random colormap (with --task4 active in config.ini).
* Use the network trained in Task 1 to classify the test mixed-color data - do not do any finetuning, use it as it is.
* Use 100 mixed-color train images to finetune (re-train) the network and test it again.
* Compare and discuss the results. 
* Train and test a classification network over these mixed-color images to classify the type of sportsball.
* Use the network trained here to classify the test data in the original colors (as in Part1) - do not do any finetuning, use it as it is.
* Use 100 original color train images to finetune (re-train) the network and test it again.
* Compare and discuss the results also in comparison to the previous case.

Part 3:
* Tweak the code in `data/sportballs.py` to label each image not only by the type of sportsball but also the colormap type (original, inverted, grayscale).
* Generate train a test data with random colormap (with --task4 active in config.ini) and the two types of labels - type of sportsball and type of colormap.
* Train a model which can predict both of these labels.
* Discuss the results.

You shall follow the instructions for the content of the report as explained in ReadMe.txt. 
In addition, you shall:
* Discuss the results as indicated above in Parts 1-3.
* Compare and discuss the performance of all models developed in Parts 1-3 for sportsballs classification.



