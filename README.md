# Age-and-Gender-Estimation

To train a Tensorflow VGG Network on the Adience Age and Gender dataset.

The implementation contains the following aspects:

1. Extracting the pre-trained VGG weights from the MatConvNet file
2. Reconstructing the VGG Model usig Tensorflow
3. Training it on the Adience Dataset
```
Tensorflow 1.15
Ubuntu 18.04
```
```
-The weights extraction is done using the scripts provided in Data/vgg_data_preprocessing. [2]
	There are multiple ways to store data such as h5, npz etc. But I felt that storing it as a pkl file would be ideal for my usage
```
```
-The Reconstruction of the VGG network can be found in vgg.py
```
```
-Training on the Adience dataset first required quite a bit of data preprocessing [3]
	-The preprocessing can be found in the Data/adience_datapreprocessing (adience_data_preprocessing.ipynb)
	-An h5 file was created that stores the image dataset information.
```

```
-Training can be started by running the "run_train.sh" It contains all the training parameters and all argument changes can be directly done in this file. It calls the "train_age.py"
```


The code works well without a GPU as well. 

## Instructions to run file:
-Change Directories in "run_train.sh" file and "vgg.py". Since I have already provided the weights, no pre-processing is required

```
References

[1] http://www.robots.ox.ac.uk/~vgg/software/vgg_face/
[2] https://sefiks.com/2019/07/15/how-to-convert-matlab-models-to-keras/
[3] https://github.com/zonetrooper32/AgeEstimateAdience
```



