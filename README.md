## This repo contains various deep learning model implementations

# Lambdas/

Contains code that host trained models on AWS lambda. The lambdas are exposed via API gateway to enable classification trough APIs

## [Lambdas/Lenet5Lambda](https://github.com/cnv1989/DeepLearning/tree/master/Lambdas/Lenet5Lambda)

Lenet5Lambda is implemented using Serverless Framework.

# Lenet5/

### [Notebook](https://github.com/cnv1989/DeepLearning/blob/master/Lenet5/Lenet5.ipynb)

Tensorflow implementation of Lenet5 model for handwritten digit recognition.

# Resnet/

### [Notebook](https://github.com/cnv1989/DeepLearning/blob/master/Resnet/ResNet.ipynb)
### [Training Script](https://github.com/cnv1989/DeepLearning/blob/master/Resnet/resnet50_train.py)

Implented modified Resnet50 (by pruning the last residual block) for classifying CIFAR-10 dataset since resnet50 is built for 64x64 images while cifar-10 as has 32x32 images.

# YOLO/

### [Notebook - Convert YAD2K model to Keras](https://github.com/cnv1989/DeepLearning/blob/master/YOLO/YOLO.ipynb)
### [Notebook - Evaluate YOLO](https://github.com/cnv1989/DeepLearning/blob/master/YOLO/Evaluate.ipynb)

Takes YAD2K configuration and weights and converts to Keras. Applies box filtering and non max supression on the output of the model. 

![SF Marathon](https://github.com/cnv1989/DeepLearning/blob/master/YOLO/out/sfmarathon.jpg)
![Street 1](https://github.com/cnv1989/DeepLearning/blob/master/YOLO/out/testAWS.jpg)
![Street 2](https://github.com/cnv1989/DeepLearning/blob/master/YOLO/out/testaws2.jpg)
![Street 3](https://github.com/cnv1989/DeepLearning/blob/master/YOLO/out/test.jpg)
