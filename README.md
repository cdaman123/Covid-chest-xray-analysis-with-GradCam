# Covid Cheast Xray Analysis with GradCam.
### Introduction
In this project I train a resnet18 models for classify chest Xray between 4 class `Normal`, `Covid19`, `Lung_Opacity` and `Viral Pheumonia`. There are different methode to identify the **Covid-19** like RT-PCR. But these are more complex, time consuming and need special equipment for testing. I try to use DeepLearning tools to predict the Covid-19 Infection using Chest X-rays Images. Also I Implement `GradCam` Algoritms for visualize prediction.

### GradCam

`GradCam` or `Gradient-weighted Class Activation Mapping` is a technique for producing "visual explanations" for decisions from a large class of CNN-based models, making them more transparent. It uses the gradients of any target concept, flowing into the final convolutional layer to produce a coarse localization map highlighting important regions in the image for predicting the concept.

![image](https://user-images.githubusercontent.com/47690957/123196869-4caebd80-d4c8-11eb-91ee-0a5664eef497.png)

### DataSet
The [DataSet](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database) I use for this project contain `3616 Covid-19`, `6012 Lung_Opacity`, `10200 Normal`, and `1345 Viral Pheumonia` Chest XRays. For Handling unbalanced dataset I use `WeightedRandomSampler`.

### For download Dataset:
```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1humrlHzGN8vKByqAvmfgkeMVqewZ993_' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1humrlHzGN8vKByqAvmfgkeMVqewZ993_" -O data.zip && rm -rf /tmp/cookies.txt
```
### Confusion Metrix and Classification Report:
![image](https://user-images.githubusercontent.com/47690957/123194866-d066ab00-d4c4-11eb-9516-8b3aa7aff383.png)

![image](https://user-images.githubusercontent.com/47690957/123194280-da3bde80-d4c3-11eb-835e-56e5f8175b2a.png)

![image](https://user-images.githubusercontent.com/47690957/123194356-f8a1da00-d4c3-11eb-881d-c18f5c99f3ae.png)

> Model get 94% Accuracy overall on TestSet.

### Result
1. Covid

> ![image](https://user-images.githubusercontent.com/47690957/123197113-b0d18180-d4c8-11eb-8a09-19ab70ba3800.png)

2. Lung_Opacity

> ![image](https://user-images.githubusercontent.com/47690957/123197170-c646ab80-d4c8-11eb-9931-7ffc8f1250f8.png)

3. Normal

> ![image](https://user-images.githubusercontent.com/47690957/123197296-f2fac300-d4c8-11eb-9d2e-87c223e32feb.png)

4. Viral Pheumonia

> ![image](https://user-images.githubusercontent.com/47690957/123197339-0312a280-d4c9-11eb-99f3-1f5b74f7589c.png)


