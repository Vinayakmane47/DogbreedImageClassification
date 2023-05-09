# Dog Breed ImageClassification

## Overview : 
This project is aimed at building an image classification model that can classify different breeds of dogs based on their images. The project uses  CNN architectures  to train a model that can accurately identify the breed of a dog from a given image. Trained 5 different architecture on this datasset - **CNN Custom Model** ,**ResNet50** , **MobileNET** ,**DenseNet201** and **InceptionV3**. Got Highest **Accuracy** of **99 %**  , **F1-Score 99 %** , **Precision and Recalll of 99%** With **DenseNET201**.

## Dataset : 

The dataset used in  this project is taken from  [Kaggle Dogbreed dataset](https://www.kaggle.com/datasets/yapwh1208/dogs-breed-dataset?datasetId=3015645&sortBy=dateRun&tab=profile). Dataset have 5 classes with 1030 images in total. I have used data augmentation for increasing size of data.
Dataset have following classes : 
1. French Bulldog
2. German Shephard
3. Golden Retriever
4. Poodle
5. Yorkshire Terrier

random  images from  dataset : 
![image](https://user-images.githubusercontent.com/103372852/236993252-f3d828c1-3eb8-45d4-a9a1-05dbb34bad7a.png)

## CNN Models : 

### 1. CNN Custom  Model : 
Created a Custom CNN Model with 3 Convolution layers and 2 Dense Layer. Filter size of **3x3** and **Maxpooling of 2x2** is used.  Augmented data using keras ImageGenerator Function and created new images with rotation , vertical and horizontal shift , shear and zoom image.Initial Learning rate is set to **0.001** and gradually **warmed up** by batches of 5, keras callback function is used for warming up the learning rate .**Early Stopping** is also used. From the custom CNN model i got **Accuracy - 43.63 %** , **Precision - 43.15 %** , **Recall -43.63%** , **f1-Score -  42.31%**. 


### 2. ResNET50 :
Finetuned **Resnet50** with pretrained **ImageNET** weights . Used **ADAM Optimizer** and train through **10 Epochs** . Last Layer of **ResNet50** is freezed because our dataset have different classes. **128** neurons used in **Dense Layer** This Models yielded results of **Accuracy - 24.02 %** , **Precision - 24.02%** , **Recall -100 %** , **f1-Score -  38.74 %**. 

### 3. MobileNET : 
Finetuned **MobileNET** with pretrained **ImageNET** weights . Used **Adam Optimizer** and trained through **10 Epochs** . Last Layer of **MobileNET** is freezed because our dataset have different classes. **128** Neurons used in **Dense Layer** . This Model Yielded results of **Accuracy 98.04 %** , **f1-Score - 98.04 %** , **Precision 98.12%** , **Recall 98.04%** . 

### 4. DenseNet201 : 
Finedtuned **DenseNet201** with pretrained **ImageNET** weights . Used **Adam Optimizer** and trained through **10 Epochs** .Last Layer of **DenseNET201** is freezed because our dataset have different classes. **128** Neurons used in **Dense Layer** with **relu activation** . This Model yielded results of **Accuracy - 99.51%** , **f1-Score - 99.51% , **Precision - 99.52%** , **Recall - 99.51%**. 

### 5. InceptionNetV3 : 
Finedtuned **InceptionNetV3** with pretrained **ImageNET** weights . Used **Adam Optimizer** and trained through **10 Epochs** .Last Layer of **InceptionNetV3** is freezed because our dataset have different classes. **128** Neurons used in **Dense Layer** with **relu activation** .This Model yielded results of **Accuracy - 98.53%** , **f1-Score - 98.53%** , **Precision - 98.59%** , **Recall - 98.53%**. 

### Plots of DenseNet201 : 

![image](https://user-images.githubusercontent.com/103372852/237001894-deee054c-f503-4edc-bde3-66e531ab737a.png)
![image](https://user-images.githubusercontent.com/103372852/237001974-648e3696-822b-498b-84aa-2388feb4307a.png)
![image](https://user-images.githubusercontent.com/103372852/237001998-67700c49-bbd2-4a06-91f5-eaabfe406bd5.png)

## Conclusion : 

The Best results obtained on **DenseNet201** with **Accuracy - 99.51%** , **f1-Score - 99.51% , **Precision - 99.52%** , **Recall - 99.51%**.
Following table shows the comparision of results : 

| Model | Accuracy | F1-Score | Precision | Recall | train loss | val loss |
|----------|----------|----------|----------|----------|----------|----------|
| Custom CNN  | 43.63 %  | 42.31 % | 43.15 % | 43.63 % | 1.1663 | 1.3922 |
| ResNET50 |24.02 % | 38.74 % | 24.02 % | 100 %  | 1.6057 | 1.6068 |
| MobileNET | 98.04 % | 98.04 % | 98.12 % | 98.04 % | 0.0375 | 0.0423 |
| DenseNET201 | 99.51 %  | 99.51 % | 99.52 % | 98.53 % | 0.0211 | 0.0377 |
| InceptionNetV3 | 98.53 % | 98.53 % | 98.59 % | 98.53 % | 0.0543 | 0.03097 |


## How to  Models locally : 

- clone this repo - `$ git clone https://github.com/Vinayakmane47/DogbreedImageClassification.git` 
- create a environment - `$ git clone conda create -n env python=3.10 -y` 
- install requirements - `$ pip install -r requirements.txt` 
- run custom cnn model - `$ python custom_cnn.py` 
- fine tune resnet50 model - `$ python resnet50.py` 
- fine tune mobilenet model - `$ python mobilenet.py` 
- fine tune densenet201 model - `$ python densenet.py` 
- fine tune inceptionnetv3 model - `$ python inceptionnet.py`


## Make Predictions : 

- run predict.py - `$ python predict.py` 


## Application Link : 
[Dog_Breed_Classifier](https://huggingface.co/spaces/VinayakMane47/Dog_Breed_Classifier)








 



