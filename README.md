# DogbreedImageClassification

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
Created a Custom CNN Model with 3 Convolution layers and 2 Dense Layer. Augmented data using keras ImageGenerator Function and created new images with rotation , vertical and horizontal shift , shear and zoom image.Initial Learning rate is set to **0.001** and gradually **warmed up** by batches of 5, keras callback function is used for warming up the learning rate .**Early Stopping** is also used. From the custom CNN model i got **Accuracy - 43.63** , **Precision - 0.4315** , **Recall - 0.4363** , **f1-Score -  0.4231**. 

 



