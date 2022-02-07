# Generate dataset

This file explains how do we get our train- and testdataset. Our data source comes from 
https://rohitgirdhar.github.io/CATER/
![image](https://user-images.githubusercontent.com/92917022/152701308-8d16f750-3541-408c-b1c0-ffe085e8d483.png)

We collected data from All_actions_cameramotion 'CATER_NEW_005200.avi'-'CATER_NEW_005499.avi' with the fixed camera position. The 0,50,100,150,200,250,300 frame were slected from videos and analysed. There are several objects with different shapes, colors, sizes, materials and 3D-Coordination in each picture.
For each object we got its segmentation and Bbox position to locate it in the picture. The data were saved as Coco 1.0 format

## Presegementation with opencv
### GrabCut: an Interactive Foreground Extraction method
![image](https://user-images.githubusercontent.com/92917022/151390965-a0cd762b-5e26-4570-9509-4487b2f3ab4b.png)
### Interactive
Manually selecting region to separate the object boundary from background.
![image](https://user-images.githubusercontent.com/92917022/152701344-eb9ee1d5-0087-4152-b29a-5511f64bc0cd.png)

## Simple Classifier for attributes color, size and material.
Count information of 50 images, about 300 objects. Use simple model from scikit-learn for classification task.The accuracy of the predictions is about 90%. Save time for annotation 
However, The result of the shape prediction by using simple ML model is poor.
![04_color_prediction](https://user-images.githubusercontent.com/92917022/152701417-8c62383e-32e1-457f-ba99-257fe2509cc6.jpg)
### Manually input Shape attribute
![image](https://user-images.githubusercontent.com/92917022/152701448-d7c72a4c-780e-433a-b60d-f21a7b402e0f.png)

## Simple Classifier for attributes color, size and material.
Pretrain a mask rcnn model using the first 1000 labeled images. Use that model for pre-segmentation and attributes prediction of objects in the second 1000 images.Better segmentation and shorter runtime with Mask rcnn model

![image](https://user-images.githubusercontent.com/92917022/152701470-a1225007-7d6b-4371-aa4a-0b8b9dfe206a.png)
![image](https://user-images.githubusercontent.com/92917022/152701473-11b85931-0516-421e-92c0-0789c1d4cad6.png)

### Check Attributes and Assign Coordinations
Cross check the initial predicted attribute with original data from CATER video. Assign coordination data to the corresponding object.
![image](https://user-images.githubusercontent.com/92917022/152701597-550a2318-c133-4d52-b4d8-0c6e11d1ea39.png)

## Fix wrong and inappropriate segmentation on https://cvat.org
### Upload predicted raw annotation
![image](https://user-images.githubusercontent.com/92917022/151435801-a3b47039-5fef-4828-ae88-9433032a2845.png)
### smooth the boundary of object and change the wrong property
![image](https://user-images.githubusercontent.com/92917022/151436124-d82549b9-d455-4f9e-87a0-4dda48401764.png)
### Save file as Coco 1.0 format
![image](https://user-images.githubusercontent.com/92917022/151436284-fab31622-577d-4b02-9083-4dbc91649394.png)

## Result
### label all objects: 193 kinds of labels in total
![image](https://user-images.githubusercontent.com/92917022/151436597-1267fc7d-7e1c-455a-9a9a-521d0c43cb10.png)
### merge all Coco file: 2044 images, 13468 objects in total
![image](https://user-images.githubusercontent.com/92917022/151437270-cb771b66-87ab-4d4c-b329-951dc13e1e51.png)
