# Public-Grievance-Image-Classification Using Keras

 There are plently of public grievances which are saved by the public during travel or in other times. The faster they are redressed by
 the authorities , the better will be the life for the public.So, this model for classifying three classes of images - Overflowing garbage,
 open manholes and patchy roads.
 
 In this project , I've explored data augmentation and what effects it has on the model performance. The observations will be made know
 in the "Observation" section below.

## Description of Folders in the repo
* Baseline model - Contains the gr_base.py file for running a model without any data augmentation. Also has plots for Accuracy and Loss for
the training and validation sets plotted as training progressed. Contains results.csv where the predictions on the test set are recorded.

* On the Fly Augmentation - Contains gr_3_aug.py for running a model withdata augmentation.Also has plots for Accuracy and Loss for
the training and validation sets plotted as training progressed. Contains results.csv where the predictions on the test set are recorded.

* Train - Contains three folders. One for each class of images

* Test - Contains the test images.

## Prerequisites

* [Python](https://www.python.org/)- The programming language used in this project.

* [Keras](https://keras.io/) - Python deep learning API for rapid prototyping

* [OpenCV](https://opencv.org/) - A open source computer vision and machine learning software library. Used in this project for loading 
resizing images

* [Pandas](https://pandas.pydata.org/)-open source data analysis and manipulation tool,built on top of the Python programming language

## Getting Started 
  
  To run any of the given two models , use the code snippet below
```
python gr_base.py #For running the baseline model
```
## Observation 
 * Base line model:
      The train and validation accuracy diverged during the course of training. Meaning that the model does not generalise well.
      
<img src="Baseline%20model/Acc%20plot.png">

   The validation loss was also considerably higher than the train loss. This also conveys the fact that the model does not generalise well.

<img src="Baseline%20model/loss%20plot.png">

 * Model with Data Augmentation:
   The perturbations on the train images added are:
    
    * Width Shift
    * Height Shift
    * Horizontal flip
    * Shearing
    
   The validation loss was also considerably closer than the train loss. This also conveys the fact that the model does generalise well.

<img src="On%20the%20Fly%20Augmentation/loss%20plot.png">

## Conclusion

 It's wise to use data augmentation more than not , in order to generalise the model , eventhough a slight dip in training accuracy is 
 inccured.
 
### NOTE:

  The dataset is a small one with just 257 train images (3 classes) and 71 test images (3 classes). This was because , the images on the 
  internet for Indian public grievances was only limited in comparison to the vast amount of public grievance images of foreign countries.
  I humbly request any developer or person with social concern add to the dataset in order to effective automated Grievance Redressal 
  system.
      
## Acknowledgments

* This was a part of an application submitted for the Sastra Daksh Smart City hackathon at Feb 2020.
* This won the first place in the hackathon.
