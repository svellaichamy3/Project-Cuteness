# Visual explanation of features that influence popularity of pet images

## Description
In this project, we explored the use of existing pre-trained deep neural network architectures (ResNet, VGG, DenseNet,and  EfficientNet) for  transfer learningand  build  custom layers on top to include the metadata features in determining the popularity of pets (cats & dogs). We further use three different  saliency  maptechniques:  Vanilla  Gradient, GradCam  and  RISE, to  visually  explain  the  important features  of  the  pet’s  image. We  came  up  with  several hypotheses to interpret popularity among different images and using saliency maps, we presented evidence to confirm or reject the hypothesis. 

## Data
We used the ‘PetFinder.my - Pawpularity Contest’ dataset from Kaggle (https://www.kaggle.com/c/petfinder-pawpularity-score) The data consists of three major items :<br />
* Pet Images of cats and dogs
* Metadata of images includes binary features describing the subject, like focus, eyes of the pet facing front, pet having a clear face, single pet, action at the time of snap, presence of accessory, group photo, collage, presence of human, occlusion, and blur.
* Pawpularity (popularity) score for each image on a scale of 0-100.

## Approach
In this work, we attempted to predict a pet’s popularity score based on an image of the pet and some metadata features of this image, using several convolutional neural network (CNN) architectures known to work well on image classification. We used these neural networks to: 
1. Perform regression on the images and metadata to predict the popularity score, and
2. Generate saliency maps to highlight which parts of the images appear to be most significant resulting in high or low popularity scores
