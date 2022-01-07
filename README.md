# Visual explanation of features that influence popularity of pet images

## Project Aim and Scope
In this project, we set out to challenge ourselves in exploring the applications of Deep Learning to emulate human’s perception or emotional stimuli towards subjects captured in images, specifically of pets by detecting a “popularity” score, a metric derived to measure a pet’s cuteness or attractiveness and was termed pawpularity. By understanding what features make an image “popular,” we can leverage those features to take photos that highlight the cuteness of the pet. This can be useful for pet adoption services wherein most people shortlist pets that they want to consider adopting prior to visiting the shelter based on how these pets look in profile images.
<p align="center">
<img src="https://github.com/svellaichamy3/Project-Cuteness/blob/main/Images/Paw_not_paw.PNG" width="400" height="200" />
</p>
  
## Approach
In this project, we explored the use of existing pre-trained deep neural network architectures (ResNet, VGG, DenseNet,and  EfficientNet) for  transfer learning and  build  custom layers on top to include the metadata features in determining the popularity of pets (cats & dogs). We further use three different  saliency  maptechniques:  Vanilla  Gradient, GradCam  and  RISE, to  visually  explain  the  important features  of  the  pet’s  image. We  came  up  with  several hypotheses to interpret popularity among different images and using saliency maps, we presented evidence to confirm or reject the hypothesis. 

## Data
We used the ‘PetFinder.my - Pawpularity Contest’ dataset from Kaggle (https://www.kaggle.com/c/petfinder-pawpularity-score) The data consists of three major items :<br />
* Pet Images of cats and dogs
* Metadata of images includes binary features describing the subject, like focus, eyes of the pet facing front, pet having a clear face, single pet, action at the time of snap, presence of accessory, group photo, collage, presence of human, occlusion, and blur.
* Pawpularity (popularity) score for each image on a scale of 0-100.

## Data Augmentation
Our training data had a highly imbalanced distribution of scores, with the majority of the images having scores within the 20-40 range. This proved to be a challenge to our models as most of the predictions will tend towards this range as well. A practical alternative is to consider oversampling by adding image transformations to augment our training data. After resizing the images such that they are all 224x224 pixels, we applied any one of the following transformations to images with scores within the 0-20 and 50-60 ranges: (1) Horizontal Flip, (2) Center Crop (to 90% of image size then resized to 224x224), or (3) Gaussian Blur (σ = 0.3)
<p align="center">
<img src = "https://github.com/svellaichamy3/Project-Cuteness/blob/main/Images/Augmentation.PNG"/>
 </p>

## Architecture
We leveraged select famous CNN architectures pretrained on the ImageNet database and used transfer learning to help us save time on training, tuning which parameter layers/blocks to freeze and which to further train on our data. However, these models were trained for image classification tasks, not regression, and as such we modified the final layer of these models and concatenated these with feature embeddings from our metadata generated from feed-forward linear layers. We passed this concatenated set of feature embeddings to a few more feed-forward layers then to a final layer with a single neuron that predicts the popularity score. Being a regression model, we used MSE Loss as our metric for assessing quality of performance. A generalized diagram of our neural network architecture is shown in the figure. 
<p align="center">
<img src = "https://github.com/svellaichamy3/Project-Cuteness/blob/main/Images/Architecture.PNG"/>
  </p>
  
 ## Neural Network Visualisation
 Since we are using a neural network to imitate the behaviour of perceived popularity, it is important to understand what aspects of the image are looked at while deciding the popularity score. Additionally, our bestperforming model does comparably well to a human and it would help build trust in the system to bring insights missed by humans or to validate initial perceptions. Visualisation is a strong tool to aid us in understanding the system as well as the data. We specifically focus on saliency maps in this project.
Saliency maps are of two types:
* **Gradient based:** Gradients calculated on the pixel values of the image by back propagating the loss holding the model constant gives us information on how the score would
change on varying the pixel slightly. This measures the sensitivity of the pixel to the final decision. Higher the sensitivity, the more crucial the pixel has been for the
prediction. It should be noted that this works only in the case of whitebox models where we know the architecture and hence have access to gradients. We use simple gradients and GradCAM.

* **Perturbation based:** In this method, we block a few pixels or patches of pixels to see how much the final prediction changes. Based on the change in score, the importance score is calculated for pixel or block of pixels. This works with black box models where we don't have access to model weights and architecture because all we need is the output for a given input. We use a method called RISE (Randomized Input Sampling for Explanation of Black-box Models) in this category. 


<p align="center">
 <img src = "https://github.com/svellaichamy3/Project-Cuteness/blob/main/Images/sample_outputs.PNG"/>
  </p>
  
## Conclusion

Given the understanding of some elements that contribute to the popularity of a pet, highlighting these features for new pet images could lead to high prediction scores on the platform which may ultimately result in faster and more successful adoptions. Below are some guidelines (not exhaustive) for capturing a pet image that better captivates human attention.
1. Maintain just one subject in the image
2. Ensure a contrasting background against the pet
3. Make the face clearly visible along with both eyes
4. Avoid unusual poses of the pet
5. Bonus points if the dog sticks out its tongue indicating health and playful energy.



