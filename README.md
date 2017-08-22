
# Build a Traffic Sign Recognition Project

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images


[//]: # (Image References)

[image1]: ./examples/testing_data_set_hist.png "Visualization"
[image2]: ./examples/gray_scale_images.png "Grayscaling"
[image3]: ./examples/additional_data.png "additional data"
[image4]: ./examples/balanced_hist.png "balanced_hist"
[image5]: ./examples/web1.png "web1"
[image6]: ./examples/web2.png "web2"
[image7]: ./examples/web3.png "web3"
[image8]: ./examples/probs.png "probs"
[image9]: ./examples/confusion_mat.png "confusion matrix"

README

####1. Here is the ipython notebook for my project [project code](https://github.com/sai19872000/Traffic_sign_classifier)


###Data Set Summary & Exploration

####1. Initially I load the pickled data set that is given to us. I used the pickle library to load the data sets. The dataset contains traning file, validation file and the testing file. In the next section I used the numpy libraries shape command to get information about the number of training examples, shape of the image, etc.. 

* The size of training set is 344799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32,32
* The number of unique classes/labels in the data set is 43

####2. Histogram of training data set for different classes

This histogram shows the distribution of number of examples of each traffic sign in the training data set

![alt text][image1]

This histogram shows uneven distribution. Later I have used image processing to balance the histogram

###Design and Test a Model Architecture

####1. With the LeNet-5 architecture, I achived a validation set accuracy of about 0.9. I have tried several different things to imporve the model performence. Here I will explain the steps I have taken to improve the model validation accurary.

I have used a grid search of parameters for each of implementation

| Batch Size     		|     Learning rate	        	| epochs	| 
|:---------------------:|:---------------------------------------------:|:-------------------------------------| 
| 64     		|     0.01	        	| 20	|
| 128     		|     0.001	        	| 50	|
| 256     		|     0.0001	        	| 100	| 

hence for we have 3*3*3=27 cases for each implementation

I started with the lenet network architecture, RGB images with following configuration


| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| RELU					|												|
| Convolution 5x5	    |1x1 stride , outputs 10x10x16      									|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| RELU					|												|
| flatten					|												|
| Fully connected		| 400 inputs, 120 outputs        									|
| RELU		|        									|
| Fully connected		| 120 inputs, 84 outputs        									|
| RELU		|        									|
| Fully connected		| 84 inputs, 43 outputs        									|
|						|												|
|						|												|

For this architecture I have achived an accurary of 0.9


First I have tried normalizing the data set (pixel - 128.)/ 128, which helps the network to find a lower minima faster. 
This helped me improve the model accuracy to 0.92-0.93 (accuracy variation here because of 27 different cases of hyperparameters).

As a first step, I decided to convert the images to grayscale along with histogram localization because the grayscale and histrogram localization images can help the network to identify edges better. For this I have used the cv2 library
I have created a new function called RGB2GRAY for grayscaling and histogram localization. Image below shows a sample of some random images after conversion.

![alt text][image2]

I have normalized these gray scale images the same way (pixel - 128.)/ 128; and used the same lenet model architecture. The only difference here is that the input which is now 
32x32x1 Gray scale image insted of 32x32x3 RGB image. When I used this data for training the performance improved to 0.93-0.94 (accuracy variation here because of 27 different cases of hyperparameters).

The model at this stage seem to predicit some signs better than other this might be because of the imbalanced data set. Hence in the next step I have generate additional training data and therefore generate a balanced data. For generating additional data I have used a cv2 library to randomly rotate, shear,translate and adjust brightness of an image. The figure below shows the additional data generated for 50 speed limit sign

![alt text][image3]

After implementing this technique I generated a balanced data set and corresponding histogram is shown below

![alt text][image4]

For this data set the model validation accuracy varied between 0.94-0.96(accuracy variation here because of 27 different cases of hyperparameters).

As a next step I tried modifying the lenet architecture; I increased the feature maps, increased number of hidden nodes in the fully connected layers and used dropout for fully connected layers to make the model more robust

####2. The modified lenet architecture is shown below

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x24 	|
| Max pooling	      	| 2x2 stride,  outputs 14x14x24 				|
| RELU					|												|
| Convolution 5x5	    |1x1 stride , outputs 10x10x64      									|
| Max pooling	      	| 2x2 stride,  outputs 5x5x64 				|
| RELU					|												|
| flatten					|												|
| Fully connected		| 1600 inputs, 480 outputs        									|
| RELU		|        									|
| Fully connected		| 480 inputs, 336 outputs        									|
| RELU		|        									|
| Fully connected		| 336 inputs, 43 outputs        									|
|						|												|
|						|												|

For this model the accuray varied between 0.96-0.98(accuracy variation here because of 27 different cases of hyperparameters).

####3. Model training

I have used the adamoptimizer as it is has less parameter tuning compared to SDG (where I also have to optimize the decay and momentum). As described about I have used 3 different values of learning rate and 3 different values of epochs and 3 different values of batch size and performed a grid search and arrived at the best validation accuracy of 0.97-0.98
To train the model, I used an ....

####4. Approach

My approch from the begining was to use lenet/modified lenet as this is proven to be good at classifying mnist data. Each of the convolutional layers capture features such as edges, shapes etc. Traffic signs also have similar features as different shapes correspond to different classes. 

My final model results were:
* validation set accuracy of 97%
* test set accuracy of 94%

the confusion matrix is shown below

![alt text][image9]

the classification report is shown below

                                                     precision    recall  f1-score   support

                              Speed limit (20km/h)       0.74      0.90      0.81        60
                              Speed limit (30km/h)       0.94      0.97      0.95       720
                              Speed limit (50km/h)       0.95      0.97      0.96       750
                              Speed limit (60km/h)       0.92      0.96      0.94       450
                              Speed limit (70km/h)       0.95      0.94      0.94       660
                              Speed limit (80km/h)       0.90      0.89      0.90       630
                       End of speed limit (80km/h)       0.90      0.85      0.87       150
                             Speed limit (100km/h)       0.95      0.93      0.94       450
                             Speed limit (120km/h)       0.93      0.96      0.95       450
                                        No passing       0.98      0.99      0.99       480
      No passing for vehicles over 3.5 metric tons       0.99      0.96      0.97       660
             Right-of-way at the next intersection       0.96      0.95      0.96       420
                                     Priority road       0.98      0.97      0.98       690
                                             Yield       1.00      1.00      1.00       720
                                              Stop       0.95      1.00      0.97       270
                                       No vehicles       0.97      1.00      0.98       210
          Vehicles over 3.5 metric tons prohibited       0.97      1.00      0.98       150
                                          No entry       0.99      0.97      0.98       360
                                   General caution       0.96      0.76      0.85       390
                       Dangerous curve to the left       0.88      0.97      0.92        60
                      Dangerous curve to the right       0.69      0.88      0.77        90
                                      Double curve       0.84      0.90      0.87        90
                                        Bumpy road       0.85      0.93      0.89       120
                                     Slippery road       0.85      0.97      0.91       150
                         Road narrows on the right       0.85      0.98      0.91        90
                                         Road work       0.97      0.94      0.95       480
                                   Traffic signals       0.86      0.93      0.89       180
                                       Pedestrians       0.44      0.48      0.46        60
                                 Children crossing       0.92      0.97      0.94       150
                                 Bicycles crossing       0.89      0.94      0.91        90
                                Beware of ice/snow       0.84      0.67      0.75       150
                             Wild animals crossing       0.89      0.97      0.93       270
               End of all speed and passing limits       1.00      0.85      0.92        60
                                  Turn right ahead       0.97      0.99      0.98       210
                                   Turn left ahead       0.98      1.00      0.99       120
                                        Ahead only       1.00      0.98      0.99       390
                              Go straight or right       0.98      0.99      0.99       120
                               Go straight or left       0.90      0.92      0.91        60
                                        Keep right       1.00      0.93      0.96       690
                                         Keep left       0.88      0.97      0.92        90
                              Roundabout mandatory       0.82      0.90      0.86        90
                                 End of no passing       0.96      0.88      0.92        60
End of no passing by vehicles over 3.5 metric tons       0.99      0.94      0.97        90

                                       avg / total       0.95      0.94      0.94     12630



I have used the sklearn library to make confusion matrix and the classification report.

These results prove that the this model architecture does a great job in classifying traffic signs.
Since my model architecture is fixed I used grid search (of hyperparmeters) for each of the data processing step. For example I did a grid seach(hyper parameters) of RGB images then I converted the images to grayscale and performed the grid search again to check for model performance. I then created additional data to balance the training set and did the grid search again to check for improvement. Finally I changed the model architecture and performed the grid search (hyperparameters) and saw great improvement in the validation accurary. Adding the dropout layers (80%) imporved the test accuracy from 0.92 to 0.93 % also helped not to overfit the training dataset. The model architecture never over fitted or under fitted the data as the testing and validation accuries were always close together. The pooling layers in the network helps the model not to overfit the training data.

More careful image processing and network tweeking will result in validation accuracies upto 99%

###Test a Model on New Images

####1. I have choosen 5 images from wikipedia  which are shown below 


![alt text][image5]

The first image might be difficult to classify because the dimentions of the image are not 32,32. I used the cv2 library to rescale the image and convert to 32,32. After resizing the images look as

![alt text][image6]

I then transformed them to grayscale and then normalized them. The gray scale images look as follows

![alt text][image7]

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set 

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| caution sign      		| caution sign   									| 
| right turn ahead     			| right turn ahead 										|
| round about					| keep left										|
| keep right	      		| keep right					 				|
| stop sign			| stop sign      							|

from the confusion matrix all these have good classification accuracy but for some reason the round about sign is mis classified here. From the probabilites the round about is present in the top 5.  

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of

####3. Here are probabilities with which the network is able to classify the images.)

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

I have made bar plot where the y axis is the softmax probability and the x axis contains labels for the corresponding top 5 probabilities.

![alt text][image8]


