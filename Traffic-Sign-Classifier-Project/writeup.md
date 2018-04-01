# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)
[image1]: ./images/label_dist.jpg "data distribution"
[image2]: ./images/new_dist.jpg "new data distribution"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./images/13.jpg "Traffic Sign 1"
[image5]: ./images/14.jpg "Traffic Sign 2"
[image6]: ./images/25.jpg "Traffic Sign 3"
[image7]: ./images/35.jpg "Traffic Sign 4"
[image8]: ./images/1.jpg "Traffic Sign 5"
[image9]: ./images/40.jpg "Traffic Sign 6"
[new_images]: ./images/new_images.jpg "Traffic New Signs"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

* Link to my [project code](https://github.com/zhouwubai/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb), much more details can be found in the notebook.
* Link to my [html file](https://github.com/zhouwubai/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.html)


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the training data distribution over different labels

![alt text][image1]

The first one is for the training dataset, the second validation and the third test. As we can see:
* The three data distribution are very similar to each other
* There exists data imbalance between different labels

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Why convert the images to grayscale?
* The performance of converting to grayscale is slightly better than doing nothing.
* It trains faster
* These signs are designed that color blind will be able to identify. So color might help little for identify those signs.
* Comparison between images and grayscaled ones can be found in notebook

Why normalize the image?
* The neural network learns its weight by stepping down the direction multiplied with a learning rate
* The gradient we computed are based on current point x_0. Theoretically, only extremly small learning rate will make the objective move the direction we expected. However, that will take long time to converge and might be very easily get stuck on a local minima
* If we use a relative large learning rate, we are assuming that current gradient will be similar for those points along the current direction (negative gradient) with certain distance so that we can move one "large step" along that direction.
* However, if we did not normalize the input, the ranges of our distributions of feature values would likely be different for each feature, and thus the learning rate would cause corrections in each dimension that would differ from one another. We might be over compensating a correction in one weight dimension while undercompensating in another. 
* We might end of oscillating state or in a slow moving without normalization.

Why generate additional data?
* More data set to prevent overfitting, better generalization
* Some augmented data might be potential real data in some conditions
    * rotated data for bad position of traffic sign or camara position
    * blurred picture for bad weather
* Techniques used to augment the data
    * Image Scale (smaller or bigger)
    * Image Translation (shifting)
    * Image Rotation
    * Image Affine Transformation
    * Brightness Change
    * Comparision can be find in notebook/html

The difference between the original data set and the augmented data set is the following:
* Totally number increases from 34799 to 51690
* Mainly generated images for the labels with less instances
* New data distribution for each labels as following


![alt text][image2]



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 gray image   							| 
| Convolution 5x5x1x6   | 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, kernal 2x2, outputs 14x14x4   	|
| Convolution 5x5x6x16  | 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| P2: Max pooling	   	| 2x2 stride, kernal 2x2, outputs 5x5x16    	|
| Convolution 5x5x16x400| 1x1 stride, valid padding, outputs 1x1x400 	|
| P3: RELU				|												|
| F1: Flatten(P2)   	| 400                   						|
| F2: Flatten(P3)   	| 400                   						|
| Concate F1, F2   		| 800                   						|
| Dropout       		| 800, keep_prob: 0.8              				|
| Fully connected		| 800x43        								|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used:
* optimizer: AdamOptimzer
    * Other optimizer (Adagrad, Adadelta, Momentum) was used, AdamOptimzer outperforms them in the perspective of both accuracy and convergence speed.
* batch size: 128
* number of epochs: 30
    * Running on local laptop, have not tried too big numbers. But usually for AdamOptimzer, it converges around the 20th epochs
* learning_rate: 0.001
* dropout probability: 0.8

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.9%
* validation set accuracy of 95.5% 
* test set accuracy of 92.6%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
    * LeNet was choosen first
    
* What were some problems with the initial architecture?
    * Accuracy is not good enough, training is slow on my laptop.
    * Tends to overfitting, discrepency between validation error and test error is large (more than 5%)
    
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
    * Add one more convolution layer referred in [Lecun Yann's paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf)
        * Going deeper might help to improve accuracy by extracting higher level features
    * Remove two fully connected layers
        * As we know, fully connected layer produce a lot of parameters which might lead to overfitting

* Which parameters were tuned? How were they adjusted and why?
    * learning rate
    * batch size
    * epoch number
    * dropout keep probablity
    
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
    * Convolution layer work well with image because it helps to extract higher level features from pixels such as edges
    * dropout leads to robust model or prevent overfitting

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![new_images][new_images]


They are labels are:
* 13,Yield
* 14,Stop
* 25,Road work
* 35,Ahead only
* 40,Roundabout mandatory
* 1,Speed limit (30km/h)


The first image might be difficult to classify because they are either too dark or too strong light/reflection

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).
13,Yield
14,Stop
25,Road work
35,Ahead only
40,Roundabout mandatory
1,Speed limit (30km/h)
Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Yield         		| Yield     									| 
| Stop      			| Stop  										|
| Road work				| Road work 									|
| Ahead only	      	| Ahead only 					 				|
| Roundabout mandatory	| Roundabout mandatory      					|
| Speed limit (30km/h) 	| Speed limit (30km/h) 							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 92.2%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 99.85%         		| Yield     									| 
| 100%      			| Stop  										|
| 99.97%				| Road work 									|
| 99.99%    	      	| Ahead only 					 				|
| 99.88%            	| Roundabout mandatory      					|
| 99.99%            	| Speed limit (30km/h) 							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Edge/Line/Boundry
