Convolutional Neural Network to Predict Ethnicity

Objective: The objective of this model is to predict the ethnicity from the facial images and to experiment  with various data balancing techniques to determine the best performing model.
Data Preparation: The distribution for ethnicity data was  imbalanced, as it has more records for the class“White American” as compared to all other classes. So we try to balance the data first so that we can prevent our model from predicting biased results or overlook the minority classes.
Train-Test Data: Entire dataset was divided into 80-20 ratio for Training-Test data.
Balancing the Train data: We make sure to balance only the Training data and not to balance the Test data , to better evaluate the performance of our model on the unseen dataset later on. 
We try to perform some experiments with the model just to see which balancing technique works best for our model and used the below mentioned technique to balance the data:
❏	Adasyn: Add synthetic samples for minority classes with some variation on the generated samples and thereby results in an unequal number of records for each class[7].
❏	SMOTE: Add synthetic samples for minority classes with no variation in the samples and thereby results in an equal number of records for each class[7].
❏	Up-Down sampling: Down sample the minority class and upsample the majority class by taking the mean number of records .
❏	Focal_loss: It is a self defined loss function which unlike cross entropy(try to increase the confidence in predicting easily classified examples) try to focus more on the hard to classify examples , it gives priority or more weightage to  minority classes while calculating the loss[8]. 


CNN Model: Model was developed using Keras sequential Api in Google Colab Notebook. Model predicts the ethnicity from the facial images for the given dataset. The  Code snippet  for the model is shown below in figure2. 
 
CNN architecture accepts input of shape 48X48X1, followed by 3 Convolution 2D layers with kernel size 3X3 and number of filters as 32, 64 and 128. Convolution 2D layers are followed by Batch Normalization layers, Activation Layer with “Relu ” activation and  MaxPooling2D of 2X2 . Flatten layer follows CCN architecture to convert the data to 1D vector before it is fed into Dense layers. There are 3 Dense layers with 1024 and 512 to create bottleneck architecture with dropout layers included to reduce overfitting. The last/output layer has 5 neurons with softmax as the activation function. The model represented in Figure2 is compiled with the learning rate of 0.0001 and trained for 500 epochs with the batch size of 32.


Transfer Learning PreTrained Model (VGG-16): As per above experiments, we came to the conclusion that CNN with Smote Balanced data performs better as compared to others,  with the accuracy of 79 percent on our data. 
To check  the performance of a smote balanced data  using some pretrained networks, we decide to train our balanced data on  VGG_16 pretrained  network by freezing all the conv layers and adding our Fully connected network layer to it with the same configuration as mentioned in figure 2. When we ran the model with above mentioned configurations,  unfortunately we weren't able to get an accuracy of more than 60 percent[4], and the reason is we were providing an input shape of 48*X 48 to the network, whose weights were already trained on the images of shape 224X224. So we decide to unfreeze all the conv 2d layers of the VGG network and try to retrain the model weights.  The final results lead to increased  performance with the test accuracy of 82 percent.
 


Testing results from VGG-16 smote balanced pretrained model: we were able to increase the performance of our model to 3 percent using VGG-16 with the test accuracy of 82 percent.
 

 

Conclusion: 
❏	Transfer learning with Smote balanced data using  VGG-16 works best.
❏	SMOTE balanced data perform better as compared to other techniques in our CNN model.
❏	There is a limit on how much we can fine tune the input shape during transfer learning both from accuracy and loss perspective.
❏	Ethnicity had an accuracy of 82%. Out of all the ethnicities, Whites had the highest accuracy and Others had the lowest accuracy. Again, the reason being, even humans when shown the image are not confident enough in predicting the ethnicity for the class label “other” and so does the model.
