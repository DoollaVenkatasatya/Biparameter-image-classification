# Image Classification using CNN from tensorflow

# In the given repository there are three classifiers in respective jupyter notebooks namely:

## 1.Binary classification - classifying among two classes of images
## 2.Multiclass classification - classifying among multiple classes with an preprocessed dataset (cifar10)
## 3.multiple image classification - Classifying among multiple classes with a raw dataset from scratch (used weather dataset from kaggle)

## Procedure:

![image classification](https://github.com/DoollaVenkatasatya/Biparameter-image-classification/assets/137089784/ebabf699-d3af-4ad8-8b55-f1c55a8650eb)

The provided code is an image classification pipeline using a Convolutional Neural Network (CNN) to classify different weather conditions based on input images. The code includes various steps, such as data loading, data preprocessing, data augmentation, model creation, training, and evaluation.

Here's a step-by-step explanation of the code:

1. **Importing Dependencies**: The necessary libraries such as pandas, numpy, cv2 (OpenCV), Matplotlib, Seaborn, and Keras (from TensorFlow) are imported for data manipulation, visualization, and building the CNN model.

2. **Removing Dodgy Images**: The code goes through the given base directory and checks each image for valid formats. If an image is not in the list of acceptable extensions (jpeg, jpg, bmp, png), it will be removed from the dataset.

3. **Loading Data**: The code loads the dataset by walking through the given data directory and storing the label and image path information into a Pandas DataFrame.

4. **Data Visualization**: The code plots a histogram to visualize the distribution of different classes in the dataset.

5. **Splitting Data into Train and Validation Sets**: The code creates a new base directory as mentioned in the code and splits the data into training and validation sets for each class. The split size is set to 80% for training and 20% for validation.

6. **Data Augmentation**: The training data is augmented using the ImageDataGenerator from Keras to increase the diversity of the training set and improve the model's generalization.

7. **Building the CNN Model**: The code defines a CNN model using the Sequential API from Keras. It consists of multiple Conv2D layers with ReLU activation, MaxPooling2D layers for downsampling, and Dense layers for classification.

8. **Model Compilation and Training**: The model is compiled with an Adam optimizer, categorical cross-entropy loss function, and accuracy as the metric. It is then trained on the augmented training data for 100 epochs, and the best model is saved using the ModelCheckpoint callback.

9. **Training and Validation Accuracy/Loss Plot**: The code plots the training and validation accuracy/loss curves to visualize the model's performance during training.

10. **Test Performance on Test Data**: The code defines a function 'preprocess_image' to preprocess a single image for prediction.

Overall, this code demonstrates a complete image classification pipeline using a CNN model for weather classification.
