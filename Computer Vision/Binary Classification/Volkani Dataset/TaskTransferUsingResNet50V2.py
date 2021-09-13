from keras_preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
import math
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.metrics import average_precision_score
import tensorflow as tf
import keras
from keras import backend as K
from keras import regularizers
from keras.layers import Dropout, BatchNormalization
from keras.initializers import glorot_normal
import numpy as np
import scipy.io as sio
import cv2 as cv
import re
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.layers import Dense
from keras import Model
import sklearn as sk
from sklearn.metrics import average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve, f1_score
from sklearn.svm import SVC
import seaborn as sns
import itertools
import scikitplot as skplt
import matplotlib
import datetime as dt
# from keras.callbacks import TensorBoard
# log_dir = r"C:\logs\fit" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = TensorBoard(log_dir=log_dir)


def set_seeds(n):
    """
    set_seeds(...) - Sets seed for reproduced results

    :param n: Seed for Generation
    :return: Set seeds
    """

    # Setting random seed generator
    np.random.seed(n)
    random.seed(n)
    tf.random.set_seed(n)

def loadImages(data_path, test_images_indices, size):
    """
    loadImages(..) - Loads and Resize images by a specified size and then splits to train and test sets

    :param data_path:
    :param test_images_indices:
    :param size:
    :return: Data - a dictionary contains Train and Test sets
            - Train["Data"] - train set array of single 4D tensors of shape (N-samples X Height x Width x Channels)
            - Train["Labels"] - a single train Labels vector of size 1×N
            - Test["Data"] - test set array of single 4D tensors of shape (N-samples X Height x Width x Channels)
            - Test["Labels"] - a single test Labels vector of size 1×N
    """

    print("################ Loading Images  #################")

    # Initialize labels, image array and data dictionary
    labels = []
    images = []
    Data = {"Train": {"Data": [], "Labels": []}, "Test": {"Data": [], "Labels": []}}

    # Define dimensions for resizing
    width = int(size)
    height = int(size)
    channels = 3
    imgDims = (width, height, channels)

    # Load Data & labels from Matlab file
    raw_data = sio.loadmat(os.path.join(data_path, "FlowerDataLabels.mat"))

    # Extract Data and Labels
    X = raw_data["Data"]
    Y = raw_data["Labels"]

    # Calculate image amount
    totalImages = X.shape[1]

    # Calculate indices for train set
    totalIndices = [index for index in range(totalImages)]

    train_images_indices = [(totalIndices[idx]) for idx in totalIndices if idx not in test_images_indices]

    # Extract images' paths from data path
    img_subset = [os.listdir(data_path)[index] for index in totalIndices]

    # Sort images' paths by numeric ascending order
    img_subset.sort(key=lambda f: int(re.sub('\D', '', f)))

    # Iterate images within folder
    for img in img_subset:

        # Filter jpeg format only
        if img.endswith(".jpeg"):
            # Create image path
            imagePath = os.path.join(data_path, img)

            # Read Raw image
            currentRawImage = cv.imread(imagePath)

            # Resize image based on specified size Gray Scale
            currentResizedImage = cv.resize(currentRawImage, (int(size), int(size)))

            # Adding image  to images array
            images.append(currentResizedImage)

    # Converting arrays to Numpy arrays
    imageData = np.stack(images, axis=0)

    # Splitting data and labels to Train and Test sets
    Data["Train"]["Data"] = imageData[train_images_indices, :, :, :]
    Data["Train"]["Labels"] = Y[:, train_images_indices]
    Data["Test"]["Data"] = imageData[test_images_indices, :, :, :]
    Data["Test"]["Labels"] = Y[:, test_images_indices]

    # Print Summary
    print("Total Images: {0}".format(totalImages))
    print("Total Classes: {0}".format(len(np.unique(Data["Train"]["Labels"]))))
    print("Train Data Shape: {0}".format(Data["Train"]["Data"].shape))
    print("Train Labels Shape: {0}".format(Data["Train"]["Labels"].shape))
    print("Test Data Shape: {0}".format(Data["Test"]["Data"].shape))
    print("Test Labels Shape: {0}".format(Data["Test"]["Labels"].shape))
    print("##################################################\n")

    return Data

def recall_m(y_true, y_pred):
    """
    recall_m(...) - Calculates Recall

    :param y_true: Real Y Labels
    :param y_pred: Predicted Y Labels
    :return: Recall Score
    """
    # Calculates True Positives
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    # Calculates Total Positives
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    # Calculate Recall
    recall = true_positives / (possible_positives + K.epsilon())

    return recall


def precision_m(y_true, y_pred):
    """
       precision_m(...) - Calculates Precision

       :param y_true: Real Y Labels
       :param y_pred: Predicted Y Labels
       :return: Precision Score
       """
    # Calculates True Positives
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    # Calculates Total Predicted Positives
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    # Calculates Precision
    precision = true_positives / (predicted_positives + K.epsilon())

    return precision


def f1_m(y_true, y_pred):
    """
       f1_m(...) - Calculates F1 Score (Harmonic mean between Precision and Recall)

       :param y_true: Real Y Labels
       :param y_pred: Predicted Y Labels
       :return: F1 Score
       """
    # Calculates Precision
    precision = precision_m(y_true, y_pred)

    # Calculates Recall
    recall = recall_m(y_true, y_pred)

    # Calculate F1 Score
    F1 = 2 * ((precision * recall) / (precision + recall + K.epsilon()))

    return F1


def createBaselinePipe(trainable_layers=10, improved=True):
    """
    createBaselinePipe{...) - Modify ResNet50V2 Network from multiclass to binary classifier

    :param trainable_layers: Amount of trainable layers to be updated (from top to bottom)
    :param improved: Adds layers to create Improved Model If True else doesn't change architecture
    :return: Modified ResNet50V2 Model
    """
    # Setting random seed
    set_seeds(678)

    # Initalize ResNet50V2 Model with average pooling and without last FC & Softmax Layers
    ResNet50V2 = keras.applications.resnet_v2.ResNet50V2(include_top=False, weights='imagenet', input_tensor=None,
                                                         input_shape=(224, 224, 3), pooling="avg")
    # Extracting Model Architecture to New Model
    newModel = ResNet50V2.output

    # Creates an Improved Model
    if improved:

        # Adding Batch Normalization Layer
        x = BatchNormalization()(newModel)

        # Adding RELU Activation Layer with 1024 neurons and L2 Regularization kernel
        x = Dense(1024, activation='relu', kernel_initializer=glorot_normal(seed=1),
                  kernel_regularizer=regularizers.l2(0.001))(x)

        # Adding Batch Normalization Layer
        x = BatchNormalization()(x)

        # Adding Dropout with dropping rate of 20 %
        x = Dropout(0.2)(x)

        # Adding RELU Activation Layer with L2 Regularization kernel
        x = Dense(1024, activation='relu', kernel_initializer=glorot_normal(seed=1),
                  kernel_regularizer=regularizers.l2(0.001))(x)

        # Adding Batch Normalization Layer
        x = BatchNormalization()(x)

        # Adding Dropout with dropping rate of 20 %
        x = Dropout(0.2)(x)

        # Adding Dense layer with 1 neuron using Sigmoid activation function
        predictions = Dense(1, activation='softmax', kernel_initializer=glorot_normal(seed=1))(x)

        # Creating Improved Model
        EditedResNet50V2Model = Model(inputs=ResNet50V2.input, outputs=predictions)

        # Number of Layers to train from top to bottom
        layersToBeTrained = trainable_layers + 2
        ly = (-int(layersToBeTrained))

        # Freezing first layers beside last specified trainable layers argument
        for layer in EditedResNet50V2Model.layers[:ly]:
            layer.trainable = False

    # Creates Baseline Model
    else:

        # Adding Dense layer with 1 neuron using Sigmoid activation function
        predictions = Dense(1, activation='sigmoid')(newModel)

        # Creating Baseline Model
        EditedResNet50V2Model = Model(inputs=ResNet50V2.input, outputs=predictions)

        # Freezing first layers beside last specified trainable layers argument
        for layer in EditedResNet50V2Model.layers[:(-1 * trainable_layers)]:
            layer.trainable = False

    return EditedResNet50V2Model

def train_network(X_train, Y_train, params, X_val=None, Y_val=None, Augmentation=True, improved=True):
    """
    train_network(..) - Trains a CNN Model

    :param X_train: Train data (N-training-samples X H X W X C)
    :param Y_train: Train labels (1 x N-training-samples)
    :param params: Dictionary of Hyper-parameters configuration
    :param X_val: Validation data (N-validation-samples X H X W X C)
    :param Y_val: Validation labels (1 x N-validation-samples)
    :param improved: Adds layers to create Improved Model If True else doesn't change architecture
    :return: Model - a KD Forest dictionary of M trees (e.g. M classes)
    """

    print("##################################################")
    print("###############  Training ConvNet  ###############")
    print("Network Hyper-parameters: \n- Optimizer: {0} \n- Batch_size: {1} \n- "
          "Epochs: {2} \n- Trainable Layers: {3}\n".format(params["optimizer"],
                                                          params["batch_size"],
                                                          params["epochs"],
                                                          params["Trainable Layers"]))

    # Create Improved Model or Baseline Model
    if improved:

        # Create an Improved CNN Model
        Model = createBaselinePipe(params["Trainable Layers"], improved=True)

    else:

        # Create a Baseline CNN Model
        Model = createBaselinePipe(params["Trainable Layers"], improved=False)

    # Performance Metrics Configuration
    Metrics = ['acc', tf.metrics.AUC(), f1_m, precision_m, recall_m]

    # Set Model Optimization Configuration
    Model.compile(optimizer=params["optimizer"], loss='binary_crossentropy', metrics=Metrics)

    # Compute Class Weights when Data is imbalanced
    class_weights = class_weight.compute_class_weight('balanced', np.unique(Y_train), Y_train)
    class_weights_dict = dict(zip(np.unique(Y_train), class_weights))

    if Augmentation:

        datagen = ImageDataGenerator(rotation_range=20,
                                    width_shift_range=0.15,
                                    height_shift_range=0.15,
                                    zoom_range=0.15,
                                    shear_range=0.2,
                                    horizontal_flip=True,
                                    fill_mode='reflect',
                                    featurewise_center=False,  # set input mean to 0 over the dataset
                                    samplewise_center=False,  # set each sample mean to 0
                                    featurewise_std_normalization=False,  # divide inputs by std of the dataset
                                    samplewise_std_normalization=False,  # divide each input by its std
                                    zca_whitening=False)  # apply ZCA whitening

        # apply augmentation
        datagen.fit(X_train)

        # configure batch size and retrieve one batch of images
        train_gen = datagen.flow(X_train, Y_train, batch_size=params["batch_size"])

        # train the model through augmented batches
        Model.fit_generator(train_gen, epochs=params["epochs"], shuffle=False,
                                                steps_per_epoch=(len(X_train)/params["batch_size"]))

    # Without Augmentation
    else:

        # Training CNN without\with Validation Set
        if X_val is None:
          Model.fit(X_train, Y_train, batch_size=params["batch_size"], epochs=params["epochs"],
                    class_weight=class_weights_dict)

        else:

            Model.fit(X_train, Y_train, batch_size=params["batch_size"], epochs=params["epochs"],
                               validation_data=(X_val, Y_val), class_weight=class_weights_dict)

    print("##################################################")

    return Model

def test_network(X_test, Y_test, Model, params, add_SVM_on_top=True):
    """
    test_network(..) - Test CNN model by a given Test set

    :param X_test: Test data (N-test-samples x H x W x C )
    :param Y_test: Test labels (1 x N-test-samples)
    :param Model: a pre-trained CNN Model
    :return: results - a dictionary contains test results
          - Accuracy - Accuracy score
          - AUC - Area Under ROC Curve (AUC) score
          - Precision - Precision score
          - Recall - Recall score
          - F1 - F1 score
          - Model - Trained Classifier
    :return: Y_pred - Predictions Vector (1 X N)
    :return: Y_p_proba - Predictions Probability Vector (1 X N)
    """

    print("\n##################################################")
    print("################# Testing Pipe ###################")
    print("##################################################\n")

    # Initialize Results KPI dictionary
    results = {"Accuracy": [],
               "AUC": [],
               "Precision": [],
               "Recall": [],
               "F1": [],
               "Model": Model}

    # Using Representation Learning with SVM
    if add_SVM_on_top:

        # Create predictions vector
        Y_pred = Model.predict(X_test)

        # Calculate True Positive Rate and False Positive Rate
        FPR, TPR, thresholds = roc_curve(Y_test, Y_pred)

        # Evaluate ResNet50V2 Model
        results["Accuracy"] = Model.score(X_test, Y_test)
        results["AUC"] = auc(FPR, TPR)
        results["F1"] = f1_score(Y_test, Y_pred)

        # Calculate precision recall curve values
        precision, recall, t = precision_recall_curve(Y_test, Y_pred)
        results["Precision"] = np.average(precision)
        results["Recall"] = np.average(recall)

        # Create predictions vector
        Y_p_prob = Model.predict_proba(X_test)

    else:

        # Evaluate ResNet50V2 Model
        loss, results["Accuracy"], results["AUC"], results["F1"], results["Precision"], results["Recall"] = Model.evaluate(X_test, Y_test, verbose=0)

        # Create predictions vector
        Y_p_prob = Model.predict(X_test).ravel()
        Y_pred = np.where(Y_p_prob<params["Threshold"], 0, 1)

    return results, Y_pred, Y_p_prob

def KfoldCrossValidation(X_train, Y_train, k, params):
    """
    KfoldCrossValidation(..) - Performs K-Fold Cross-Validation on a given data

    :param X_train: Train data (N-samples X Height x Width x Channels)
    :param Y_train: Train labels (1 x N-samples)
    :param k: Amount of folds to be tested (e.g. K-fold)
    :param params:
    :return: averagedError - The mean validation set error
    """

    print("{0}-Folds Cross Validation is being executed...".format(k))

    # Initialize arrays
    errors = []

    # Seed the random number generator
    np.random.seed(0)

    # Extract unique classes
    uniqueClasses = np.unique(Y_train)

    # Iterating K folds
    for fold in range(0, k):

        # Initialize indices arrays
        trainIndices, validationIndices = [], []

        # Change seed per each fold
        set_seeds(fold)

        # Split to Train and Validation sets
        for c in uniqueClasses:

            # Extracting current class indices
            idxList = np.argwhere(Y_train == c)
            idxList = idxList.reshape(1, len(idxList))[0]

            # Finding the index for partitioning the data
            trainSetMaxInd = math.floor((1 - params["validation_split"]) * len(idxList))

            # Shuffle indices to be selected
            np.random.shuffle(idxList)

            # Create and extend Train & Validation sets selected indices
            trainIndices = np.concatenate((trainIndices, np.array(idxList[0:trainSetMaxInd]))).astype(int)
            validationIndices = np.concatenate((validationIndices, np.array(idxList[trainSetMaxInd:]))).astype(int)

        # Extracting train and validation sets
        X_t, X_v = X_train[trainIndices, :, :, :], X_train[validationIndices, :, :, :]

        # Extracting train and validation labels
        Y_t, Y_v = Y_train[trainIndices], Y_train[validationIndices]

        # Train classifier on train set
        Model = train_network(X_t, Y_t, params, X_v, Y_v)

        # Test model on validation set
        results, Y_pred, Y_p_proba = test_network(X_v, Y_v, Model)

        # Calculate current fold error
        error = 100*(1.0 - float(results["Accuracy"]))

        # Appending current fold's error
        errors.append(error)

    # Calculate the averaged error
    averagedError = np.average(np.array(errors))

    print("{0}-Folds Cross Validation Averaged Error of {1}%".format(k, averagedError))

    return averagedError


def plot3DErrorPerformanceChart(x_param, c_param, y_param, k=3):
    """
    plot3DErrorPerformanceChart(...) - Plot a 3D chart of experiment results

    :param x_param: Parameter to be plotted as X axis
    :param c_param: Parameter to be plotted as Color axis
    :param y_param: Parameter to be plotted as Y axis
    :param k: Amount of folds to be tested e.g. K-fold. Default is 5.
    :return:
    """
    x_axis = np.array(x_param)  # X-axis
    y_axis = np.array(y_param)  # Y-Axis
    c_axis = np.array(c_param)  # Color Axis

    # Data Dictionary
    data = {"Learning Rate": x_axis,
            "rho": c_axis,
            "Error": y_axis}

    # Calculate unique colors parameter
    uniqueColorParameter= len(np.unique(c_axis))

    # Plot Configuration
    plt.figure(figsize=(20, 8))
    plt.title('Hyperparameters Configurations Error on Validation Set', fontsize=20)

    # Make the plot
    plot = sns.lineplot(x="Learning Rate", y="Error", hue="rho", legend="full",
                        palette=sns.color_palette(n_colors=uniqueColorParameter),
                        data=data)
    # Set axis labels
    plt.ylabel('{0}-Fold CV Error (%)'.format(k), fontsize=18)
    plt.xlabel('Learning Rate', fontsize=18)
    plt.legend(title='rho', fontsize=20)

    # plot = sns.barplot(x="Optimizer", y="Error", data=data)

    # for p in plot.patches:
    #     plot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center',
    #                    va='center', xytext=(0, 10), textcoords='offset points')

    # Display plot
    plt.show()
    plt.close()

def plot_recall_precision(y_pred, y_test):
    """
    plot_recall_precision(...) - plot recall precision graph

    :param y_pred: Predictions vector (1 X N)
    :param y_test: True Labels vector (1 X N)
    :return:
    """

    # Calculates Curve Values
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    ap = average_precision_score(y_test, y_pred)

    # Plot Configuration
    fig, ax = plt.subplots()
    plt.title('2-class Precision-Recall curve: average precision %.2f' % ap)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])

    # plot the roc curve for the model
    plt.plot(recall, precision, marker='.')

    # Display plot
    plt.show()
    plt.close()


def hyperparameterOptimizationExperiment(X_train, Y_train, params):
    """
    hyperparameterOptimizationExperiment(...) - Conduct Hyper-parameters tuning Experiments

    :param X_train: Train data (N-training-samples X H X W X C)
    :param Y_train: Train labels (1 x N-training-samples)
    :param params: Dictionary of Experiment Configuration of Hyper-parameters
    :return: Dictionary of Independent Experiments Configurations and Performance Results:
            - expResults["Optimizer"] - Used Optimizers
            - expResults["Batch Size"] - Used Batch Size
            - expResults["Epochs"] - Used Epochs
            - expResults["Trainable Layers"] - Used Trainable Layers
            - expResults["validation_split"] - Used Validation Split
            - expResults["Errors"] - Averaged K-CV Error
    """
    # Metrics Configuration
    Metrics = [tf.metrics.AUC(), 'acc', f1_m, precision_m, recall_m]

    # Initialize Experiment's Results Dictionary
    expResults = {"Errors": [],
                  "Optimizer": [],
                  "Batch Size": [],
                  "Epochs": [],
                  "Trainable Layers":[],
                  "validation_split":[],
                  "Class Weight":[],
                  'Threshold':[]}

    for lr in np.arange(0,1,0.1):

        currentExpConfig = {'batch_size': params["batch_size"],
                     'optimizer': params["optimizer"],
                     'epochs': params["epochs"],
                     'Trainable Layers': params["Trainable Layers"],
                     'validation_split': params["validation_split"],
                     'LR':lr,
                     'Threshold':params["Threshold"]}

        # Setting random seeds
        set_seeds(678)

        # Perform K-Cross-Validation using current experiment configuration
        averagedError = KfoldCrossValidation(X_train, Y_train, k=3, params=currentExpConfig)

        # Document Experiment Configurations and Results
        expResults["Optimizer"].append(currentExpConfig["optimizer"])
        expResults["Batch Size"].append(currentExpConfig["batch_size"])
        expResults["Epochs"].append(currentExpConfig["epochs"])
        expResults["Trainable Layers"].append(currentExpConfig["Trainable Layers"])
        expResults["validation_split"].append(currentExpConfig["validation_split"])
        expResults["Threshold"].append(currentExpConfig["Threshold"])
        expResults["LR"].append(lr)
        expResults["Errors"].append(averagedError)

        # EvaluateModel(test_X, test_Y, EditedResNet50V2Model)
        K.clear_session()

    # Plot Experiment's results
    plot3DErrorPerformanceChart(expResults["LR"], expResults["Optimizer"], expResults["Errors"], k=3)

    return expResults

def plot_confusion_matrix(cm, classes, isTrueVector, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
  """
    plot_confusion_matrix(...) - Prints and plots a confusion matrix.
    Normalization can be applied by setting `normalize=True`.

  :param cm - Confusion Matrix ()
  :param classes - Class labels (array)
  :param normalize  - Normalization parameter (Boolean)
  :param title - Plot title
  :param cmap - a Matplotlib Color Map
  """

  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


  width_in_inches = 6
  height_in_inches = 5
  dots_per_inch = 70

  plt.figure(figsize=(width_in_inches, height_in_inches))

  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.show()


def Evaluate(X_test, Y_test, Y_pred, Y_p_prob, results, threshold=0.5):
    """
    Evaluate(..) - Performance Evaluation of Test Results
    :param: X_test - Test data (N-test-samples x H x W x C)
    :param Y_test - Contains a list of the actual test images labels
    :param Y_pred - Contains a list of the predicted test images labels
    :param results - a dictionary contains test results
          - Accuracy - Accuracy score
          - AUC - Area Under ROC Curve (AUC) score
          - Precision - Precision score
          - Recall - Recall score
          - F1 - F1 score
    :param: threshold - Values above this threshold will be classified as flowers (Default is 0.5)
    :return:
        - accuracy - Accuracy Score
        - Presents Accuracy score & Confusion Matrix
    """

    # Print Evaluation Summary
    print("##################################################")
    print("############# CNN Evaluation Results #############")
    print("##################################################")
    print("Accuracy Score = " + str(results["Accuracy"]))
    print("AUC Score = " + str(results["AUC"]))
    print("Precision = " + str(results["Precision"]))
    print("Recall = " + str(results["Recall"]))
    print("F1 Score = " + str(results["F1"]))
    print("##################################################")

    # Plotting Precision-Recall Curve
    plot_recall_precision(Y_pred, Y_test)

    # Extracting Actual and Predicted Labels from Results
    Y_t = np.where(Y_test==0, "Non-Flower", "Flower")
    Y_p = np.where(Y_pred==0, "Non-Flower", "Flower")

    isTrueVector = Y_t==Y_p

    # Calculating amount of classes
    classes = np.unique(Y_t)

    # Computing confusion matrix
    cnf_matrix = sk.metrics.confusion_matrix(Y_t, Y_p)
    np.set_printoptions(precision=2)

    # Accuracy calculation
    accuracy = (np.sum(isTrueVector) / len(isTrueVector)) * 100

    # Plotting confusion matrix
    fig = plot_confusion_matrix(cnf_matrix, classes, isTrueVector, title='Improved Pipe Confusion matrix')

    # Find top five errors of type one & two
        # OneMistakeInd, TwoMistakeInd = mistakeTypes(Y_test, Y_p_prob, isTrueVector)
        #
        # # Get top 5 Errors magnitude
        # OneMistakeTop_5_Errors = Nmaxelements(list(OneMistakeInd.keys()),5)
        # TwoMistakeTop_5_Errors = Nmaxelements(list(TwoMistakeInd.keys()),5)
        #
        # # Get top 5 errors indices
        # MistakeOneIndices = [OneMistakeInd[k] for k in OneMistakeTop_5_Errors if k in OneMistakeInd]
        # MistakeTwoIndices = [TwoMistakeInd[k] for k in TwoMistakeTop_5_Errors if k in TwoMistakeInd]
        #
        # # Display Images
        # DisplayImages(MistakeOneIndices, OneMistakeTop_5_Errors, "Type One",X_test)
        # DisplayImages(MistakeTwoIndices, TwoMistakeTop_5_Errors, "Type Two", X_test)

    return accuracy


def train_improved_pipe(X_train, Y_train, ImprovedModel):
    """
    train_improved_pipe(...) - Trains the Improve Pipe using Representation Learning and SVM Classifier

    :param X_train: Train data (N-training-samples X H X W X C)
    :param Y_train: Train labels (1 x N-training-samples)
    :param ImprovedModel: CNN for feature engineering
    :return: svm_model: Trained SVM Classifier
    :return: feature_engineer: Trained CNN
    """

    set_seeds(678)

    # ResNet50V2 Model Representation Layer Modifier
    feature_engineer = Model(inputs=ImprovedModel.input, outputs=ImprovedModel.get_layer(index=-2).output)

    print("\n##################################################")
    print("## Feature Extraction Using an improved ConvNet ##")
    print("##################################################\n")

    # Extracting Features using Improved CNN
    train_features = feature_engineer.predict(X_train)

    # Creates a SVM classifier
    svm_model = SVC()

    # Best Hyper-parameters configuration (Derived from our 1st Task results)
    bestParamsClf = {'decision_function_shape': 'ovo', 'probability': True, 'random_state': 42, 'kernel': 'poly',
                                                                                    'gamma': 0.1, 'degree': 2, 'C': 0.1}

    # Setting Hyper-parameters
    svm_model.set_params(**bestParamsClf)

    print("##################################################")
    print("########### Training a SVM Classifier ############")
    print("##################################################\n")

    # Train SVM on Engineered Features
    svm_model.fit(train_features, Y_train)

    return svm_model, feature_engineer


def DisplayImages(idx, errorVec, errorType, X_test):
    """
    DisplayImages(...) - Displays the top N images with the largest errors

    :param idx: Index
    :param errorVec: Vector of Errors (1 X N samples)
    :param errorType: Type 1 or Type 2 Error
    :param X_test: Test data (N-test-samples x H x W x C )
    """
    print("\n")
    print("### Display " + errorType + " Errors #####")
    for idx, ind_val in enumerate(idx):
        print("Error Value " + str(errorVec[idx]))
        plt.imshow(X_test[ind_val, :, :, :])
        plt.show()


def Nmaxelements(list1, N):
    """
     Nmaxelements(...) - Retrives the N largest values from a given list
    :param list1:
    :param N:
    :return:
    """
    final_list = []

    if len(list1) < N:
        N = len(list1)

    for i in range(0, N):
        max1 = 0

        for j in range(len(list1)):
            if list1[j] > max1:
                max1 = list1[j];

        list1.remove(max1);
        final_list.append(max1)

    return final_list

def mistakeTypes(Y_t, Y_p, mistakesInd):
    """
    mistakeTypes(...) - Finds type one and two errors and retrieves indices and magnitude as
                        dictionaries - Key = Error magnitude, Value = Index

    :param: mistakesIndices: Mistakes indices
    :param: Y_t: Ground truth Vector
    :param: Y_p: Predictions Vector
    :param: OneMistakeInd - all the mistakes one indices in the ground truth vector (self.y_test)
    :return: TwoMistakeInd - all the mistakes Two indices in the ground truth vector (self.y_test)
    """
    Y_p_abs = roundTresh(Y_p)

    OneMistakeInd = {}
    TwoMistakeInd = {}

    for idx, val in enumerate(mistakesInd):
      if val==False:
        if Y_t[idx] == 1 & Y_p_abs[idx][0] == 0:
          TwoMistakeInd[1 - Y_p[idx][0]] = idx
        else:
          OneMistakeInd[Y_p[idx][1]] = idx
    return OneMistakeInd, TwoMistakeInd

def roundTresh(Y_pred):
  """
  roundTresh(...) - converts two categorical scores into binary scores,
                    for example:  [0.01 0.98] => [0 1]  or [0.89 0.2] => [1 0]

  :param Y_pred: two categorical vector- the ConvNet output
  :return: binary scores prediction vector
  """
  binary_pred = []
  for i in range(0, np.size(Y_pred, 0)):
      if Y_pred[i][0] > Y_pred[i][1]:
          binary_pred.append([1, 0])
      else:
          binary_pred.append([0, 1])
  binary_pred = np.asanyarray(binary_pred)
  return binary_pred


def main(data_path, test_images_indices, data_augmentation=True, add_SVM_on_top=True ,improved=True):
    """
    main(...) - Main Procedure which executes Baseline Pipe or Improved Pipe

    :param data_path: Data path to Flower Data directory
    :param test_images_indices: Selected test images indices
    :param data_augmentation: Perform Data Augmentation If True
    :param add_SVM_on_top: Perform Representation Learning & SVM Classifier If True
    """

    # Best found Hyper-parameters Configuration
    params = {'Trainable Layers': 9,
              'batch_size': 32,
              'optimizer': "Adadelta",
               'epochs': 46,
               'validation_split': 0.1,
               'Threshold': 0.5}

    # Load Images, Resize to 224 x 224 x 3 and Split to Train and Test
    data = loadImages(data_path, test_images_indices, size=224)

    # Normalizing Data
    X_train = data["Train"]["Data"]/255.0
    X_test = data["Test"]["Data"]/255.0

    # Extracting Labels
    Y_train = data["Train"]["Labels"][0]
    Y_test = data["Test"]["Labels"][0]

    # Random seed generation
    set_seeds(678)

    # Using Representation Learning with SVM
    if add_SVM_on_top:

        # Train Improved Pipe on training set
        improvedModel = train_network(X_train, Y_train, params, Augmentation=data_augmentation, improved=improved)

        # Train SVM on Engineered Features
        svm_model, feature_engineer = train_improved_pipe(X_train, Y_train, improvedModel)

        # Extracting Features using Improved CNN
        test_features = feature_engineer.predict(X_test)

        # Test model on Test set
        results, Y_pred, Y_p_prob = test_network(test_features, Y_test, svm_model, params, add_SVM_on_top=True)

        # Evaluate Model Performance
        Evaluate(X_test, Y_test, Y_pred, Y_p_prob, results, params["Threshold"])

    # Baseline Pipe
    else:

        # Train Improved Pipe on training set
        BaselineModel = train_network(X_train, Y_train, params, Augmentation=data_augmentation, improved=improved)

        # Test model on validation set
        results, Y_pred, Y_p_prob = test_network(X_test, Y_test, BaselineModel,params, add_SVM_on_top=improved)

        # Evaluate Model Performance
        Evaluate(X_test, Y_test, Y_pred, Y_p_prob, results, params["Threshold"])


if __name__ == '__main__':

    # FlowerData folder path
    data_path = r"D:\MSc Degree\Courses\1st Year\1. Learning, representation, and Computer Vision\Task 2\FlowerData"

    # Setting test images indices
    test_images_indices = list(range(301, 472))

    # Execute Main Procedure
    main(data_path, test_images_indices, data_augmentation=True, add_SVM_on_top=True, improved=False)