import numpy as np
import os
import time
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Loads and preprocess MNIST dataset
def loadAndPreprocessMNISTDataset():

    # Print messages
    print("\nLoading MNIST dataset...")
    print("Splitting to train and test sets...")

    # Load MNIST dataset
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    # Reshape training set to have a single channel (i.e., grayscale) from (60000, 28, 28) into (60000, 28, 28, 1)
    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))

    # Reshape test set to have a single channel (i.e., grayscale) from (10000, 28, 28) into (10000, 28, 28, 1)
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

    # Convert tarrget values to one hot encode representation
    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)

    # Print train and test sets dimensions
    print("X_train Dimensions: {0}".format(X_train.shape))
    print("Y_train Dimensions: {0}".format(Y_train.shape))
    print("X_test Dimensions: {0}".format(X_test.shape))
    print("Y_test Dimensions: {0}".format(Y_test.shape))

    return X_train, Y_train, X_test, Y_test


# Normalize data
def dataNormalization(train, test):

    # Convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')

    # Normalize train & test sets from range 0-255 to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0

    # return normalized images
    return train_norm, test_norm



# Define your first CNN model architecture here
def CNN_Model():

    # Import layers from Keras
    from tensorflow.keras.layers import Conv2D
    from tensorflow.keras.layers import MaxPooling2D
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.optimizers import SGD

    # Sequential model is appropriate for a stack of layers where each layer has exactly one input & output tensors.
    model = Sequential()

    # Add a 2D Convolution layer with 32 kernels of size of (3, 3) that receive a 28x28x1 grayscale image input.
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))

    # Add a 2D Max-pooling layer with a kernel of size of (2, 2).
    model.add(MaxPooling2D((2, 2)))

    # Add a 2D Convolution layer with 32 kernels of size of (3, 3) that receive an input shape of (24, 24, 32).
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(24, 24, 32)))

    # Add a 2D Max-pooling layer with a kernel of size of (2, 2).
    model.add(MaxPooling2D((2, 2)))

    # Flatten the previous layer output.
    model.add(Flatten())

    # Add a FC layer with 100 neurons. Use relu activation and "he_uniform" variance scaling initializer.
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))

    # Add a FC layer with a single neuron per class. Use "softmax" activation.
    model.add(Dense(10, activation='softmax'))

    # Create SGD optimizer object with learning rate of 0.01, and momentum of 0.9.
    opt = SGD(lr=0.01, momentum=0.9)

    # Use compile() to configure the model for training (e.g., optimizer, loss, and metrics).
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# Evaluate the CNN model using K-Fold Cross-Validation procedure
def evaluateCNNModel(dataX, dataY, n_folds=3):

    # For reproducible results
    np.random.seed(1111)

    # Initialize lists of scores and histories
    scores, histories = list(), list()

    # Define the cross validation object
    kfold = KFold(n_folds, shuffle=True, random_state=1)

    # Fold counter
    f = 0

    # Enumerate splits per each fold
    for train_ix, test_ix in kfold.split(dataX):

        f += 1

        # Create CNN model architecture
        model = CNN_Model()

        # Select train and test sets for the current fold
        trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]

        print("\n # Fold %d - Training a CNN classifier..." % f)

        # Fit the CNN model using 10 epochs and a batch size of 32
        history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=1)

        # Evaluate CNN model
        _, acc = model.evaluate(testX, testY, verbose=0)
        print('Fold %d Accuracy: %.3f' % (f, acc * 100.0))

        # Store scores and history
        scores.append(acc)
        histories.append(history)

    return scores, histories


# Plot diagnostic learning curves based on loss and accuracy
def generateDiagnosticsSummary(histories):

    # Create Fig template with two subplots
    fig, (ax1, ax2) = plt.subplots(nrows=2)

    # Iterate performance's history
    for i in range(len(histories)):

        # Plot Categorical Cross Entropy loss as function of Epochs
        ax1.title.set_text('Categorical Cross Entropy Loss vs. Epochs (3-CV)')
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("CCE Loss")
        ax1.plot(histories[i].history['loss'], color='blue', label='train')
        ax1.plot(histories[i].history['val_loss'], color='green', label='test')
        ax1.legend(("Train", "Validation"))

        # Plot Classification Accuracy as function of Epochs
        ax2.title.set_text('Classification Accuracy vs. Epochs (3-CV)')
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Accuracy")
        ax2.plot(histories[i].history['accuracy'], color='blue', label='train')
        ax2.plot(histories[i].history['val_accuracy'], color='green', label='test')
        ax2.legend(("Train", "Validation"))

    # Use tight layout
    fig.tight_layout()

    # Display plot
    plt.show()


# CNN performance summary
def generatePerformanceSummary(scores):

    # Print results per fold
    for i in range(len(scores)):

        print("Fold %d - Accuracy of %.3f" % (i+1, scores[i]))

    # Print performance statistics
    print('Accuracy Statistics: mean=%.3f, std= %.3f, folds=%d' % (np.mean(scores) * 100, np.std(scores) * 100, len(scores)))

    # Box and Whisker plots of results
    fig = plt.figure()
    fig.suptitle('Accuracy Box Plot', fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    ax.boxplot(scores)
    ax.set_title('(3-CV)')
    ax.set_xlabel('Fold')
    ax.set_ylabel('Accuracy')

    # Display plot
    plt.show()

def plotDigits(X_train, Y_train, N=10):
    images = X_train[:N]
    labels = Y_train[:N]

    num_row = 2
    num_col = 5

    # plot images
    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5 * num_col, 2 * num_row))
    for i in range(N):
        ax = axes[i // num_col, i % num_col]
        ax.imshow(images[i][:, :, 0], cmap='gray')
        ax.set_title('Label: {}'.format(np.argmax(labels[i])))
    plt.tight_layout()
    plt.show()

# run the test harness for evaluating a model
def Main():

    # Calculate start time
    start_time = time.time()

    # Loading and preprocess MNIST dataset
    X_train, Y_train, X_test, Y_test = loadAndPreprocessMNISTDataset()

    # Normalize images
    X_train, X_test = dataNormalization(X_train, X_test)

    # Evaluate CNN Model
    scores, histories = evaluateCNNModel(X_train, Y_train)

    print("--- Total training time: %.3f minutes ---" % ((time.time() - start_time)/60))

    # learning curves
    generateDiagnosticsSummary(histories)

    # summarize estimated performance
    generatePerformanceSummary(scores)

    # Plot 10 digits from MNIST's training set
    plotDigits(X_train, Y_train)


# Run Main procedure
if __name__ == "__main__":
    Main()
