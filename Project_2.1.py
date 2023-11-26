# import libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# %% Step 1 Data Processing
##############################################################################
print('Step 1...')
# input image shape
image_width = 100
image_height = 100
image_channel = 3

# establish the train, validation, and testing data directories
train_dir = '/Users/nima.db/Documents/Fall 2023/AER850/Project 2/AER_850_Project_2/Data/Train'
validation_dir = '/Users/nima.db/Documents/Fall 2023/AER850/Project 2/AER_850_Project_2/Data/Validation'

# data augmentation
train_datagen = ImageDataGenerator(rescale=1./255,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale=1./255)

# creating the train, validation, and test generators
batch_size = 32

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(image_width, image_height),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(image_width, image_height),
    batch_size=batch_size,
    class_mode='categorical'
)

test_datagen = ImageDataGenerator(rescale=1./255)

##############################################################################
# %% Step 2 Neural Network Architecture Design
##############################################################################
print('Step 2...')

# create a Sequential model
model = Sequential()

# convolutional layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_width,image_height,image_channel)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))

# maxPooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# flatten layer
model.add(Flatten())

# dense layers
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='elu'))
model.add(Dropout(0.5))  # Adding dropout for regularization

# softmax dense layer with 4 neurons
model.add(Dense(4, activation='softmax'))  # Final output layer with 4 neurons for 4 classes

# compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# summary of the model architecture
model.summary()

# fitting the model
history = model.fit(train_generator,
                    steps_per_epoch=len(train_generator),
                    epochs=20,
                    validation_data=validation_generator,
                    validation_steps=len(validation_generator))

# Accessing loss values from the history object
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Plotting training and validation loss on a separate figure
plt.figure()
epochs = range(1, len(train_loss) + 1)
plt.plot(epochs, train_loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Accessing accuracy values from the history object
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Plotting training and validation accuracy on a separate figure
plt.figure()
epochs = range(1, len(train_acc) + 1)
plt.plot(epochs, train_acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

model.save('model_5.h5')
##############################################################################
