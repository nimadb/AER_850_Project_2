# %% Step 2 Neural Network Architecture Design - 1
##############################################################################
'''
# convolutional layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_width,image_height,image_channel)))
model.add(Conv2D(64, (3, 3), activation='relu'))

# maxPooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# flatten layer
model.add(Flatten())

# dense layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Adding dropout for regularization
model.add(Dense(64, activation='elu'))

# softmax dense layer with 4 neurons
model.add(Dense(4, activation='softmax'))  # Final output layer with 4 neurons for 4 classes

# compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# summary of the model architecture
model.summary()

# fitting the model
history = model.fit(train_generator,
                    steps_per_epoch=len(train_generator),
                    epochs=10,
                    validation_data=validation_generator,
                    validation_steps=len(validation_generator))

Results:
Step 1...
Found 1600 images belonging to 4 classes.
Found 800 images belonging to 4 classes.
Step 2...
Model: "sequential_3"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_6 (Conv2D)           (None, 98, 98, 32)        896       
                                                                 
 conv2d_7 (Conv2D)           (None, 96, 96, 64)        18496     
                                                                 
 max_pooling2d_3 (MaxPoolin  (None, 48, 48, 64)        0         
 g2D)                                                            
                                                                 
 flatten_3 (Flatten)         (None, 147456)            0         
                                                                 
 dense_8 (Dense)             (None, 128)               18874496  
                                                                 
 dropout_3 (Dropout)         (None, 128)               0         
                                                                 
 dense_9 (Dense)             (None, 64)                8256      
                                                                 
 dense_10 (Dense)            (None, 4)                 260       
                                                                 
=================================================================
Total params: 18902404 (72.11 MB)
Trainable params: 18902404 (72.11 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
Epoch 1/10
50/50 [==============================] - 23s 454ms/step - loss: 1.1989 - accuracy: 0.4156 - val_loss: 0.5501 - val_accuracy: 0.7500
Epoch 2/10
50/50 [==============================] - 22s 441ms/step - loss: 0.5472 - accuracy: 0.7094 - val_loss: 0.3602 - val_accuracy: 0.7500
Epoch 3/10
50/50 [==============================] - 22s 445ms/step - loss: 0.4056 - accuracy: 0.7394 - val_loss: 0.3547 - val_accuracy: 0.7500
Epoch 4/10
50/50 [==============================] - 22s 446ms/step - loss: 0.3906 - accuracy: 0.7487 - val_loss: 0.3924 - val_accuracy: 0.7475
Epoch 5/10
50/50 [==============================] - 23s 454ms/step - loss: 0.3868 - accuracy: 0.7369 - val_loss: 0.3640 - val_accuracy: 0.7500
Epoch 6/10
50/50 [==============================] - 22s 435ms/step - loss: 0.3724 - accuracy: 0.7556 - val_loss: 0.3716 - val_accuracy: 0.7500
Epoch 7/10
50/50 [==============================] - 22s 442ms/step - loss: 0.4111 - accuracy: 0.7362 - val_loss: 0.3467 - val_accuracy: 0.7513
Epoch 8/10
50/50 [==============================] - 22s 448ms/step - loss: 0.3759 - accuracy: 0.7437 - val_loss: 0.3533 - val_accuracy: 0.7500
Epoch 9/10
50/50 [==============================] - 21s 430ms/step - loss: 0.3689 - accuracy: 0.7531 - val_loss: 0.3518 - val_accuracy: 0.7500
Epoch 10/10
50/50 [==============================] - 21s 429ms/step - loss: 0.3682 - accuracy: 0.7556 - val_loss: 0.3478 - val_accuracy: 0.7500
Part 5...
Found 192 images belonging to 4 classes.
6/6 [==============================] - 2s 251ms/step - loss: 0.5833 - accuracy: 0.7396
Test accuracy: 0.7395833134651184
Test loss: 0.583323061466217

##############################################################################
'''

# %% Step 2 Neural Network Architecture Design - 2
##############################################################################
'''
print('Step 2...')

# convolutional layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_width,image_height,image_channel)))
model.add(Conv2D(64, (3, 3), activation='relu'))

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
                    epochs=10,
                    validation_data=validation_generator,
                    validation_steps=len(validation_generator))

Results:
Step 1...
Found 1600 images belonging to 4 classes.
Found 800 images belonging to 4 classes.
Step 2...
Model: "sequential_4"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_8 (Conv2D)           (None, 98, 98, 32)        896       
                                                                 
 conv2d_9 (Conv2D)           (None, 96, 96, 64)        18496     
                                                                 
 max_pooling2d_4 (MaxPoolin  (None, 48, 48, 64)        0         
 g2D)                                                            
                                                                 
 flatten_4 (Flatten)         (None, 147456)            0         
                                                                 
 dense_11 (Dense)            (None, 128)               18874496  
                                                                 
 dense_12 (Dense)            (None, 64)                8256      
                                                                 
 dropout_4 (Dropout)         (None, 64)                0         
                                                                 
 dense_13 (Dense)            (None, 4)                 260       
                                                                 
=================================================================
Total params: 18902404 (72.11 MB)
Trainable params: 18902404 (72.11 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
Epoch 1/10
50/50 [==============================] - 22s 439ms/step - loss: 1.1915 - accuracy: 0.4575 - val_loss: 0.4925 - val_accuracy: 0.7500
Epoch 2/10
50/50 [==============================] - 22s 438ms/step - loss: 0.5567 - accuracy: 0.7044 - val_loss: 0.3618 - val_accuracy: 0.7500
Epoch 3/10
50/50 [==============================] - 22s 436ms/step - loss: 0.4212 - accuracy: 0.7519 - val_loss: 0.3757 - val_accuracy: 0.7500
Epoch 4/10
50/50 [==============================] - 22s 436ms/step - loss: 0.4210 - accuracy: 0.7331 - val_loss: 0.3874 - val_accuracy: 0.7500
Epoch 5/10
50/50 [==============================] - 22s 440ms/step - loss: 0.3831 - accuracy: 0.7550 - val_loss: 0.3510 - val_accuracy: 0.7500
Epoch 6/10
50/50 [==============================] - 22s 433ms/step - loss: 0.3796 - accuracy: 0.7625 - val_loss: 0.3610 - val_accuracy: 0.7500
Epoch 7/10
50/50 [==============================] - 22s 440ms/step - loss: 0.4034 - accuracy: 0.7406 - val_loss: 0.3466 - val_accuracy: 0.7500
Epoch 8/10
50/50 [==============================] - 22s 431ms/step - loss: 0.3760 - accuracy: 0.7469 - val_loss: 0.3715 - val_accuracy: 0.7500
Epoch 9/10
50/50 [==============================] - 22s 437ms/step - loss: 0.3732 - accuracy: 0.7556 - val_loss: 0.3453 - val_accuracy: 0.7500
Epoch 10/10
50/50 [==============================] - 23s 459ms/step - loss: 0.3602 - accuracy: 0.7638 - val_loss: 0.3492 - val_accuracy: 0.7500
'''
##############################################################################
# %%
'''
# convolutional layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_width,image_height,image_channel)))
model.add(Conv2D(64, (3, 3), activation='relu'))

# maxPooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# flatten layer
model.add(Flatten())

# dense layers
model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='elu'))
# model.add(Dropout(0.5))  # Adding dropout for regularization

# softmax dense layer with 4 neurons
model.add(Dense(4, activation='softmax'))  # Final output layer with 4 neurons for 4 classes

# compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# summary of the model architecture
model.summary()

# fitting the model
history = model.fit(train_generator,
                    steps_per_epoch=len(train_generator),
                    epochs=10,
                    validation_data=validation_generator,
                    validation_steps=len(validation_generator))
Results:
Step 1...
Found 1600 images belonging to 4 classes.
Found 800 images belonging to 4 classes.
Step 2...
Model: "sequential_5"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_10 (Conv2D)          (None, 98, 98, 32)        896       
                                                                 
 conv2d_11 (Conv2D)          (None, 96, 96, 64)        18496     
                                                                 
 max_pooling2d_5 (MaxPoolin  (None, 48, 48, 64)        0         
 g2D)                                                            
                                                                 
 flatten_5 (Flatten)         (None, 147456)            0         
                                                                 
 dense_14 (Dense)            (None, 128)               18874496  
                                                                 
 dense_15 (Dense)            (None, 4)                 516       
                                                                 
=================================================================
Total params: 18894404 (72.08 MB)
Trainable params: 18894404 (72.08 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
Epoch 1/10
50/50 [==============================] - 22s 441ms/step - loss: 1.1632 - accuracy: 0.4906 - val_loss: 0.4110 - val_accuracy: 0.7500
Epoch 2/10
50/50 [==============================] - 22s 446ms/step - loss: 0.4589 - accuracy: 0.7362 - val_loss: 0.3734 - val_accuracy: 0.7500
Epoch 3/10
50/50 [==============================] - 22s 435ms/step - loss: 0.3834 - accuracy: 0.7506 - val_loss: 0.3695 - val_accuracy: 0.7500
Epoch 4/10
50/50 [==============================] - 22s 431ms/step - loss: 0.3868 - accuracy: 0.7719 - val_loss: 0.3408 - val_accuracy: 0.8050
Epoch 5/10
50/50 [==============================] - 22s 435ms/step - loss: 0.3689 - accuracy: 0.7606 - val_loss: 0.3396 - val_accuracy: 0.7625
Epoch 6/10
50/50 [==============================] - 22s 439ms/step - loss: 0.3490 - accuracy: 0.7769 - val_loss: 0.5226 - val_accuracy: 0.6612
Epoch 7/10
50/50 [==============================] - 23s 462ms/step - loss: 0.3517 - accuracy: 0.7788 - val_loss: 0.3405 - val_accuracy: 0.7500
Epoch 8/10
50/50 [==============================] - 23s 453ms/step - loss: 0.3432 - accuracy: 0.7869 - val_loss: 0.3435 - val_accuracy: 0.9337
Epoch 9/10
50/50 [==============================] - 22s 439ms/step - loss: 0.3488 - accuracy: 0.7919 - val_loss: 0.4814 - val_accuracy: 0.6725
Epoch 10/10
50/50 [==============================] - 22s 440ms/step - loss: 0.3714 - accuracy: 0.7681 - val_loss: 0.4223 - val_accuracy: 0.8413
'''
#############################3
# 4th one
'''
# convolutional layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_width,image_height,image_channel)))
model.add(Conv2D(64, (3, 3), activation='relu'))

# maxPooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# flatten layer
model.add(Flatten())

# dense layers
model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='elu'))
# model.add(Dropout(0.5))  # Adding dropout for regularization

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

Step 1...
Found 1600 images belonging to 4 classes.
Found 800 images belonging to 4 classes.
Step 2...
Model: "sequential_6"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_12 (Conv2D)          (None, 98, 98, 32)        896       
                                                                 
 conv2d_13 (Conv2D)          (None, 96, 96, 64)        18496     
                                                                 
 max_pooling2d_6 (MaxPoolin  (None, 48, 48, 64)        0         
 g2D)                                                            
                                                                 
 flatten_6 (Flatten)         (None, 147456)            0         
                                                                 
 dense_16 (Dense)            (None, 128)               18874496  
                                                                 
 dense_17 (Dense)            (None, 4)                 516       
                                                                 
=================================================================
Total params: 18894404 (72.08 MB)
Trainable params: 18894404 (72.08 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
Epoch 1/20
50/50 [==============================] - 23s 454ms/step - loss: 0.9514 - accuracy: 0.5444 - val_loss: 0.3533 - val_accuracy: 0.7500
Epoch 2/20
50/50 [==============================] - 22s 441ms/step - loss: 0.4012 - accuracy: 0.7538 - val_loss: 0.3514 - val_accuracy: 0.7500
Epoch 3/20
50/50 [==============================] - 22s 436ms/step - loss: 0.3728 - accuracy: 0.7519 - val_loss: 0.3345 - val_accuracy: 0.7500
Epoch 4/20
50/50 [==============================] - 23s 465ms/step - loss: 0.3877 - accuracy: 0.7663 - val_loss: 0.3527 - val_accuracy: 0.7812
Epoch 5/20
50/50 [==============================] - 23s 459ms/step - loss: 0.3618 - accuracy: 0.7613 - val_loss: 0.3435 - val_accuracy: 0.7837
Epoch 6/20
50/50 [==============================] - 22s 450ms/step - loss: 0.3451 - accuracy: 0.7894 - val_loss: 0.3643 - val_accuracy: 0.7513
Epoch 7/20
50/50 [==============================] - 22s 449ms/step - loss: 0.3609 - accuracy: 0.7788 - val_loss: 0.4067 - val_accuracy: 0.7312
Epoch 8/20
50/50 [==============================] - 23s 454ms/step - loss: 0.3487 - accuracy: 0.7788 - val_loss: 0.4295 - val_accuracy: 0.6950
Epoch 9/20
50/50 [==============================] - 22s 445ms/step - loss: 0.3540 - accuracy: 0.7775 - val_loss: 0.4022 - val_accuracy: 0.7212
Epoch 10/20
50/50 [==============================] - 20s 406ms/step - loss: 0.3391 - accuracy: 0.8019 - val_loss: 0.3239 - val_accuracy: 0.7550
Epoch 11/20
50/50 [==============================] - 20s 403ms/step - loss: 0.3739 - accuracy: 0.7731 - val_loss: 0.3492 - val_accuracy: 0.7500
Epoch 12/20
50/50 [==============================] - 21s 417ms/step - loss: 0.3513 - accuracy: 0.7750 - val_loss: 0.3320 - val_accuracy: 0.7550
Epoch 13/20
50/50 [==============================] - 21s 418ms/step - loss: 0.3486 - accuracy: 0.7850 - val_loss: 0.3304 - val_accuracy: 0.7500
Epoch 14/20
50/50 [==============================] - 21s 411ms/step - loss: 0.3489 - accuracy: 0.7750 - val_loss: 0.3404 - val_accuracy: 0.8625
Epoch 15/20
50/50 [==============================] - 21s 416ms/step - loss: 0.3350 - accuracy: 0.7937 - val_loss: 0.3188 - val_accuracy: 0.7675
Epoch 16/20
50/50 [==============================] - 21s 423ms/step - loss: 0.3363 - accuracy: 0.7919 - val_loss: 0.3212 - val_accuracy: 0.7550
Epoch 17/20
50/50 [==============================] - 20s 407ms/step - loss: 0.3366 - accuracy: 0.7962 - val_loss: 0.3210 - val_accuracy: 0.7500
Epoch 18/20
50/50 [==============================] - 21s 416ms/step - loss: 0.3483 - accuracy: 0.7987 - val_loss: 0.3435 - val_accuracy: 0.7500
Epoch 19/20
50/50 [==============================] - 21s 414ms/step - loss: 0.3593 - accuracy: 0.7638 - val_loss: 0.3256 - val_accuracy: 0.7500
Epoch 20/20
50/50 [==============================] - 21s 413ms/step - loss: 0.3331 - accuracy: 0.8056 - val_loss: 0.3429 - val_accuracy: 0.7500
'''
############################
# 5th 
'''
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
Step 1...
Found 1600 images belonging to 4 classes.
Found 800 images belonging to 4 classes.
Step 2...
Model: "sequential_9"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_20 (Conv2D)          (None, 98, 98, 32)        896       
                                                                 
 conv2d_21 (Conv2D)          (None, 96, 96, 64)        18496     
                                                                 
 conv2d_22 (Conv2D)          (None, 94, 94, 128)       73856     
                                                                 
 max_pooling2d_9 (MaxPoolin  (None, 47, 47, 128)       0         
 g2D)                                                            
                                                                 
 flatten_9 (Flatten)         (None, 282752)            0         
                                                                 
 dense_24 (Dense)            (None, 128)               36192384  
                                                                 
 dense_25 (Dense)            (None, 64)                8256      
                                                                 
 dropout_7 (Dropout)         (None, 64)                0         
                                                                 
 dense_26 (Dense)            (None, 4)                 260       
                                                                 
=================================================================
Total params: 36294148 (138.45 MB)
Trainable params: 36294148 (138.45 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
Epoch 1/20
50/50 [==============================] - 31s 606ms/step - loss: 0.9240 - accuracy: 0.5387 - val_loss: 0.4126 - val_accuracy: 0.7500
Epoch 2/20
50/50 [==============================] - 30s 602ms/step - loss: 0.4395 - accuracy: 0.7325 - val_loss: 0.3463 - val_accuracy: 0.7500
Epoch 3/20
50/50 [==============================] - 30s 597ms/step - loss: 0.3855 - accuracy: 0.7688 - val_loss: 0.6859 - val_accuracy: 0.5612
Epoch 4/20
50/50 [==============================] - 30s 597ms/step - loss: 0.4046 - accuracy: 0.7581 - val_loss: 0.6203 - val_accuracy: 0.6575
Epoch 5/20
50/50 [==============================] - 30s 598ms/step - loss: 0.3764 - accuracy: 0.7569 - val_loss: 0.3476 - val_accuracy: 0.7500
Epoch 6/20
50/50 [==============================] - 31s 625ms/step - loss: 0.5054 - accuracy: 0.7262 - val_loss: 0.3879 - val_accuracy: 0.7500
Epoch 7/20
50/50 [==============================] - 30s 604ms/step - loss: 0.4327 - accuracy: 0.7387 - val_loss: 0.3845 - val_accuracy: 0.7500
Epoch 8/20
50/50 [==============================] - 31s 620ms/step - loss: 0.3713 - accuracy: 0.7613 - val_loss: 0.3322 - val_accuracy: 0.7825
Epoch 9/20
50/50 [==============================] - 31s 616ms/step - loss: 0.3630 - accuracy: 0.7750 - val_loss: 0.3190 - val_accuracy: 0.9750
Epoch 10/20
50/50 [==============================] - 30s 605ms/step - loss: 0.3652 - accuracy: 0.7681 - val_loss: 0.3296 - val_accuracy: 0.7513
Epoch 11/20
50/50 [==============================] - 30s 597ms/step - loss: 0.4010 - accuracy: 0.7769 - val_loss: 0.4000 - val_accuracy: 0.7500
Epoch 12/20
50/50 [==============================] - 30s 594ms/step - loss: 0.3583 - accuracy: 0.7625 - val_loss: 0.3480 - val_accuracy: 0.7500
Epoch 13/20
50/50 [==============================] - 30s 595ms/step - loss: 0.3423 - accuracy: 0.8131 - val_loss: 0.3293 - val_accuracy: 0.7500
Epoch 14/20
50/50 [==============================] - 31s 621ms/step - loss: 0.3444 - accuracy: 0.7825 - val_loss: 0.3241 - val_accuracy: 0.7500
Epoch 15/20
50/50 [==============================] - 31s 615ms/step - loss: 0.3295 - accuracy: 0.8069 - val_loss: 0.4262 - val_accuracy: 0.7500
Epoch 16/20
50/50 [==============================] - 31s 620ms/step - loss: 0.3210 - accuracy: 0.8081 - val_loss: 0.2933 - val_accuracy: 0.8950
Epoch 17/20
50/50 [==============================] - 31s 612ms/step - loss: 0.3386 - accuracy: 0.8031 - val_loss: 0.3842 - val_accuracy: 0.7500
Epoch 18/20
50/50 [==============================] - 42s 850ms/step - loss: 0.3167 - accuracy: 0.8300 - val_loss: 0.3677 - val_accuracy: 0.7500
Epoch 19/20
50/50 [==============================] - 69s 1s/step - loss: 0.5171 - accuracy: 0.7638 - val_loss: 0.3294 - val_accuracy: 0.7500
Epoch 20/20
50/50 [==============================] - 30s 588ms/step - loss: 0.3394 - accuracy: 0.8025 - val_loss: 0.3673 - val_accuracy: 0.7500
'''