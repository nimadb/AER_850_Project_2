import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image  # Import Image from PIL library

class_labels = {
    0: 'Large',
    1: 'Medium',
    2: 'Small',
    3: 'None'
}

# Load the saved model
model = load_model('model_1.h5')

# Load the image
image_path = '/Users/nima.db/Documents/Fall 2023/AER850/Project 2/AER_850_Project_2/Data/Test/Medium/Crack__20180419_06_19_09,915.bmp' 
img = Image.open(image_path)
img = img.resize((100, 100))  # Resize the image to match the model's input size

# Convert the image to RGB if it's not in RGB mode
img = img.convert('RGB')

# Normalize the image
img = np.array(img) / 255.0

# Add an extra dimension to create a batch of size 1 (as the model expects)
img = np.expand_dims(img, axis=0)

# Make predictions
predictions = model.predict(img)

# Display predicted probabilities for each class
print("Predicted Probabilities:")
for i, prob in enumerate(predictions[0]):
    print(f"Class {i + 1}: Probability {prob:.4f}")
    
# Open the bitmap file
bitmap = Image.open('/Users/nima.db/Documents/Fall 2023/AER850/Project 2/AER_850_Project_2/Data/Test/Medium/Crack__20180419_06_19_09,915.bmp')

# Convert the bitmap image to a numpy array with RGB mode
bitmap_rgb = bitmap.convert('RGB')
bitmap_array = bitmap_rgb.copy()  # Make a copy to avoid modifying the original image

# Plot the bitmap using Matplotlib with the correct colormap
plt.figure()
plt.imshow(bitmap_array)
plt.axis('off')  # Turn off axis labels and ticks
plt.title('Bitmap Image')

# Annotate predicted probabilities on the image
for i, prob in enumerate(predictions[0]):
    label = class_labels[i]
    plt.annotate(f"{label.capitalize()}: {prob:.4f}", (10, 30 + i * 100), color='white')

plt.title('Medium Crack')
plt.show()


# Load the image
image_path1 = '/Users/nima.db/Documents/Fall 2023/AER850/Project 2/AER_850_Project_2/Data/Test/Large/Crack__20180419_13_29_14,846.bmp' 
img1 = Image.open(image_path1)
img1 = img1.resize((100, 100))  # Resize the image to match the model's input size

# Convert the image to RGB if it's not in RGB mode
img1 = img1.convert('RGB')

# Normalize the image
img1 = np.array(img1) / 255.0

# Add an extra dimension to create a batch of size 1 (as the model expects)
img1 = np.expand_dims(img1, axis=0)

# Make predictions
predictions1 = model.predict(img1)

# Display predicted probabilities for each class
print("Predicted Probabilities:")
for i, prob in enumerate(predictions1[0]):
    print(f"Class {i + 1}: Probability {prob:.4f}")
    
# Open the bitmap file
bitmap1 = Image.open('/Users/nima.db/Documents/Fall 2023/AER850/Project 2/AER_850_Project_2/Data/Test/Large/Crack__20180419_13_29_14,846.bmp')

# Convert the bitmap image to a numpy array with RGB mode
bitmap_rgb1 = bitmap1.convert('RGB')
bitmap_array1 = bitmap_rgb1.copy()  # Make a copy to avoid modifying the original image

# Plot the bitmap using Matplotlib with the correct colormap
plt.figure()
plt.imshow(bitmap_array1)
plt.axis('off')  # Turn off axis labels and ticks
plt.title('Bitmap Image')

# Annotate predicted probabilities on the image
for i, prob in enumerate(predictions1[0]):
    label1 = class_labels[i]
    plt.annotate(f"{label1.capitalize()}: {prob:.4f}", (10, 30 + i * 100), color='white')
    
plt.title('Large Crack')
plt.show()