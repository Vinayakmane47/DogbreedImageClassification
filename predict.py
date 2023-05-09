from keras.models import load_model


classes = {'french_bulldog': 0,
 'german_shepherd': 1,
 'golden_retriever': 2,
 'poodle': 3,
 'yorkshire_terrier': 4}

# Load the saved model from disk
model = load_model('model/best_model.h5')

# Use the loaded model for prediction, evaluation, or further training
import numpy as np
from keras.preprocessing import image

# Load the image you want to predict
img_path = 'image/image.jpg'
img = image.load_img(img_path, target_size=(256, 256))

# Convert the image to a numpy array and preprocess it
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = x / 255.

# Make the prediction using the trained model
preds = model.predict(x)
class_idx = np.argmax(preds[0])
predicted_class = [k for k, v in classes.items() if v == class_idx][0]

# Print the predicted class
print("The predicted class is:", predicted_class)

