import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras import layers, regularizers, Model
from tensorflow.keras.preprocessing import image
import os
import tempfile

def grad_CAM(original_img_path):
    def load_custom_model():
        # Load the model architecture
        model = tf.keras.applications.EfficientNetB2(
            include_top=False, weights=None, input_shape=(224, 224, 3), pooling='max'
        )

        # Add custom layers
        x = model.output
        x = layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
        x = layers.Dense(256, kernel_regularizer=regularizers.l2(0.016), 
                         activity_regularizer=regularizers.l1(0.006),
                         bias_regularizer=regularizers.l1(0.006), activation='relu')(x)
        x = layers.Dense(128, kernel_regularizer=regularizers.l2(0.016), 
                         activity_regularizer=regularizers.l1(0.006),
                         bias_regularizer=regularizers.l1(0.006), activation='relu')(x)
        x = layers.Dropout(rate=0.45, seed=42)(x)
        output = layers.Dense(class_count, activation='softmax')(x)

        # Complete the model
        model = Model(inputs=model.input, outputs=output)
        
        # Load pretrained weights (ensure path is correct)
        weights_path = 'model_weights.h5'  # Replace with actual weights file path
        if os.path.exists(weights_path):
            model.load_weights(weights_path)
        else:
            raise FileNotFoundError(f"Model weights not found at {weights_path}")
        
        return model
    
    def get_grad_cam(model, img_array, class_index, layer_name="top_conv"):
        # 1. Get the last convolutional layer
        try:
            last_conv_layer = model.get_layer(layer_name)
        except ValueError:
            raise ValueError(f"Layer {layer_name} not found in the model. Check the layer name.")
        
        # 2. Create a new model for Grad-CAM
        grad_model = Model(
            inputs=model.input,
            outputs=[last_conv_layer.output, model.output]
        )
        
        # 3. Compute gradients
        with tf.GradientTape() as tape:
            conv_output, predictions = grad_model(img_array)
            class_output = predictions[:, class_index]
        
        grads = tape.gradient(class_output, conv_output)
        
        # 4. Pool gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # 5. Weighted sum of feature maps
        conv_output = conv_output[0]
        cam = conv_output @ pooled_grads[..., tf.newaxis]
        cam = cam.numpy().squeeze()
        
        # 6. ReLU and normalize
        cam = np.maximum(cam, 0)
        if np.max(cam) != 0:
            cam = cam / np.max(cam)
        
        return cam
    
    def preprocess_image(image_path, target_size=(224, 224)):
        try:
            img = image.load_img(image_path, target_size=target_size)
        except Exception as e:
            raise ValueError(f"Error loading image: {str(e)}")
        
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
        return img_array
    
    def plot_grad_cam(image_path, model, class_index):
        img_array = preprocess_image(image_path)
        
        # Get the Grad-CAM heatmap
        heatmap = get_grad_cam(model, img_array, class_index)
        
        # Load the original image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image at {image_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize the heatmap to match the image size
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        
        # Convert the heatmap to RGB (jet colormap)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Superimpose the heatmap on the image
        superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
        
        # Save the result to a temporary file and return the path
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            output_path = tmp_file.name
            cv2.imwrite(output_path, superimposed_img)
        
        # Debugging: print the output path
        print(f"Grad-CAM image saved at: {output_path}")
        
        return output_path

    def predict_class(image_path, model):
        img_array = preprocess_image(image_path)
        predictions = model.predict(img_array)
        class_index = np.argmax(predictions[0])
        return class_index

    # Define the number of classes
    class_count = 7

    # Load the model
    model = load_custom_model()

    # Test Grad-CAM
    class_index = predict_class(original_img_path, model)
    print(f"Predicted class index: {class_index}")
    
    grad_cam_output_path = plot_grad_cam(original_img_path, model, class_index)
    
    return grad_cam_output_path
