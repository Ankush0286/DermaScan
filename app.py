from flask import Flask, jsonify, request, render_template, send_from_directory, send_file
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from groq import Groq
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers, regularizers
from gradcam import grad_CAM  # Ensure gradcamshow is correctly imported

app = Flask(__name__)

# Class indices for predictions
class_names = [
    'Eczema',
    'Melanoma',
    'Psoriasis pictures Lichen Planus and related diseases',
    'Seborrheic Keratoses and other Benign Tumors',
    'Tinea Ringworm Candidiasis and other Fungal Infections',
    'Warts Molluscum and other Viral Infections',
    'Acne'
]
class_count = len(class_names)

# Global variable to store the latest prediction result
prediction_result = None



# Create uploads directory if it doesn't exist
if not os.path.exists('uploads'):
    os.makedirs('uploads')



# Load the model
def load_custom_model():
    model = tf.keras.applications.EfficientNetB2(
        include_top=False, weights=None, input_shape=(224, 224, 3), pooling='max'
    )
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

    model = Model(inputs=model.input, outputs=output)
    weights_path = 'model_weights.h5'
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
    else:
        raise FileNotFoundError(f"Model weights not found at {weights_path}")
    return model


# Load the model once at startup
model = load_custom_model()

# Prediction helper function
def make_prediction(img, model, input_shape=(224, 224)):
    try:
        img = img.resize(input_shape)  # Resize the image
        img_array = image.img_to_array(img)  # Convert to numpy array
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)  # Preprocess for EfficientNet

        predictions = model.predict(img_array)
        class_idx = np.argmax(predictions, axis=1)[0]
        confidence = predictions[0][class_idx]

        # Map class index to name
        class_label = class_names[class_idx]
        return class_label, confidence
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None


@app.route('/')
def index():
    """Serve the homepage from the templates folder."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload, make prediction, and store the result in the global variable."""
    global prediction_result  # Access the global variable
    try:
        # Get the uploaded image
        file = request.files['image']
        img = Image.open(file.stream)

        # Make prediction
        class_label, confidence = make_prediction(img, model)

        # Store the result in the global variable
        if class_label is not None:
            prediction_result = {
                'class': class_label,
                'confidence': round(float(confidence) * 100, 2)  # Convert to percentage and ensure JSON serialization
            }
            return jsonify(prediction_result)
        else:
            return jsonify({'error': 'Prediction failed'}), 500
    except Exception as e:
        return jsonify({'error': f'An error occurred: {e}'}), 500


@app.route('/remedy', methods=['POST'])
def get_remedy():
    global prediction_result

    # Check if a prediction exists before attempting to fetch a remedy
    if not prediction_result:
        return jsonify({'error': 'No prediction made yet'}), 400

    try:
        print(f"Prediction Result: {prediction_result}")  # Log the prediction result
        print(f"Requesting remedies for: {prediction_result['class']}")

        # Initialize the Groq API client
        client = Groq(api_key="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")
        completion = client.chat.completions.create(
            model="llama3-8b-8192",  # Check if the model is available
            messages=[  
                {
                    "role": "user",
                    "content": f"Suggest 5 scientifically backed home remedies for {prediction_result['class']} that can be safely used by patients."
                },
                {
                    "role": "assistant",
                    "content": "I'm happy to help!"
                }
            ],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=True
        )

        response_content = ""
        for chunk in completion:
            if chunk.choices[0].delta.content:
                response_content += chunk.choices[0].delta.content
            else:
                print(f"Unexpected chunk structure: {chunk}")  # Log any unexpected structure

        # Format the remedy into a list
        remedy_points = response_content.split("\n")
        formatted_remedy = [f"<li>{point.lstrip('*').strip()}</li>" for point in remedy_points if point.strip()]

        remedy_html = "<ul>" + "".join(formatted_remedy) + "</ul>"
        print(f"Remedy HTML: {remedy_html}")  # Log the remedy HTML response

        return jsonify({'remedy': remedy_html})

    except Exception as e:
        print(f"Error in /remedy route: {e}")  # Log the full error message
        return jsonify({'error': f'An error occurred: {e}'}), 500


@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        # Ensure that an image is uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Save the image temporarily
        img_path = os.path.join('uploads', file.filename)
        file.save(img_path)

        # Call Grad-CAM function to process the image
        grad_cam_result_path = grad_CAM(img_path)

        # Send the Grad-CAM result image as response
        return send_file(grad_cam_result_path, mimetype='image/jpeg')

    except Exception as e:
        return jsonify({'error': str(e)}), 500




@app.route('/get_result', methods=['GET'])
def get_result():
    """Return the latest prediction result stored in the global variable."""
    global prediction_result  # Access the global variable
    if prediction_result:
        return jsonify(prediction_result)
    else:
        return jsonify({'error': 'No prediction available'}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
