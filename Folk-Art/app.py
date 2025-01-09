import os
from flask import Flask, request, render_template, redirect
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import joblib  # Import joblib for loading the decision tree model
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Initialize the Flask app
app = Flask(__name__)

# Define the paths for the models and static folder
MODELS_PATHS = {
    'CNN': 'model/indian_folk_art_cnn_model_update50.h5',
    'Hybridcnn': 'model/hybrid_cnn_folk_art_model.h5',
    'Deepcnn': 'model/deep_cnn_folk_art_model.h5',
    'Inception': 'model/inception_folk_art_model50.h5',
    'VGG16': 'model/indian_folk_art_vgg16_model50.h5',
    'ResNet50': 'model/resnet_folk_art_model50.h5',
    'Nasnet': 'model/nasnet_folk_art_model.h5',
    'Mobilenet': 'model/mobilenet_folk_art_model50.h5',
    'Efficientnet': 'model/efficientnet_folk_art_model50.h5',
    'Densnet': 'model/densenet_folk_art_model50.h5',
}
UPLOAD_FOLDER = 'static/uploads/'

# Load your trained models
cnn_model = load_model(MODELS_PATHS['CNN'])
hybridcnn_model = load_model(MODELS_PATHS['Hybridcnn'])
deepcnn_model = load_model(MODELS_PATHS['Deepcnn'])
inception_model = load_model(MODELS_PATHS['Inception'])
vgg16_model = load_model(MODELS_PATHS['VGG16'])
resnet50_model = load_model(MODELS_PATHS['ResNet50'])
nasnet_model = load_model(MODELS_PATHS['Nasnet'])
mobilenet_model = load_model(MODELS_PATHS['Mobilenet'])
efficientnet_model = load_model(MODELS_PATHS['Efficientnet'])
densnet_model = load_model(MODELS_PATHS['Densnet'])

# Image size must match the size your model was trained on
IMG_SIZE = 128

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Define class labels manually
class_labels = [
    'Aipan Art (Uttarakhand)', 'Assamese Miniature Painting (Assam)', 'Basholi Painting (Jammu and Kashmir)',
    'Bhil Painting (Madhya Pradesh)', 'Chamba Rumal (Himachal Pradesh)', 'Cheriyal Scroll Painting (Telangana)',
    'Dokra Art (West Bengal)', 'Gond Painting (Madhya Pradesh)', 'Kalamkari Painting (Andhra Pradesh and Telangana)',
    'Kalighat Painting (West Bengal)', 'Kangra Painting (Himachal Pradesh)', 'Kerala Mural Painting (Kerala)',
    'Kondapalli Bommallu (Andhra Pradesh)', 'Kutch Lippan Art (Gujarat)', 'Leather Puppet Art (Andhra Pradesh)',
    'Madhubani Painting (Bihar)', 'Mandala Art', 'Mandana Art (Rajasthan)', 'Mata Ni Pachedi (Gujarat)',
    'Meenakari Painting (Rajasthan)', 'Mughal Paintings', 'Mysore Ganjifa Art (Karnataka)',
    'Pattachitra Painting (Odisha and Bengal)', 'Patua Painting (West Bengal)', 'Pichwai Painting (Rajasthan)',
    'Rajasthani Miniature Painting (Rajasthan)', 'Rogan Art from Kutch (Gujarat)', 'Sohrai Art (Jharkhand)',
    'Tikuli Art (Bihar)', 'Warli Folk Painting (Maharashtra)'
]

# Preprocess the image for prediction
def preprocess_image(image_path):
    image = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image /= 255.0  # Normalize image
    return image

# Mapping numerical labels to class names using the class_labels list
def map_label_to_class(index):
    return class_labels[index]

# Generate individual accuracy chart for each model
def generate_individual_accuracy_chart(model_name, prediction):
    confidence = np.max(prediction[0]) * 100  # Ensure we are accessing the first dimension of the prediction
    plt.figure(figsize=(6, 4))
    plt.bar([model_name], [confidence], color='#3498db')
    plt.ylim(0, 100)
    plt.ylabel("Confidence (%)")
    plt.title(f"{model_name} Prediction Confidence")
    
    # Save to an image buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    return img_base64

# Generate accuracy charts for models
def generate_accuracy_chart(predictions, model_names):
    plt.figure(figsize=(10, 6))
    for i, (model_name, pred) in enumerate(predictions.items()):
        plt.bar(i, max(pred) * 100, label=model_name)
    plt.ylabel("Prediction Confidence (%)")
    plt.xticks(range(len(predictions)), model_names)
    plt.title("Prediction Confidence of Different Models")
    plt.legend()
    
    # Save to an image buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    return img_base64

# Generate confusion matrix image
def generate_confusion_matrix_image(actual_classes, predicted_classes):
    cm = confusion_matrix(actual_classes, predicted_classes, labels=range(len(class_labels)))

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')

    # Save to an image buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    return img_base64

# Define the route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Define the route to handle image uploads and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    # Save the uploaded file to the static/uploads folder
    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Preprocess the image
        processed_image = preprocess_image(file_path)

        # Perform prediction using the loaded models
        predictions = {
            'CNN': cnn_model.predict(processed_image),
            'HybridCNN': hybridcnn_model.predict(processed_image),
            'DeepCNN': deepcnn_model.predict(processed_image),
            'Inception': inception_model.predict(processed_image),
            'VGG16': vgg16_model.predict(processed_image),
            'ResNet50': resnet50_model.predict(processed_image),
            'Nasnet': nasnet_model.predict(processed_image),
            'Mobilenet': mobilenet_model.predict(processed_image),
            'Efficientnet': efficientnet_model.predict(processed_image),
            'Densenet': densnet_model.predict(processed_image),
        }

        results = {}
        individual_charts = {}
        
        # Placeholder: Set this to the true label index of your uploaded image
        actual_class = 0  
        predicted_classes = []

        for model_name, prediction in predictions.items():
            predicted_class = np.argmax(prediction, axis=1)[0]  # For other models
            confidence = max(prediction[0])  # Get the maximum confidence score
            
            predicted_classes.append(predicted_class)  # Collecting predicted class index
            results[model_name] = {
                'label': map_label_to_class(predicted_class),
                'confidence': confidence * 100  # Convert confidence to percentage
            }
            # Generate individual accuracy chart for each model
            individual_charts[model_name] = generate_individual_accuracy_chart(model_name, prediction)

        # Generate chart for prediction accuracy
        accuracy_chart = generate_accuracy_chart(predictions, results.keys())

        # Generate confusion matrix image with consistent lengths
        cm_image = generate_confusion_matrix_image([actual_class] * len(predicted_classes), predicted_classes)

        # Final prediction (based on the model with the highest confidence score)
        final_model = max(results, key=lambda x: results[x]['confidence'])
        final_label = results[final_model]['label']
        final_confidence = results[final_model]['confidence']

        # Render the result template with predictions and charts
        return render_template('result.html', results=results, image_url=file_path, 
                               accuracy_chart=accuracy_chart, individual_charts=individual_charts,
                               cm_image=cm_image, final_label=final_label, final_confidence=final_confidence)  # Include final prediction

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
