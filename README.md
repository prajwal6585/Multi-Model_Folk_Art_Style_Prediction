# Multi-Model_Folk_Art_Style_Prediction
A Flask-based web app for predicting Indian folk art styles using deep learning models like CNN, VGG16, ResNet50, and more. Upload an image to get predictions, model-wise confidence, and a comparison chart. Includes confusion matrix visualization and supports class mapping for 30+ Indian folk art categories.



https://github.com/user-attachments/assets/38a8f8d3-8642-4ae9-ab55-aea13e606f87



Environment Setup and Dependencies

Conda Environment Setup

Create a new Conda environment with Python 3.8:

    conda create --name mini python=3.8

Activate the environment:

    conda activate mini
    
Install the required Python packages using pip:

    pip install Flask tensorflow matplotlib numpy scikit-learn seaborn
    
Verify installed packages in the Conda environment:

    conda list



How to Run:

Clone or Download Project

Setup Environment

Run app.py


Note:

There are only 3 Models have uploaded. you can Train more models using the Train Files.

To train make use of Google Colab

The following major libraries are used in this project:

Flask: Web application framework

TensorFlow: Deep learning framework

NumPy: Numerical computing library

Matplotlib: Data visualization

Scikit-learn: Machine learning tools

Seaborn: Statistical data visualization

OpenCV: Image processing and computer vision


The project includes support for the following models:

CNN50

EfficientNet50

VGG16

MobileNet50

ResNet50

InceptionV3

DenseNet50

etc...
