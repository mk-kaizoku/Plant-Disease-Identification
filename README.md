# Plant Disease Detection using Deep Learning

## Project Description
This project implements a **Convolutional Neural Network (CNN)** using **TensorFlow** and **Keras** to classify and detect diseases in plant leaves. The goal is to create an accurate model capable of assisting farmers and researchers in quickly identifying various plant diseases from leaf images.

The notebook demonstrates the full machine learning workflow, including:
* Dataset acquisition from Kaggle.
* Data preprocessing and augmentation.
* Building and training a custom CNN model.
* Evaluating the model's performance.
* Saving the final trained model.

## Dataset
The project utilizes the **PlantVillage Dataset**, which is downloaded directly from Kaggle within the notebook using the following command: `!kaggle datasets download -d abdallahalidev/plantvillage-dataset`.

### Dataset Characteristics
* **Source:** PlantVillage Dataset.
* **Data Used:** The model is trained on the **'color'** images from the dataset.
* **Classes:** The dataset contains **38** distinct classes, representing various plant species (e.g., Tomato, Potato, Apple) and their corresponding disease or healthy states (e.g., Apple___Cedar_apple_rust, Potato___healthy).
* **Image Format:** Color (RGB) images with a standard size of **256x256x3** pixels.

## Model and Technology

### Technology Stack
* **Language:** Python
* **Deep Learning Framework:** TensorFlow and Keras
* **Libraries:** `numpy`, `matplotlib`, `Pillow`, `ImageDataGenerator` (for augmentation)

### Model Architecture
The project uses a **Sequential CNN model** built with Keras. The architecture includes multiple:
1.  **Convolutional Layers (`Conv2D`):** For extracting features from the leaf images.
2.  **Pooling Layers (`MaxPooling2D`):** For reducing spatial dimensions and computational cost.
3.  **Flatten Layer:** To convert the 2D feature maps into a 1D vector.
4.  **Dense Layers:** For classification, ending with an output layer that has 38 units (one for each class) and a **softmax** activation function.

### Key Steps
* **Data Augmentation:** The `ImageDataGenerator` is used to augment the training data, helping the model generalize better and prevent overfitting. This includes rescaling, rotation, and other transformations.
* **Model Training:** The model is compiled using the **Adam optimizer** and **categorical cross-entropy** loss, and trained for **20 epochs**.
* **Model Saving:** The trained model is saved as `plant_disease_prediction_model_color.h5`.

## Requirements

To run this notebook, you will need:

1.  **Python** (3.6+)
2.  **Jupyter Notebook** or **Google Colab** 
3.  **Required Python Libraries:**
    ```bash
    pip install tensorflow numpy matplotlib pillow kaggle
    ```

## Usage and Setup

1.  **Kaggle Setup:** To download the dataset automatically, you need to set up your Kaggle API credentials.

2.  **Run the Notebook:** Open and run all cells in `plant_disease_detection.ipynb`. The notebook will:
    * Load libraries and set random seeds for reproducibility.
    * Download and extract the PlantVillage Dataset.
    * Set up data generators for training and validation.
    * Define, compile, and train the CNN model.
    * Evaluate the model and save the final `.h5` model file.

3.  **Making Predictions:** The final sections of the notebook demonstrate how to load the saved model and make a prediction on a new image.
