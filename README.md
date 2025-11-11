🥗 Food Image Classification using TensorFlow and OpenCV
📘 Project Overview

This project aims to classify different food categories using Image Processing and Deep Learning techniques.
With the help of TensorFlow and OpenCV, the model automatically detects and predicts the type of food from an image, showcasing the power of Artificial Intelligence in visual recognition tasks.

🧠 Model Description

The model is a Convolutional Neural Network (CNN) built and trained using TensorFlow/Keras.
It processes images through multiple layers of convolution, pooling, and fully connected layers to classify them into predefined food categories.

🔹 Key Features

Implemented image preprocessing using OpenCV (resizing, normalization, color conversion).

Built a CNN architecture from scratch using TensorFlow.

Achieved efficient feature extraction and accurate predictions.

Integrated model prediction visualization for better understanding.

⚙️ Technologies Used
Category	Tools / Libraries
Programming Language	Python
Deep Learning	TensorFlow, Keras
Image Processing	OpenCV
Data Handling	NumPy, Pandas
Visualization	Matplotlib, Seaborn
🧩 Dataset

A labeled food image dataset containing multiple classes such as:

Fruits 🍎

Fried Food 🍟

Vegetables 🥦

Grains 🍚

Snacks 🍪

(Dataset source: Custom dataset prepared for training and testing purposes)

🧮 Model Architecture
Layer	Type	Activation	Purpose
1	Convolution2D	ReLU	Extract image features
2	MaxPooling2D	—	Reduce dimensionality
3	Flatten	—	Convert features to 1D
4	Dense (Fully Connected)	ReLU	Learn nonlinear combinations
5	Output Layer	Softmax	Multi-class classification

Optimizer: Adam
Loss Function: Categorical Crossentropy
Evaluation Metric: Accuracy

🧾 How It Works

Load and preprocess the dataset using OpenCV.

Train the CNN model with TensorFlow/Keras.

Evaluate accuracy and visualize performance.

Predict the class of a new image using the trained model.

💻 Sample Output
Input Image	Predicted Category
🍔 Burger Image	Fast Food
🍎 Apple Image	Fruit
🍚 Rice Image	Grain
🥦 Broccoli Image	Vegetable

Example Output:

Predicted Category: Fried Food
Accuracy: 85%

🚀 Future Enhancements

Deploy the model as a web app using Flask or Streamlit.

Integrate real-time camera input for instant classification.

Use Transfer Learning (VGG16, MobileNet) for higher accuracy.

Add calorie estimation based on classified food type.

🧾 How to Run the Project
# Clone this repository
git clone https://github.com/<your-username>/Food_Classification_Project.git

# Navigate to the folder
cd Food_Classification_Project

# Install required dependencies
pip install -r requirements.txt

# Train the model
python model_training.py

# Predict on a new image
python predict.py

📂 Project Structure
Food_Classification_Project/
│
├── model_training.py           # Model building and training script
├── predict.py                  # Script for testing new images
├── requirements.txt            # Dependencies
├── README.md                   # Project documentation
├── images/                     # Sample prediction results
└── .gitignore                  # Ignored files and folders

👩‍💻 Author

Rayikanti Sushmitha
B.Tech – Computer Science and Engineering (AIML)
📍 India
🔗 LinkedIn
