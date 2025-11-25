import streamlit as st
import time as t
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix, classification_report
import keras
from keras.activations import relu, linear
from keras.layers import Dense, Dropout,Input
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping,ReduceLROnPlateau, TensorBoard
from keras.regularizers import L1, L2
from keras.layers import Conv2D, MaxPool2D, Flatten
import cv2, os
import datetime
from datetime import datetime
import joblib
from openpyxl import load_workbook

st.set_page_config(
    page_title="Arunava's Streamlit",
    page_icon="𓆩♛𓆪"
)

# ✅Initialize session state for login
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
# # ✅Initialize session state for User Name
# if "user_name" not in st.session_state:
#     st.session_state.user_name = "" 
    
st.write('# :rainbow[Vehicle Classification]🔥')
st.info("Vehicle classification is a computer vision task that uses machine learning to automatically identify and categorize vehicles\
             like cars, trucks,  bikes from images.")  

# Inject custom CSS for gradient sidebar
st.markdown("""
    <style>
        [data-testid="stSidebar"] > div:first-child {
            background-image: linear-gradient(to bottom right, #3F0112, #0E1117);
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

if not st.session_state.logged_in:
    st.markdown("""
    <h5><span style='color: orange; font-size: 36px;'>🔐Login</span></h5>
    """, unsafe_allow_html=True)
    email = st.text_input("Email Id")
    pw = st.text_input("Password", type='password')
    # ✅ Allowed usernames
    allowed_emails = ['arunava.samanta@capgemini.com', 'arunavasamanta001@gmail.com', 'saptarshi.jana@capgemini.com', 'shubham.g.sharma@capgemini.com']
    if email and pw:
        if st.button("Submit"):
            if email.lower() in allowed_emails and pw == '1234':
                st.session_state.logged_in = True  # ✅ Set login status
                st.success("✅Logged in successfully. Wait a second...")
                t.sleep(1)
                st.rerun()
            else: 
                st.error("❌Wrong credentials. Try again...")
    else:
        st.button("Submit", disabled=True)

else:
    # ✅ After login → Show sidebar and pages
    st.sidebar.image("https://github.com/IamArunavaSamanta/CNN-Streamlit-Vehicles-Classification-Project/blob/main/images/logo.png?raw=true", width=100)
    menu = st.sidebar.radio("",["🏠Home", "💻Prediction", "📍Fun Quiz", "⭐Feedback"])
#--------------------------------------------------------------------------------------------------------------------------
    
    if menu == "🏠Home":
        st.image("https://github.com/IamArunavaSamanta/CNN-Streamlit-Vehicles-Classification-Project/blob/main/images/Group_Vehicles.png?raw=true", width=450)
        st.markdown('''#### :red-background[:orange[CNN]]:orange[, or Convolutional Neural Network,] ''')
        
        st.markdown("""
        It is a type of deep learning model designed to automatically and adaptively learn features from input images. Convolution is a mathematical operation that combines two functions to produce a third function.
        It uses small matrices called filters (kernels) that slide across the input, computing dot products to detect specific features such as edges, textures, and shapes.
        CNNs are trained on labeled datasets and can achieve high accuracy in recognizing vehicle types even under varying conditions like lighting, angle, and background.
        
        
        ##### 🔍 How CNN Works:
        - **Convolution Layers**: Extract features from the image using filters.
        - **Pooling Layers**: Reduce the spatial dimensions, making the model efficient.
        - **Fully Connected Layers**: Interpret the extracted features to classify the image.
        - **Activation Functions**: Introduce non-linearity to learn complex patterns.
        
        ##### 🚘 Applications:
        - Classifying vehicle types (e.g., car, truck, bike)
        - Traffic monitoring systems
        - Autonomous driving
        - Smart parking solutions

         ##### 📒 Transfer Learning:
        - Transfer learning is a technique where a model trained on a large dataset for one task is reused for a different but related task. Instead of training from scratch, we leverage pre-trained models to save time and improve accuracy, especially when our dataset is small.
        - We used VGGNet (specifically VGG16) as the base model for transfer learning.
        - It is a deep convolutional neural network architecture developed by Visual Geometry Group (VGG) at Oxford.
        - Pre-trained weights on ImageNet are widely available, making it ideal for transfer learning.

        
        """)
    
        st.markdown(
            "---🔗Developed by [Arunava Samanta](https://www.linkedin.com/in/arunava-samanta-7439071ba/)"        
        )
        st.markdown(
         "---🔗Additional Learning [CNN](https://www.geeksforgeeks.org/deep-learning/convolutional-neural-network-cnn-in-machine-learning/)"
        )
        st.markdown(
            "---🔗Additional Learning [VGG-Net](https://www.geeksforgeeks.org/computer-vision/vgg-net-architecture-explained/)"
        )
            
#-------------------------------------------------------------------------------------------------------------------
    elif menu =='💻Prediction':
        if st.session_state.logged_in:
            st.success("Welcome to Prediction Page!")
    
            st.markdown("##### Upload a vehicle image 👇")
            uploaded_file = st.file_uploader("Choose a JPG file", type=["jpg"])
        
                #Function
            class ModelWrapper():
                def __init__(self, model, encoder):
                    self.model = model
                    self.encoder = encoder
                    
                def single_img_read(self, file_obj):
                    from PIL import Image
                    IMAGE_INPUT_SIZE = 150
                
                    # Read image from file-like object using PIL, then convert to NumPy array
                    img = Image.open(file_obj).convert("RGB")  # Convert to grayscale
                    img = img.resize((IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE))
                    img = np.array(img)
                    # Reshape for model input
                    img = img.reshape(1, IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE, 3)
                    # Predict
                    z = self.model.predict(img)
                    index = np.argmax(z)
                    Predicted_accuracy = z[0][index]*100
                    if Predicted_accuracy<60:
                        predicted_label = str("Unknown")
                    else:
                        predicted_label = self.encoder.inverse_transform([index])[0]
                    
                    # Bar chart of probabilities
                    category_labels = self.encoder.classes_  # Assuming encoder has all class labels
                    percentages = z[0] * 100
                    fig, ax = plt.subplots(figsize=(7, 2))
                    fig.patch.set_facecolor((0, 0, 0, 0.6))  #Set figure background to black
                    ax.set_facecolor((0, 0, 0, 0.2))  
                    # Define 7 distinct colors
                    c = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta']
                    ax.bar(category_labels, percentages, color=c, edgecolor='white')
                    ax.set_xlabel('Categories', color='white')
                    ax.set_ylabel('Probability (%)', color='white')
                    ax.set_ylim(0, 100)
                    plt.xticks(rotation=35, color='white')
                    
                    # Set tick labels to white
                    ax.tick_params(axis='y', colors='white')
                    # Set axis lines (spines) to white
                    for spine in ax.spines.values():
                        spine.set_color('white')
                    st.pyplot(fig)
                    
                    return Predicted_accuracy, predicted_label
        
            savedModel = joblib.load("models/NewVehicleModel.pkl")
            
            if uploaded_file is not None:
                st.image(uploaded_file, caption="Uploaded Image", width=300)
            
                # Run prediction only after file is uploaded
                accuracy, label = savedModel.single_img_read(uploaded_file)
            
                st.markdown(f"### 🤖 Predicted Class: `{label}`")
                st.markdown(f"### 📊 Prediction Accuracy(%): `{accuracy:.3f}`")
                st.success("✅ Prediction completed!")
            else:
                st.info("Please upload a vehicle image to see predictions.")
#--------------------------------------------------------------------------------------------------------        
            st.text("🗒 Examples")    
            # Image paths or URLs
            images = ["https://github.com/IamArunavaSamanta/CNN-Streamlit-Vehicles-Classification-Project/blob/main/images/Bike.jpg?raw=true", 
                      "https://github.com/IamArunavaSamanta/CNN-Streamlit-Vehicles-Classification-Project/blob/main/images/Car.jpg?raw=true",
                      "https://github.com/IamArunavaSamanta/CNN-Streamlit-Vehicles-Classification-Project/blob/main/images/Big%20Truck.jpg?raw=true"]
            
            # Create 3 columns
            col1, col2, col3 = st.columns(3)
            
            # Set a fixed width (e.g., 200px)
            image_width = 250
            
            # Display images
            col1.image(images[0], width=image_width)
            col1.success("🛵 Predicted Class: Bikes")
            col1.success("📊 Prediction Accuracy(%): 99.1")
            col2.image(images[1], width=image_width)
            col2.success("🚗 Predicted Class: Cars")
            col2.success("📊 Prediction Accuracy(%): 98.1")
            col3.image(images[2], width=image_width)
            col3.success("🚛 Predicted Class: Truck")
            col3.success("📊 Prediction Accuracy(%): 96.8")
        else:
            st.warning("Please login first to access this page.")
#--------------------------------------------------------------------------------------------------------------------
    elif menu == '📍Fun Quiz':
        st.text("1. Which of the following is a type of Machine Learning?")
        ans = st.radio("Choose any one",  ['A. Supervised', 'B. Unsupervised', 'C. Reinforcement', 'D. All of the above'], index=None)
        if ans is None:
            st.warning("⚠️ Select an option")
        elif ans == 'D. All of the above':
            st.success("✅ Correct!")
        else:
            st.error("❌ Try again")
    #---------------------------------------------------------------------------------------------------        
        st.text("2. Which component of a neural network adjusts weights during training?")
        ans = st.radio("Choose any one",  ['A. Loss function', 'B. Optimizer', 'C. Activation Function', 'D. Bias'], index=None)
        if ans is None:
            st.warning("⚠️ Select an option")
        elif ans == 'B. Optimizer':
            st.success("✅ Correct!")
        else:
            st.error("❌ Try again")
    #------------------------------------------------------------------------------------------------------        
        st.text("3.  What is the full name of the OpenCV?")
        ans = st.radio("Choose any one",  ['A. Open Computer Vector', 'B. Open Computer Vision', 'C. Open Common Vector', 'D.  Open Common Vision'], index=None)
        if ans is None:
            st.warning("⚠️ Select an option")
        elif ans == 'B. Open Computer Vision':
            st.success("✅ Correct!")
        else:
            st.error("❌ Try again")
        if st.button("Submit"):
            st.balloons()
#-------------------------------------------------------------------------------------------------------------
    elif menu == "⭐Feedback":
        st.markdown("""
            <style>  
                .stats-number {
                    font-size: 1.5rem;
                    font-weight: 500;
                    color: #FFD700;
                    margin-bottom: 0.5rem;
                }
                
                /* Feature icons */
                .feature-icon {
                    font-size: 2rem;
                    margin-bottom: 1rem;
                }
                
                /* Stats cards */
                .stats-card {
                    background: linear-gradient(to bottom right, #3F0112, #0E1117);
                    border-radius: 20px;
                    padding: 1.5rem;
                    text-align: center;
                    margin: 0.5rem;
                }
            </style>
        """, unsafe_allow_html=True)
    #---------------------------------------------------------------------------------------------------------------------    
        st.markdown("##### 📈 Model Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
                <div class='stats-card'>
                    <div class='feature-icon'>🎯</div>
                    <div class='stats-number'>96.5%</div>
                    <div class='stats-label'>Accuracy</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class='stats-card'>
                    <div class='feature-icon'>⚡</div>
                    <div class='stats-number'>2.3s</div>
                    <div class='stats-label'>Analysis Time</div>
                </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
                <div class='stats-card'>
                    <div class='feature-icon'>✉️</div>
                    <div class='stats-number'>6500+</div>
                    <div class='stats-label'>Trained Img</div>
                </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown("""
                <div class='stats-card'>
                    <div class='feature-icon'>🔎</div>
                    <div class='stats-number'>7</div>
                    <div class='stats-label'>Total Class</div>
                </div>
            """, unsafe_allow_html=True)
    #--------------------------------------------------------------------------------------------------------------    
        st.text("")
        st.markdown("##### ⭐ Feedback")
        rating = st.radio('Rating',['Very Bad', 'Bad', 'Average', 'Good', 'Very Good', 'Excellent'], index=None, horizontal=True)
        feedback_comment = st.text_area("Additional comments (optional):")
        if st.button("Submit Feedback"):
            if rating is not None:
                st.success("Thank you for your feedback! 🙏")
            else:
                st.warning("Please select rating! 😊")
        if st.button("🚪Logout"):
            st.session_state.logged_in = False
            st.rerun()
        


















