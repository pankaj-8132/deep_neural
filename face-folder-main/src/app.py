from flask import Flask, request, jsonify, render_template
from keras.models import load_model
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
model = load_model(os.path.join(os.getcwd(), 'trained_model.h5')) 

@app.route('/')
def home():
    return render_template('index.html')  

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        img = Image.open(file).convert('L')  
        img = img.resize((64, 64))  
        img_array = np.array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension

        prediction = model.predict(img_array)
        result = 'Fake' if prediction[0][0] > 0.5 else 'Real'
        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)


# import streamlit as st
# from keras.models import load_model
# import numpy as np
# from PIL import Image
# import os

# # Load the trained model
# MODEL_PATH = os.path.join(os.getcwd(), 'trained_model.h5')
# model = load_model(MODEL_PATH)

# # Streamlit app function
# def main():
#     st.title("Fake vs Real Face Detection")
#     st.write("Upload an image to check if it's Real or Fake.")

#     # File uploader
#     uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

#     if uploaded_file is not None:
#         try:
#             # Process the uploaded image
#             img = Image.open(uploaded_file).convert('L')  # Convert to grayscale
#             img = img.resize((64, 64))  # Resize to match the model input
#             img_array = np.array(img) / 255.0  # Normalize
#             img_array = np.expand_dims(img_array, axis=0)
#             img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension

#             # Predict using the model
#             prediction = model.predict(img_array)
#             result = 'Fake' if prediction[0][0] > 0.5 else 'Real'

#             # Display the results
#             st.image(img, caption="Uploaded Image", use_column_width=True)
#             st.write(f"**Prediction:** {result}")
#             st.write(f"**Confidence Score:** {prediction[0][0]:.2f}")

#         except Exception as e:
#             st.error(f"An error occurred: {e}")

# if __name__ == "__main__":
#     main()

