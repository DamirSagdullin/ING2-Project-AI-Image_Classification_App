from fastapi import FastAPI, File, UploadFile
from fastapi.responses import RedirectResponse
from PIL import Image
from io import BytesIO
# Server for FastAPI application
import uvicorn
# Interface service
import gradio as gr
import numpy as np
import tensorflow as tf

# Define FastAPI application
app = FastAPI()

# Choose a model to use 
model = tf.keras.models.load_model('./my_model')
# Do not forget to use IMG_SIZE of chosen model
IMG_SIZE=224

# Define class names
class_names=['aluminium_foil', 'carton', 'chips_bag', 'drink_carton', 'glass_bottle', 'metal_bottle_cap', 'metal_can', 'paper', 'paper_cup', 'paper_tissues', 'plastic_bag', 'plastic_bottle', 'plastic_bottle_cap', 'plastic_container', 'plastic_cup', 'plastic_lid', 'plastic_straw', 'plastic_tableware', 'styrofoam']

# Gradio auxiliary function
def classify_image(img_path):
    img_PIL = tf.keras.utils.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.utils.img_to_array(img_PIL)
    img_batch = np.array([img_array])
    img_normalized = tf.keras.layers.Rescaling(1./255)(img_batch)
    predictions = model.predict(img_normalized)[0]
    confidences = {class_names[i]: predictions.tolist()[i] for i in range(len(class_names))}
    return confidences

# Redirect to predict UI
@app.get('/')
def redirect_to_predict_interface():
    return RedirectResponse('/ui/predict')

# Define API predict endpoint
@app.post('/api/predict')
async def predict_service(file: bytes = File()):
    img_bytes = BytesIO(file)
    img_PIL = tf.keras.utils.load_img(img_bytes, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.utils.img_to_array(img_PIL)
    img_batch = np.array([img_array])
    img_normalized = tf.keras.layers.Rescaling(1./255)(img_batch)
    predictions = model.predict(img_normalized)[0]
    confidences = {class_names[i]: predictions.tolist()[i] for i in range(len(class_names))}
    predicted_class = class_names[np.argmax(predictions)]
    return {'confidences': confidences, 'predicted_class': predicted_class, 'uncertain_result' : True if np.argmax(predictions) > 0.3 else False }

# Define Gradio app
io = gr.Interface(
    # Use auxiliary function to get confidences
    fn=classify_image,
    # Input type is filepath to temporary file
    inputs=gr.Image(type="filepath"),
    # Show top 5 classes
    outputs=gr.Label(num_top_classes=5),
    # Disable flagging
    allow_flagging="never")

# Add Gradio to FastAPI application 
app = gr.mount_gradio_app(app, io, path="/ui/predict")

# Run the application server
if __name__ == '__main__':
    uvicorn.run(app)