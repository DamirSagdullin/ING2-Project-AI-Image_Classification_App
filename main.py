from fastapi import FastAPI, File, UploadFile
from fastapi.responses import RedirectResponse
import uvicorn
import gradio as gr

from PIL import Image
from io import BytesIO

import numpy as np
import tensorflow as tf

app = FastAPI()

IMG_SIZE=224
class_names=['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
normalization_layer = tf.keras.layers.Rescaling(1./255)
model = tf.keras.models.load_model('./my_model')

@app.get('/')
def redirect_to_predict_interface():
    return RedirectResponse('/ui/predict')

@app.post('/api/predict')
async def predict_service(file: bytes = File()):
    image = BytesIO(file)
    pil_image = tf.keras.utils.load_img(image, target_size=(IMG_SIZE, IMG_SIZE))
    input_arr = tf.keras.utils.img_to_array(pil_image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    input_arr = normalization_layer(input_arr)
    predictions = model.predict(input_arr)
    predictions = predictions[0]
    confidences = {class_names[i]: predictions.tolist()[i] for i in range(len(class_names))}
    predicted_class = class_names[np.argmax(predictions)]
    return {'confidences': confidences, 'predicted_class': predicted_class}

def classify_image(image):
    input_arr = image.reshape((1, IMG_SIZE, IMG_SIZE, 3))
    input_arr = normalization_layer(input_arr)
    predictions = model.predict(input_arr)
    predictions = predictions[0]
    confidences = {class_names[i]: predictions.tolist()[i] for i in range(len(class_names))}
    return confidences

io = gr.Interface(
            fn=classify_image, 
            inputs=gr.Image(shape=(IMG_SIZE, IMG_SIZE)),
            outputs=gr.Label(num_top_classes=3),
            allow_flagging="never")
app = gr.mount_gradio_app(app, io, path="/ui/predict")

if __name__ == '__main__':
    uvicorn.run(app)