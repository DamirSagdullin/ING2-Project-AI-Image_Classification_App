# ING2 2022-2023 | AI project

This repository contains a simple FastApi application with an endpoint and a Gradio UI to classify trash image to one of 19 classes. The classification models are based on transfer learing using efficientnet_v2 for "larger" model and mobilenet_v3 for "smaller" one.

! larger model is unavailable because of file size limit

## 0. Dev_notes documents

- "Neovision Test Cases" : tested models and their parameters with key results
- "Neovision Test Screens" : graphs of val_loss and val_acc during training of tested models
- "Neovision_Model_Build_Script" : script used to perform tests in Google Colab environment

## 1. Classic version

### 1.1 Installation :

- Goto app/
    ```bash
    cd app
    ```

- Install dependencies :
    ```bash
    pip install -r requirements.txt
    ```
 
- Configure model to use in main.py :
   
    Better performance (68%) but slower inference (900ms):
    ```python
    model = tf.keras.models.load_model('./larger_model')
    IMG_SIZE=480
    ```

    Faster inference (60ms) but lower performance (58%):
    ```python
    model = tf.keras.models.load_model('./smaller_model')
    IMG_SIZE=224
    ```

### 1.2 Usage :

- Run the app
    ```bash
    python3 main.py
    ```

- Navigate to http://localhost:8000/ for predict service UI

- Navigate to http://localhost:8000/docs for predict service API details

- Stop the app
    ```bash
    CTRL+C
    ```

## 2. Docker version

### 2.1 Installation :  

- Configure model to use in app/main.py :
   
    Better performance (68%) but slower inference (900ms):
    ```python
    model = tf.keras.models.load_model('./larger_model')
    IMG_SIZE=480
    ```

    Faster inference (60ms) but lower performance (58%):
    ```python
    model = tf.keras.models.load_model('./smaller_model')
    IMG_SIZE=224
    ```

- Build the image and install dependencies
    ```bash
    sudo docker build -t neovision .
    ```

### 2.2 Usage :

- Run the app
    ```bash
    sudo docker run -p 80:80 neovision
    ```

- Navigate to http://localhost/ for predict service UI

- Navigate to http://localhost/docs for predict service API details

- Stop the app
    ```bash
    sudo docker stop $(sudo docker ps -a -q --filter ancestor=neovision --format="{{.ID}}")
    ```
