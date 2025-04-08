# Reto DaCiberSalut - Generación de modelos IA para identificación de tejidos
# Autores: Ivan Falcon Monzon, Ismael Diaz Sancha, David Rodriguez Gerrard
# Instalar automáticamente las dependencias si no están instaladas
import subprocess
import sys

try:
    import flask
    import numpy
    import PIL
    import torch
    import torchvision
except ImportError:
    print("Instalando dependencias desde requirements.txt...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

# Librerías necesarias
from flask import Flask, render_template, request, jsonify
from flask import send_from_directory
import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms

app = Flask(__name__)

# ================================
# CONFIGURACIÓN DEL MODELO
# ================================
# Configurar el dispositivo (GPU si está disponible, de lo contrario CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Definir la arquitectura del modelo
modelo = models.resnet50(weights=None)
num_ftrs = modelo.fc.in_features
modelo.fc = torch.nn.Linear(num_ftrs, 5)  # Ajustamos para 5 clases

# Cargar pesos del modelo preentrenado
state_dict = torch.load('model//resnet50_cancer.pth', map_location=device)
modelo.load_state_dict(state_dict)
modelo.to(device)
modelo.eval()  # Poner el modelo en modo evaluación

# ================================
# PREPROCESAMIENTO DE IMÁGENES
# ================================
transformaciones = transforms.Compose([
    transforms.Resize((512, 512)),  # Redimensionar a 512x512 píxeles
    transforms.ToTensor(),          # Convertir la imagen a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalización estándar de ImageNet
                         std=[0.229, 0.224, 0.225])
])

# ================================
# RUTAS DE LA APLICACIÓN
# ================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/index_en')
def index_en():
    return render_template('index_en.html')

@app.route('/static/images/favicon.png')
def favicon():
    return send_from_directory('static/images', 'favicon.png', mimetype='image/x-icon')

# Clases del modelo
CLASSES = [
    "colon_aca",  # Colon adenocarcinoma
    "colon_n",    # Colon tejido benigno
    "lung_aca",   # Pulmón adenocarcinoma
    "lung_n",     # Pulmón tejido benigno
    "lung_scc"    # Pulmón carcinoma de células escamosas
]

@app.route('/predict', methods=['POST'])
def predict():
    """
    Ruta para predecir la clase de una imagen enviada por el usuario.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No se encontró ninguna imagen'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nombre de archivo vacío'})

    try:
        # Cargar y preprocesar la imagen
        img = Image.open(file).convert('RGB')
        img_tensor = transformaciones(img).unsqueeze(0).to(device)  # Aplicar transformaciones y añadir dimensión de batch

        # Hacer la predicción
        with torch.no_grad():
            outputs = modelo(img_tensor)
            probabilidades = torch.nn.functional.softmax(outputs, dim=1)  # Obtener probabilidades por clase

        # Convertir las probabilidades a un diccionario con formato JSON
        prediction_dict = {CLASSES[i]: float(probabilidades[0][i].item()) * 100 for i in range(len(CLASSES))}

        # Verificar si la probabilidad máxima es menor al 70%, en tal caso asignar "desconocido"
        max_prob = max(prediction_dict.values())
        if max_prob < 70:
            prediction_dict = {"desconocido": 100.0}  # Asignar "desconocido" si la probabilidad es menor a 70%

        return jsonify(prediction_dict)
    
    except Exception as e:
        return jsonify({'error': f'Error procesando la imagen: {str(e)}'})

# ================================
# EJECUCIÓN DE LA APLICACIÓN
# ================================
if __name__ == '__main__':
    app.run(debug=True)
