import os
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_uploads import UploadSet, configure_uploads, IMAGES
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Configuration pour l'upload d'images
photos = UploadSet('photos', IMAGES)
app.config['UPLOADED_PHOTOS_DEST'] = './static/img'
configure_uploads(app, photos)

# Chargement des modèles
# 1. Modèle CO2 (Régression polynomiale)
with open('model.pickle', 'rb') as f:
    poly, co2_model = pickle.load(f)

# 2. Modèle Image (ResNet50 - pré-entrainé)
img_model = ResNet50(weights='imagenet')

# 3. Modèle Spam (Classification de texte)
cv = pickle.load(open("models/cv.pkl", 'rb'))
clf = pickle.load(open("models/clf.pkl", 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

# --- PARTIE 1 : CO2 ---
@app.route('/predict_co2', methods=['POST'])
def predict_co2():
    # Ordre : MODELYEAR, ENGINESIZE, CYLINDERS, FUELCONSUMPTION_COMB
    input_data = [float(request.form[x]) for x in ['year', 'engine', 'cylinders', 'fuel']]
    poly_data = poly.transform([input_data])
    prediction = co2_model.predict(poly_data)
    res_text = f"Le taux de CO2 du véhicule est {round(prediction[0], 2)}"
    return render_template('index.html', co2_text=res_text)

# --- PARTIE 2 : IMAGES ---
@app.route('/predict_img', methods=['POST'])
def predict_img():
    file = request.files['photo']
    filename = photos.save(file)
    img_path = os.path.join(app.config['UPLOADED_PHOTOS_DEST'], filename)
    
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    preds = img_model.predict(x)
    _, label, prob = decode_predictions(preds, top=1)[0][0]
    res_text = f"La prédiction est {label} avec la probabilité {round(prob*100, 2)}"
    return render_template('index.html', img_text=res_text)

# --- PARTIE 3 : SPAM (WEB + API) ---
@app.route('/predict_spam', methods=['POST'])
def predict_spam():
    email_text = request.form['email']
    vect = cv.transform([email_text])
    pred = clf.predict(vect)
    res = "Spam" if pred == 1 else "Non Spam"
    return render_template('index.html', spam_text=res)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json(force=True)
    vect = cv.transform([data['email']])
    pred = int(clf.predict(vect)[0])
    return jsonify(email=data['email'], prediction=pred)

if __name__ == "__main__":
    app.run(debug=True)