import pandas as pd
import pickle
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# 1. Chargement des données 
df = pd.read_csv("FuelConsumption.csv")

# 2. Sélection des attributs demandés 
# On utilise : MODELYEAR, ENGINESIZE, CYLINDERS, FUELCONSUMPTION_COMB
X = df[['MODELYEAR', 'ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']]
y = df['CO2EMISSIONS']

# 3. Création du modèle de régression polynomiale de degré 3 
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

# 4. Sauvegarde du modèle et de l'objet poly [cite: 24]
# Note : On sauvegarde aussi 'poly' car on en aura besoin dans app.py 
# pour transformer les saisies utilisateur avant la prédiction.
with open('model.pickle', 'wb') as f:
    pickle.dump((poly, model), f)

print("Modèle entraîné et sauvegardé dans model.pickle !")