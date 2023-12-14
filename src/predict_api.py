import pandas as pd
import requests


# Fonction pour effectuer la requête à l'API
def make_prediction(api_url: str, description: str):
    params = {"description": description, "nb_echoes_max": 2}
    response = requests.get(api_url, params=params)
    return response.json()


# URL des données
data_path = "https://minio.lab.sspcloud.fr/projet-formation/diffusion/mlops/data/data_to_classify.parquet"

# Charge le fichier Parquet dans un DataFrame pandas
df = pd.read_parquet(data_path)

# Votre API URL
api_url = "https://yves-colt-api.lab.sspcloud.fr/predict"

# Effectue les requêtes
responses = df["text"].apply(lambda x: make_prediction(api_url, x))

# Affiche le DataFrame avec les résultats des prédictions
print(pd.merge(df, pd.json_normalize(responses),
               left_index=True,
               right_index=True))