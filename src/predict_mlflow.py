import mlflow

model_name = "modele_test1"
version = 1

model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/{version}"
)

list_libs = ["vendeur d'huitres", "boulanger"]

data = {"query": list_libs,
"k": 2}
results = model.predict(data)
print(results)
