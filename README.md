# HALCYON

Hybrid A-priori-informed Learning of Connections by Yoking ODEs with Neural Networks.

WIP by 
Thijs van Loo & Ben Noordijk

## MLflow integration

To setup logging to [MLFlow server of CropXR](https://mlflow.cropresilience.org) create a file called `.env` in the root HALCYON directory and provide the following pieces of information. The access token can be generated [here](https://mlflow.cropresilience.org/oidc/ui/user).
```
MLFLOW_TRACKING_URI="https://mlflow.cropresilience.org"
MLFLOW_TRACKING_PASSWORD="YOUR_ACCESS_TOKEN"
MLFLOW_TRACKING_USERNAME="YOUR_EMAIL"
```

The python package `dotenv` takes care of loading these environment variables.