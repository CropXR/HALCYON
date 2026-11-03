# HALCYON
[![bioRxiv](https://img.shields.io/badge/bioRxiv-10.64898%2F2026.10.20.758482-b31b1b.svg)](https://doi.org/10.64898/2026.10.20.758482)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests](https://github.com/CropXR/HALCYON/actions/workflows/build_and_run_tests.yml/badge.svg)](https://github.com/CropXR/HALCYON/actions)
[![License](https://img.shields.io/github/license/CropXR/HALCYON)](LICENSE)

_Hybrid A-priori-informed Learning of Connections by Yoking ODEs with Neural Networks_, preprint available [here](https://doi.org/10.64898/2026.10.20.758482).

## MLflow integration

To setup logging to [MLFlow server of CropXR](https://mlflow.cropresilience.org) create a file called `.env` in the root HALCYON directory and provide the following pieces of information. The access token can be generated [here](https://mlflow.cropresilience.org/oidc/ui/user).
```
MLFLOW_TRACKING_URI="https://mlflow.cropresilience.org"
MLFLOW_TRACKING_PASSWORD="YOUR_ACCESS_TOKEN"
MLFLOW_TRACKING_USERNAME="YOUR_EMAIL"
```

The python package `dotenv` takes care of loading these environment variables.
