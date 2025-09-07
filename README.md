# BitCoin-Price-Prediction

A machine learning-based project for predicting short-term Bitcoin price movements leveraging historical market data.

## Overview

This repository implements a predictive model trained on historical Bitcoin data—including Open, High, Low, Close prices, Volume, and Daily Returns. The model forecasts short-term Bitcoin price movements, intended to assist traders and analysts in decision-making. :contentReference[oaicite:0]{index=0}

## Repository Contents

- `bitcoin_2020-01-13_2024-11-12.csv` — Dataset containing Bitcoin price data from January 13, 2020 to November 12, 2024. :contentReference[oaicite:1]{index=1}  
- `train.ipynb` — Jupyter notebook for data preprocessing and model training. :contentReference[oaicite:2]{index=2}  
- `bitcoin_price_model.pkl` & `modelv2.pkl` — Serialized model versions for deployment or inference. :contentReference[oaicite:3]{index=3}  
- `app.py` — Script for serving predictions via a web interface or API. :contentReference[oaicite:4]{index=4}  
- `static/` and `templates/` — Frontend assets and HTML templates for the app interface. :contentReference[oaicite:5]{index=5}  
- `requirements.txt` — Lists project dependencies. :contentReference[oaicite:6]{index=6}  

## Prerequisites

- Python 3.x  
- Install required dependencies:

  ```bash
  pip install -r requirements.txt
