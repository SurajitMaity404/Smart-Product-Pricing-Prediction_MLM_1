# Smart-Product-Pricing-Prediction_MLM_1
This LightGBM regression model forecasts product prices from catalog text (IPQ counts, stats, brand hashing, SVD vectors) and optional image features (RGB means, histograms). Trained via 5-fold CV on log-transformed targets, it optimizes for SMAPE and outputs submission predictions.
