
## **Methodology**

The overall workflow follows a standard end-to-end machine learning pipeline for regression:

1. **Data Loading and Preprocessing**: Load training and test datasets from CSV files. Handle missing values by filling empty strings for text fields and NaNs for numerics.  
2. **Feature Engineering**: Transform raw text (catalog\_content) and image URLs (image\_link) into numerical features. This includes rule-based extraction, statistical summaries, vectorization, and image-derived descriptors (detailed in the Feature Engineering section).  
3. **Target Transformation**: Apply log1p transformation to the target variable (price) to stabilize variance and handle skewness in price distributions, common in e-commerce data.  
4. **Model Training with Cross-Validation**: Use 5-fold KFold cross-validation (shuffled, seeded for reproducibility) to train an ensemble regressor. For each fold:  
   * Split data into train/validation sets.  
   * Fit the model with early stopping based on validation RMSE.  
   * Generate out-of-fold (OOF) predictions for evaluation and average fold predictions for the test set.  
5. **Evaluation and Submission**: Compute OOF SMAPE on the original price scale (inverse log1p). Generate test predictions and save as submission.csv with sample\_id and price columns.

## **Model Architecture/Algorithms Selected**

The core model is **LightGBM (Gradient Boosting Decision Trees)**, selected for its speed, scalability, and strong performance on tabular data with mixed feature types (numerical, hashed categoricals).

**Key Hyperparameters**

The model uses a regression objective with RMSE metric and the following configuration:

* **Boosting Type**: gbdt (Gradient Boosting Decision Tree).  
* **Learning Rate**: 0.05 (balanced for convergence).  
* **Num Leaves**: 31 (controls tree complexity).  
* **Feature/Bagging Fractions**: 0.8 each, with bagging frequency of 5 (introduces stochasticity for generalization).  
* **Regularization**: L1/L2 lambdas set to 0.0 (relies on early stopping instead).  
* **Training Rounds**: Up to 5000, with early stopping after 100 rounds of no validation improvement.  
* **Seed**: 42 for reproducibility.

**Feature Engineering Techniques Applied**

### **Text-Based Features**

* **IPQ Extraction**: Rule-based regex patterns to parse quantity indicators (e.g., "pack of 12", "IPQ:5") from text. Fallback to single-digit extraction if no match. Handles non-string inputs as NaN.  
* **Basic Text Statistics**: Length-based descriptors:  
  * chars: Character count.  
  * words: Word count (space-split).  
  * upper\_ratio: Proportion of uppercase characters (indicative of emphasis/brands).  
* **Brand Extraction**: Heuristic to grab the first token from the text head (pre-processed to remove separators like '-', '•'). Lowercased and hashed (brand\_hash) to a 1M-bucket categorical for model input (avoids high-cardinality one-hot encoding).  
* **Dimensionality Reduction**:  
  * HashingVectorizer (2^16 features, 1-2 n-grams, L2-normalized) for sparse, memory-efficient text representation.  
  * TruncatedSVD (64 components, seeded) on hashed vectors to capture latent semantics (e.g., product categories).

### **Image-Based Features (Optional)**

If USE\_IMAGE\_FEATURES=True, features are extracted via HTTP requests to image URLs:

* Download and resize images to 64x64 pixels (thumbnail for speed).  
* Convert to RGB array (normalized to \[0,1\]).  
* Compute:  
  * Mean RGB values (3 features: red, green, blue – for color profiling).  
  * Image area (height × width post-resize).  
  * Channel-wise histograms (8 bins per RGB channel \= 24 features) for texture/color distribution.  
* Handles errors (e.g., invalid URLs) by returning NaN vectors (28 total per image).  
* Progress tracked via tqdm for \~thousands of images.

### **Feature Pipeline**

* Numeric columns (all except raw brand) are selected, imputed, and scaled globally (fit on train, transform test).  
* Brand hash is concatenated post-scaling as a low-cardinality numeric proxy.  
* Final shapes: Train (\~N rows × 100+ cols), Test (matching).

## **Additional Relevant Information**

* **Evaluation Metric**: Custom SMAPE implementation (100 × |y \- ŷ| / (|y| \+ |ŷ|)/2, with denom=1 for zeros) on original scale. OOF SMAPE printed for quick assessment (e.g., \~X% in runs).  
* **Reproducibility & Environment**: Fixed SEED=42; runs in Google Colab with pip-installed LightGBM/tqdm. No external package installs beyond basics; assumes CSV uploads via files.upload().