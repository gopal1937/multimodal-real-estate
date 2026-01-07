# 🏠 Satellite Imagery Based Property Valuation (Multimodal Regression)

## 📌 Project Overview
This project focuses on predicting residential property prices using a **multimodal regression approach** that combines traditional tabular housing data with satellite imagery. In addition to structural property attributes, visual environmental context such as surrounding infrastructure, green cover, and waterfront proximity is incorporated to improve valuation accuracy.

The project uses latitude and longitude coordinates to programmatically fetch satellite images for each property and extracts visual features using a Convolutional Neural Network (CNN). These image-based features are fused with engineered tabular features to train a multimodal regression model for property price prediction.

All experiments and data processing were implemented using **Google Colab**, with datasets and intermediate files stored and accessed via **Google Drive** for persistence and reproducibility.

## 📂 Repository Structure

The repository is organized as follows:

├── data_fetcher.py
├── preprocessing.ipynb
├── model_training.ipynb
├── prediction.ipynb
├── geospatial_visual_eda.ipynb
├── GradCAM.ipynb
├── 23115047_final.csv
├── 23115047_report.pdf
└── README.md


### File Descriptions
- **data_fetcher.py** – Script to programmatically download satellite images using latitude and longitude coordinates  
- **preprocessing.ipynb** – Tabular data exploration, feature engineering, image acquisition, and CNN-based feature extraction  
- **model_training.ipynb** – Training and evaluation of tabular-only and multimodal regression models  
- **prediction.ipynb** – Final inference pipeline to generate price predictions on the test dataset  
- **geospatial_visual_eda.ipynb** – Geospatial and visual exploratory analysis using coordinates and satellite imagery  
- **GradCAM.ipynb** – Model explainability using Grad-CAM visualizations  
- **23115047_final.csv** – Final prediction file submitted for evaluation (format: `id, predicted_price`)  
- **23115047_report.pdf** – Final project report containing EDA, architecture, results, and insights  
- **README.md** – Setup and execution instructions for the project

## ⚙️ Setup & Environment

This project is designed to run entirely on **Google Colab**, with all data storage and intermediate outputs handled through **Google Drive**. No local environment setup is required.

### Execution Platform
- Google Colab (Python 3.x)

### Storage
- Google Drive is used to store:
  - Input datasets
  - Downloaded satellite images
  - Processed features
  - Trained models
  - Final prediction outputs

### Required Input Files
The only files that must be manually uploaded by the user are:
- `train(1).xlsx` – Training dataset  
- `test2.xlsx` – Test dataset  

All other files are generated automatically by running the notebooks in the specified order.

### Library Dependencies
The project uses the following major Python libraries:
- pandas, numpy  
- matplotlib, seaborn  
- scikit-learn  
- catboost  
- torch / tensorflow  
- opencv-python, pillow  


## 📁 Google Drive Directory Structure

All project data and outputs are organized within a single Google Drive folder. Once Google Drive is mounted in Google Colab, the notebooks automatically read from and write to this structure.

```
Drive/
└── multimodal-real-estate/
    ├── data/
    │   ├── raw/
    │   │   ├── train(1).xlsx
    │   │   ├── test2.xlsx
    │   │   ├── test_image_coords.csv
    │   │   └── images/
    │   │       ├── train/
    │   │       └── test/
    │   └── processed/
    │       ├── train_image_coords.csv
    │       ├── X_img_train.npy
    │       ├── X_img_val.npy
    │       ├── X_img_test2.npy
    │       ├── X_tab_train.npy
    │       ├── X_tab_val.npy
    │       ├── X_tab_test2.npy
    │       ├── X_train.csv
    │       ├── X_val.csv
    │       ├── y_train.csv
    │       ├── y_val.csv
    │       ├── y_train_mm.npy
    │       └── y_val_mm.npy
    ├── models/
    │   └── catboost_multimodal.cbm
    ├── predictions/
    │   └── 23115047_final.csv
    └── data_fetcher.py
```



### Directory Description
- **data/raw/** – Original input datasets and raw metadata files  
- **data/raw/images/train/** – Satellite images corresponding to training properties  
- **data/raw/images/test/** – Satellite images corresponding to test properties  
- **data/processed/** – All processed tabular data, CNN-extracted image features, and train/validation splits  
- **predictions/** – Final price predictions generated using the trained multimodal model  
- **data_fetcher.py** – Script for downloading satellite images using latitude and longitude  
- **models/** - Saved final model used for prediction

All processed files and outputs are generated automatically once the input datasets are placed in the `data/raw/` directory.

## ▶️ How to Run the Project

Follow the steps below to execute the project end-to-end using Google Colab.

### Step 1: Upload Input Data
- Upload `train(1).xlsx` and `test2.xlsx` into the following directory in Google Drive:
  multimodal-real-estate/data/raw/


### Step 2: Preprocessing & Feature Engineering
- Open and run **`preprocessing.ipynb`**
- This notebook performs:
  - Tabular data exploration and feature engineering
  - Satellite image downloading for the training dataset
  - CNN-based image feature extraction
  - Generation of processed tabular and image feature files

All processed outputs are automatically saved to:
multimodal-real-estate/data/processed/


### Step 3: Model Training
- Open and run **`model_training.ipynb`**
- This notebook:
  - Trains a tabular-only baseline model using Random Forest
  - Trains a multimodal regression model using fused tabular and image features with CatBoost
  - Saves the trained multimodal model for later inference to:
    multimodal-real-estate/models/

### Step 4: Generate Predictions
- Open and run **`prediction.ipynb`**
- This notebook:
  - Downloads satellite images for the test dataset
  - Extracts CNN-based image features for test data
  - Applies the same tabular feature engineering as training
  - Loads the saved multimodal model and generates price predictions

The final prediction file is saved automatically to:
multimodal-real-estate/predictions/23115047_final.csv


### Optional Analysis Notebooks
- `geospatial_visual_eda.ipynb` – Geospatial and visual exploratory data analysis  
- `GradCAM.ipynb` – Model explainability using Grad-CAM visualizations  

These notebooks are optional and are intended for analysis and interpretability.

## 🧠 Model Summary

Two modeling approaches are implemented in this project:

### Tabular-Only Baseline Model
- A **Random Forest Regressor** is trained using engineered tabular features
- This model serves as a baseline to measure the predictive power of numerical data alone

### Multimodal Regression Model
- Satellite images are processed using a **Convolutional Neural Network (CNN)** to extract visual feature embeddings
- Engineered tabular features are fused with CNN-extracted image features
- A **CatBoost Regressor** is trained on the combined feature representation to predict property prices

### Evaluation Metrics
- Root Mean Squared Error (RMSE)
- R² Score

## 📤 Output Files

The primary output of this project is the final prediction file generated using the trained multimodal regression model.

### Prediction File
- **File name:** `23115047_final.csv`
- **Location:** `multimodal-real-estate/predictions/`
- **Format (strict):**
  id, predicted_price

This CSV file contains the predicted property prices for the test dataset and follows the required submission format exactly.

### Project Report
- **File name:** `23115047_report.pdf`
- The report includes exploratory data analysis, geospatial and visual insights, model architecture, results comparison, and explainability analysis.

## 📝 Notes & Additional Information

- All satellite images are programmatically fetched using latitude and longitude coordinates.
- CNN-based image feature extraction is used to capture environmental and neighborhood context.
- Additional notebooks such as `geospatial_visual_eda.ipynb` and `GradCAM.ipynb` are included for exploratory analysis and model explainability.
- The complete experimental analysis, visualizations, and results discussion are provided in `23115047_report.pdf`.
- The project follows the prescribed submission guidelines and file naming conventions.







