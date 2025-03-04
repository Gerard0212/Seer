# Phishing Detector - Random Forest Model

This repository contains the training and evaluation code for a phishing detection model using Random Forest. The code (`rf_model.py`) performs the following tasks:

- Loads and preprocesses the dataset (ARFF to CSV conversion if needed)
- Performs feature engineering
- Trains a Random Forest classifier
- Evaluates the model and prints accuracy and classification metrics

## Setup

1. **Clone the Repository**

   ```bash
   git clone <repository_url>
   cd phishing-detector
   ```

2. **Create and Activate a Virtual Environment**

   ```bash
   python -m venv env
   Windows: .\env\Scripts\activate
   ```
   
3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```
   
4. **Run the Model Training Script**
   ```bash
   python rf_model.py
   ```