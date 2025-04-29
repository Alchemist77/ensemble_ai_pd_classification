# PD-Ensemble Deep Learning Classification

A Deep Learning framework for Parkinson’s Disease detection based on upper and lower limb motor data analysis.

---

## Goal

- Classification of Parkinson’s Disease (PD) vs Healthy Controls (HC)  
- Feature extraction from wearable sensor data (accelerometer and gyroscope)  
- Ensemble of multiple deep learning models (LSTM, GRU, BiLSTM, BiGRU, Transformer, CNN1D)  

---

## Environment Setup (Python 3.8+)

1. Clone the repository:

   ```bash
   git clone https://github.com/Alchemist77/PD-Ensemble-Classification.git
   cd PD-Ensemble-Classification
   ```
2. Install Python packages:
      ```bash
  pip install -r requirements.txt
   ```


## Dataset Preparation
Place your dataset .xlsx file in the dataset/ directory.

Example: dataset/DATA_HC_PD_total.xlsx (After publish, we will upload this dataset)

## Run Training and Testing
   ```bash
   python3 train_test_avg.py
   ```
References
This work is based on the paper:
Kim, J., Fiorini, L., Maremmani, C., Cavallo, F., Rovini, E. (2025)
"PD-Ensemble Deep Learning Classification model for Parkinson’s Disease detection based on body motion analysis."
(Submitted to IEEE Transactions)



