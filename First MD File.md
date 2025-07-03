# Credit Card Fraud Detection with TensorFlow

This project builds and compares neural network models using different optimizers to detect fraudulent credit card transactions. The dataset is highly imbalanced and contains anonymized features, making it a realistic challenge for machine learning.

## 📊 Dataset

- **Source:** [Credit Card Fraud Detection Dataset](https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv)
    
- **Description:** Contains transactions made by European cardholders in September 2013.
    
    - **Features:**
        
        - `Time`: Seconds elapsed between each transaction and the first transaction
            
        - `Amount`: Transaction amount
            
        - `V1`–`V28`: Principal components from PCA transformation
            
        - `Class`: Target variable (1 = fraud, 0 = normal)
            
- The dataset is highly imbalanced: frauds account for only 0.172% of all transactions.
    

## 🚀 Project Objective

Develop and compare deep learning models with different optimizers to classify transactions as fraudulent or legitimate.

## 🛠️ Requirements

- Python 3.x
    
- TensorFlow
    
- scikit-learn
    
- pandas
    
- matplotlib
    

Install dependencies using:

bash

`pip install tensorflow scikit-learn pandas matplotlib`

## 📝 How It Works

1. **Data Loading & Preprocessing**
    
    - Downloads the dataset directly from the web.
        
    - Drops the `Time` and `Class` columns for features; `Class` is the target.
        
    - Splits data into training and test sets.
        
    - Scales features using `StandardScaler`.
        
2. **Model Architecture**
    
    - A simple feedforward neural network with three hidden layers (`relu` activation) and a sigmoid output for binary classification.
        
3. **Training & Comparison**
    
    - Compares the performance of different optimizers:
        
        - SGD (Vanilla, with Momentum, with Nesterov)
            
        - Adagrad
            
        - RMSprop
            
        - Adam
            
    - Trains each model and plots loss and accuracy curves.
        
4. **Visualization**
    
    - Plots training and validation loss and accuracy for each optimizer comparison.
        

## 💻 Usage

Clone the repository and run the script:

bash

`python your_script_name.py`

You can modify the optimizer settings or model architecture as needed.

## 📈 Results

The script will display plots comparing the performance of different optimizers on loss and accuracy metrics for fraud detection.

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## 📄 License

This project is for educational purposes.

**References:**

- The dataset and problem description are inspired by the Kaggle Credit Card Fraud Detection challenge and similar public resources[](https://github.com/NLGRF/Credit-Card-Fraud-Detection-using-CNN-in-TensorFlow-2.0)[](https://github.com/dheerajrathee/TensorFlow-for-Detection-Credit-Card-Fraud)[](https://www.geeksforgeeks.org/ml-credit-card-fraud-detection/).
    

If you want to add more sections (such as example output, advanced usage, or acknowledgements), just let me know!
