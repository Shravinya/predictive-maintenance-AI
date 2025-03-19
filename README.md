
# AI-Powered Predictive Maintenance System

## 🚀 Project Overview
This project implements a **Predictive Maintenance System** that leverages **Machine Learning** to predict equipment failures based on sensor data. By analyzing time-series metrics, the system helps reduce downtime and optimize maintenance schedules.

## 🛠️ Features
- **Data Preprocessing**: Cleans and normalizes sensor data for training.
- **ML Model Training**: Uses a classifier to predict machine failures.
- **Flask API Deployment**: Serves predictions via a REST API.
- **Streamlit Dashboard**: Provides real-time monitoring and visualization.
- **Model Persistence**: Saves and loads models using `joblib`.

## 📂 Repository Structure
```
📦 predictive-maintenance-ai
├── 📂 data                # Raw and processed dataset
├── 📂 models              # Trained models and scalers
├── 📂 scripts             # Training and utility scripts
├── 📜 app.py              # Flask API for predictions
├── 📜 train.py            # Model training script
├── 📜 dashboard.py        # Streamlit dashboard
├── 📜 requirements.txt    # Required Python libraries
└── 📜 README.md           # Project documentation
```

## 📊 Dataset
- **Source**: Industrial IoT sensor data (Kaggle or simulated)
- **Features**:
  - `metric1, metric2, ..., metric9`: Sensor readings
  - `failure`: Target variable (0 = No failure, 1 = Failure)

## 🔧 Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/predictive-maintenance-ai.git
cd predictive-maintenance-ai

# Install dependencies
pip install -r requirements.txt
```

## 🏋️‍♂️ Training the Model
```bash
python train.py
```
After training, the model and scaler will be saved in the `models/` directory.

## 🔮 Running the API
```bash
python app.py
```
- API Endpoint: `http://127.0.0.1:5000/predict`
- Example request (POST):
  ```json
  {
    "metric1": [215630672],
    "metric2": [55],
    "metric3": [0],
    "metric4": [52],
    "metric5": [6],
    "metric6": [407438],
    "metric7": [0],
    "metric8": [0],
    "metric9": [7]
  }
  ```

## 📊 Running the Streamlit Dashboard
```bash
streamlit run dashboard.py
```
This launches an interactive UI for real-time monitoring.

## 📈 Model Performance
- **Accuracy**: 99.61%
- **Issues**: Model struggles with recall for failure class (0.00). Consider class balancing techniques.

## 🚀 Future Improvements
- **Enhance Model Recall**: Use SMOTE or undersampling to improve detection.
- **Deploy on Cloud**: Host API and dashboard using **AWS/GCP/Azure**.
- **Integrate Real-time IoT Data**: Connect with industrial sensors for live predictions.

## 📜 License
This project is licensed under the **MIT License**.

---

### 💡 Contribute
Feel free to fork, modify, and contribute! Open issues or pull requests if you have improvements. 😃
