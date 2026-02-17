# Predictive Maintenance for Compressor Machines

An end-to-end Data Science project for predictive maintenance and sales opportunities in compressor machines. This project uses machine learning to predict equipment failures and identify potential sales opportunities.

## 🎯 Features

- **Predictive Maintenance**: Predict potential equipment failures before they occur
- **Sales Opportunity Detection**: Identify machines that may need upgrades or replacements
- **Interactive Dashboard**: Streamlit-based web application for real-time predictions
- **Comprehensive Analysis**: Full EDA and model evaluation included

## 📊 Tech Stack

- **Python**: Core programming language
- **XGBoost/RandomForest**: Machine learning models
- **Streamlit**: Interactive web dashboard
- **Pandas & NumPy**: Data manipulation
- **Scikit-learn**: Model training and evaluation
- **Matplotlib/Seaborn**: Data visualization

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

2. Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

### Running the Application

1. **Run the Streamlit Dashboard**:

```bash
streamlit run src/app.py
```

2. Open your browser and navigate to `http://localhost:8501`

## 📁 Project Structure

```
.
├── data/               # Data files
├── models/             # Trained models
├── notebooks/          # Jupyter notebooks for EDA
├── plots/              # Generated visualizations
├── src/                # Source code
│   ├── app.py         # Streamlit application
│   ├── data_generator.py  # Mock data generation
│   └── ...
├── requirements.txt    # Python dependencies
└── README.md          # Project documentation
```

## 🔍 Model Performance

The model focuses on optimizing recall and precision to minimize false negatives (missed failures) while maintaining accuracy.

## 📝 License

This project is open source and available under the MIT License.

## 👥 Contributing

Contributions, issues, and feature requests are welcome!

## 📧 Contact

Your Name - your.email@example.com

Project Link: [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME)
