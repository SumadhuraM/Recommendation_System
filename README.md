# 🎯 E-Commerce Recommendation System

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/ML-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)

A comprehensive recommendation system with three AI models and PCA vs SVD comparison for e-commerce applications.

## 📁 Project Structure

```
├── app.py                    # Main Streamlit application
├── recommendation.ipynb      # Jupyter notebook with analysis
├── requirements.txt          # Python dependencies
├── README.md                 # Documentation
└── data/                     # Data files (not in repo)
    ├── product_descriptions.csv
    └── ratings_Beauty.csv
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data
Place your data files in the project directory:
- `product_descriptions.csv` - Product data with descriptions
- `ratings_Beauty.csv` - User ratings data

### 3. Run the Application
```bash
streamlit run app.py
```

## 🎯 Features

### Three Recommendation Models

| Model | Purpose | Best For |
|-------|---------|----------|
| **🔥 Popularity-Based** | Shows trending products | New customers |
| **👥 Collaborative Filtering** | Personalized recommendations | Returning customers |
| **🔍 Content-Based** | Similar product suggestions | Cold start scenarios |

### 📊 Advanced Analysis
- **PCA vs SVD Comparison**: Find the best dimensionality reduction technique
- **Error Metrics**: MSE, MAE, and Explained Variance
- **Visual Comparisons**: Interactive charts and graphs

## 🛠️ Technical Details

### Requirements
- Python 3.8+
- Streamlit 1.28+
- Pandas, NumPy, Scikit-learn, Plotly

### Data Format
The system automatically handles these column names:
- **Products**: `product_id`, `product_name`, `product_description`
- **Ratings**: `user_id`, `product_id`, `rating`

## 📊 Notebook Analysis
The `recommendation.ipynb` notebook contains:
- Data exploration and preprocessing
- Model development and testing
- Performance analysis
- Visualization examples

## 🔧 Configuration

Modify these paths in `app.py` if needed:
```python
products_path = r"C:\Users\sumad\Downloads\recommendation\product_descriptions.csv"
ratings_path = r"C:\Users\sumad\Downloads\recommendation\ratings_Beauty.csv"
```

## 📈 Usage Guide

1. **Load Data**: Click "Load All Data" in the sidebar
2. **Navigate**: Use the top navigation buttons
3. **Generate Recommendations**: Click buttons in each section
4. **Compare PCA vs SVD**: Adjust components and run comparison

## 🐛 Troubleshooting

### Common Issues

1. **Data not loading**: Check file paths and ensure CSV files exist
2. **Package errors**: Install all requirements: `pip install -r requirements.txt`
3. **Memory issues**: Reduce `nrows` parameter in load_data() function

### For Windows Users
```cmd
# Create virtual environment
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## 📄 License

MIT License

## 👤 Author

Sumadhura M

---
