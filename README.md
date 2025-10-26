#      NEXUS - Intelligent Catering Optimization Platform

<div align="center">

![NEXUS Logo](https://img.shields.io/badge/NEXUS-AI%20Powered-1e3c72?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-FFD700?style=for-the-badge&logo=python&logoColor=1e3c72)
![Machine Learning](https://img.shields.io/badge/ML-Random%20Forest-2a5298?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Advanced predictive analytics for flight catering optimization**

[Demo](#-demo) • [Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Model](#-machine-learning-model)

</div>

---

## 📋 Overview

**NEXUS** is an AI-powered dashboard designed for **Gategroup** flight operations to optimize catering load planning. By leveraging machine learning predictions, it reduces weight, saves fuel, and minimizes food waste across commercial flights.

### 🎯 Problem Statement

Airlines face significant challenges in catering planning:
- **Overstocking**: Excess food increases weight → higher fuel costs
- **Understocking**: Running out of items → passenger dissatisfaction
- **Manual planning**: Time-consuming and prone to errors

### 💡 Solution

NEXUS uses historical consumption data to predict optimal catering quantities per flight, considering:
- Flight characteristics (origin, type, service class)
- Passenger count (with real-time adjustment capabilities)
- Product-specific consumption patterns
- Seasonal and temporal trends

---

## 🚀 Features

### 🔮 AI-Powered Predictions
- **Random Forest Regression** model trained on historical flight data
- Predicts consumption for 10+ catering products per flight
- Adjustable safety buffer (5-20%) to prevent stockouts

### 📊 Interactive Dashboard
- **Flight selection**: Browse and filter from complete flight database
- **Passenger simulation**: Adjust passenger count to test different scenarios
- **Real-time recalculations**: Instant updates to predictions and metrics

### 💰 Impact Analytics
- **Weight savings**: Calculate kg reduced vs. standard specification
- **Fuel savings**: Estimate kg of fuel saved (3% weight-to-fuel ratio)
- **Cost savings**: Convert fuel savings to USD ($0.80/kg)
- **Reduction percentage**: Overall weight optimization metric

### 🧳 Trolley Optimization
- **Automated trolley calculation**: Based on 80kg capacity per trolley
- **Load visualization**: Progress bars showing utilization per trolley
- **Last trolley gauge**: Real-time utilization percentage

### 📥 Export & Reports
- **CSV export**: Download packing checklist for ground crew
- **Filename intelligence**: Auto-appends passenger count if adjusted
- **Product-level breakdown**: Clear instructions per item

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Clone Repository
```bash
git clone https://github.com/yourusername/nexus-catering-optimization.git
cd nexus-catering-optimization

pip install -r requirements.txt

streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.17.0
joblib>=1.3.0
