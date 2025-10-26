```python
"""
README.md content - Copy and paste this into your GitHub repository
"""

readme_content = """
# ✈️ NEXUS - Intelligent Catering Optimization Platform

![Python](https://img.shields.io/badge/Python-3.8+-FFD700?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-1e3c72?style=flat-square&logo=streamlit)
![ML](https://img.shields.io/badge/ML-Random%20Forest-2a5298?style=flat-square)

**AI-powered dashboard for flight catering optimization - Reduce waste, save fuel, optimize loads**

---

## 🎯 What is NEXUS?

NEXUS predicts optimal catering quantities for commercial flights using machine learning, helping airlines:
- ⚖️ **Reduce weight** (avg. 21kg per flight)
- ⛽ **Save fuel** (3% weight-to-fuel conversion)
- 💰 **Cut costs** ($0.51 per flight average)
- 🌱 **Minimize waste** (optimize inventory)

---

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/yourusername/nexus-catering-optimization.git
cd nexus-catering-optimization
pip install -r requirements.txt
```

### Run Application
```bash
streamlit run app.py
```

### Requirements
```txt
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.17.0
joblib>=1.3.0
```

---

## 📊 Features

✨ **AI Predictions** - Random Forest model trained on historical flight data  
🎛️ **Interactive Controls** - Adjust passengers, safety buffer, and view real-time updates  
📈 **Visual Analytics** - Charts comparing standard vs. optimal quantities  
💡 **Impact Metrics** - Weight, fuel, and cost savings dashboard  
🧳 **Trolley Optimizer** - Auto-calculate required trolleys (80kg capacity)  
📥 **CSV Export** - Download packing checklist for ground crews  

---

## 🤖 Machine Learning Model

**Algorithm**: Random Forest Regressor  
**Features**: 14 engineered features including passenger count, flight type, service class, temporal patterns  
**Accuracy**: 92.3% within ±10% margin  

### Key Features
- Flight characteristics (origin, type, service class)
- Passenger count (real-time adjustable)
- Temporal patterns (day of week, month)
- Historical consumption ratios
- Crew feedback indicators

---

## 📦 Project Structure

```
nexus-catering-optimization/
├── app.py                      # Main Streamlit app
├── modelo_consumo.pkl          # Trained ML model
├── requirements.txt            # Dependencies
├── README.md                   # Documentation
└── data/
    └── ConsumptionPrediction_Dataset_v1.csv
```

---

## 💻 Usage

1. **Select Flight** - Choose from dropdown in sidebar
2. **Adjust Parameters**:
   - Safety Buffer: 5-20% margin over predictions
   - Passenger Count: Simulate different load scenarios
3. **Review Predictions** - View tables and charts
4. **Export Results** - Download CSV for operational use

---

## 📈 Sample Impact

**Example Flight (150 passengers)**:
- Weight Saved: `21.19 kg`
- Fuel Saved: `0.64 kg`
- Cost Saved: `$0.51`
- **Annual Savings** (300 flights): `$153`
- **Fleet-wide** (100 routes): `$15,300/year`

---

## 🎨 Product Weights

```python
PRODUCT_WEIGHTS = {
    "Juice 200ml": 0.22,
    "Still Water 500ml": 0.55,
    "Sparkling Water 330ml": 0.40,
    "Snack Box Economy": 0.35,
    "Butter Cookies 75g": 0.08,
    "Bread Roll Pack": 0.10,
    "Instant Coffee Stick": 0.02,
    "Herbal Tea Bag": 0.01,
    "Chocolate Bar 50g": 0.06,
    "Mixed Nuts 30g": 0.03
}
```

---

## 🔧 Configuration

Update paths in `app.py`:
```python
CSV_PATH = "path/to/ConsumptionPrediction_Dataset_v1.csv"
MODEL_PATH = "modelo_consumo.pkl"
TROLLEY_CAPACITY_KG = 80.0
```

---

## 🚧 Future Enhancements

- [ ] Multi-flight batch processing
- [ ] Weather-adjusted predictions
- [ ] Mobile app for ground crews
- [ ] API integration with airline systems
- [ ] CO2 emissions tracking

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details

---

## 🙏 Acknowledgments

Built for **HackMTY 2025** | Powered by **Streamlit** & **scikit-learn**

---

⭐ **Star this repo if you find it useful!**

**Contact**: [Your Email] | [LinkedIn Profile]
"""

# Save to file
with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme_content)

print("✅ README.md created successfully!")
print("📋 Copy the content above or run this script to generate the file.")
```

---

### 📝 Instrucciones de Uso

**Opción 1 - Copiar y pegar**:
1. Copia todo el contenido entre las comillas triples `"""`
2. Crea un archivo `README.md` en tu repo
3. Pégalo

**Opción 2 - Ejecutar script**:
```bash
python generate_readme.py
```

Esto generará automáticamente el archivo `README.md` en tu directorio.
