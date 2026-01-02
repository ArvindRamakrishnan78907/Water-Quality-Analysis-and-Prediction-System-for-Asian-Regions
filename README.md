# 🌊 Asian Water Quality Dashboard

A real-time water quality monitoring dashboard for Asian countries, built with Streamlit.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/asian-water-quality-dashboard.git
cd asian-water-quality-dashboard

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
```

Open http://localhost:8501 in your browser.

## ✨ Features

- **49 Asian Countries** with comprehensive river basin data
- **Real-time API** integration for Hong Kong (EPD)
- **WHO/EPA Standards** for accurate water quality assessment
- **Multiple Visualizations**: Heatmap, Bar Chart, Line Chart, Data Table
- **Water Quality Status**: Excellent/Good/Moderate/Poor ratings
- **Prediction System**: Risk assessment based on historical data
- **File Upload**: Support for custom CSV/Excel data

## 📊 Data Sources

| Country | Source | Type |
|---------|--------|------|
| Hong Kong | Environmental Protection Department | Live API |
| India | Central Pollution Control Board (CPCB) | Reference |
| China | China National Environmental Monitoring Centre | Reference |
| Japan | NIES via GEMStat | Reference |
| South Korea | Water Environment Information System | Reference |
| Others | Simulated (WHO/EPA standards) | Simulated |

## 🔬 Water Quality Indicators

Based on WHO/EPA scientific standards:

- **Chlorophyll-a** (µg/L) - Algal biomass
- **pH** (6.5-8.5 optimal) - Acidity level
- **Total Nitrogen** (mg/L) - Nutrient pollution
- **Total Phosphorus** (mg/L) - Eutrophication
- **E. coli** (CFU/100mL) - Fecal contamination
- **Dissolved Oxygen** (mg/L) - Aquatic life support
- **Turbidity** (NTU) - Water clarity

## 📁 Project Structure

```
├── streamlit_app.py          # Main Streamlit application
├── asian_water_quality_data.py  # Data module with API integrations
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🛠️ Requirements

- Python 3.8+
- Dependencies in `requirements.txt`

## 📄 License

MIT License - Feel free to use and modify.

## 🤝 Contributing

Pull requests welcome! Please open an issue first for major changes.
