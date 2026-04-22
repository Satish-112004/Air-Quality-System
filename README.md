# 🇮🇳 India Air Quality Analysis (2015–2020)
### End-to-End Data Preprocessing + EDA | Python · Pandas · Scikit-learn · Seaborn

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Pandas](https://img.shields.io/badge/Pandas-1.5%2B-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-2A9D8F?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-57CC99?style=flat-square)]()

---

## 📌 Project Overview

A **complete data science preprocessing and EDA pipeline** on India's Air Quality dataset (2015–2020), covering **29,531 daily pollution records** across **26 Indian cities**.

The pipeline cleans, engineers, encodes, and scales raw data into an ML-ready dataset — while extracting insights on pollution patterns, seasonal trends, city-wise comparisons, and the **COVID-19 lockdown impact** on air quality.

---

## 📊 Key Insights

| Finding | Detail |
|---|---|
| 🏆 Worst city | **Ahmedabad (AQI 371)** — higher than Delhi on average |
| 📉 India improving | AQI dropped from **202 (2015) → 112 (2020)**, nearly halved |
| ❄️ Winter is worst | Winter AQI **(213)** is 46% higher than non-winter (146) |
| 🦠 COVID effect | Delhi AQI crashed visibly in **April 2020** — no traffic, no industry |
| ⚠️ Rarely clean air | Only **4.5% of days** across all cities qualify as "Good" |
| 🔗 PM2.5 ↔ PM10 | Highly correlated — same pollution source |

---

## 🗂 Dataset

**Source:** [Kaggle — Air Quality Data in India (2015–2020)](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)  
**Used file:** `city_day.csv` — one row per city per day, cleanest aggregation level.

**Columns:** City, Date, PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene, AQI, AQI_Bucket

---

## 🏗 Project Structure

```
india-air-quality/
│
├── air_quality_preprocessing.py   ← Full pipeline (Steps 0–10 + Visualization)
├── city_day_clean.csv             ← Cleaned data, original scale
├── city_day_ml.csv                ← Scaled + encoded, ML-ready
├── India_AQI_EDA.png              ← 8-chart EDA dashboard
├── outlier_boxplots.png           ← IQR outlier check
└── README.md
```

---

## ⚙️ Preprocessing Pipeline

### STEP 0 — ZIP Extraction
Automated extraction of all 5 CSV files using Python's built-in `zipfile` library.

### STEP 1 — Load Data
`city_day.csv` loaded into a pandas DataFrame. Initial shape: **29,531 rows × 16 columns**.

### STEP 2 — Initial Inspection
Full inspection using `df.info()`, `df.describe()`, and `df.isnull().sum()`.

| Column | Missing % | Decision |
|---|---|---|
| PM2.5 | 15.6% | `ffill + bfill` per city |
| PM10 | 37.7% | `ffill + bfill` per city |
| NH3 | 35.0% | `ffill + bfill` per city |
| **Xylene** | **61.3%** | **Dropped** |
| AQI_Bucket | 15.9% | Re-derived from AQI |

### STEP 3 — Missing Values
Time-series correct imputation — sort by `City + Date`, then `ffill()` + `bfill()` within each city group. AQI_Bucket re-derived using official CPCB scale.

```python
df.sort_values(["City", "Date"], inplace=True)
df[col] = df.groupby("City")[col].transform(lambda x: x.ffill().bfill())
```

### STEP 4 — Duplicates
Checked on `subset=["City", "Date"]` (logical primary key). Result: **0 duplicates**.

### STEP 5 — Outlier Treatment
IQR Capping (Winsorization) with `factor=3.0` — values clipped to fence, not removed. Pollution spikes like Diwali or stubble burning are real events, not errors.

```python
def cap_iqr(series, factor=3.0):
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = Q3 - Q1
    return series.clip(Q1 - factor * IQR, Q3 + factor * IQR)
```

### STEP 6 — Feature Engineering
5 new features created with clear physical meaning:

| Feature | Formula | Meaning |
|---|---|---|
| `Combustion_Index` | `NO2 + CO` | Traffic + industrial burning signal |
| `Smog_Risk` | `O3 × NO2 / 100` | Photochemical smog formation risk |
| `PM_Ratio` | `PM2.5 / PM10` | Particle source — exhaust vs dust |
| `Is_Winter` | `Month ∈ {12,1,2}` | Temperature inversion season flag |
| `High_Pollution_Day` | `AQI > 200` | Dangerous air quality binary flag |

### STEP 7 — Encoding
- **AQI_Bucket** → Manual ordered mapping (`Good=0` to `Severe=5`)
- **Season** → One-Hot Encoding with `drop_first=True`

```python
bucket_order = {"Good":0, "Satisfactory":1, "Moderate":2,
                "Poor":3, "Very Poor":4, "Severe":5}
df["Pollution_Level"] = df["AQI_Bucket"].map(bucket_order)
```

### STEP 8 — Scaling
`StandardScaler` applied to raw pollutant columns only. AQI excluded from features to prevent **data leakage** (AQI is mathematically derived from the same pollutants).

```python
feature_cols = ["PM2.5","PM10","NO","NO2","NOx","NH3",
                "CO","SO2","O3","Benzene","Toluene",
                "Combustion_Index","Smog_Risk","PM_Ratio"]

scaler = StandardScaler()
df_ml[feature_cols] = scaler.fit_transform(df[feature_cols])
```

### STEP 9 — Final Output

```
Final shape      : 29,531 rows × 23 columns
Missing values   : 0
Duplicate rows   : 0
Target variable  : Pollution_Level (0 = Good → 5 = Severe)
```

---

## 📈 EDA Dashboard

![India AQI EDA Dashboard](India_AQI_EDA.png)

8 charts covering city ranking, year-wise trend, monthly heatmap, PM2.5 distribution, correlation heatmap, seasonal patterns, and the COVID lockdown AQI dip in Delhi.

---

## 🚀 How to Run

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/india-air-quality.git
cd india-air-quality
```

**2. Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

**3. Place the dataset ZIP in the project root**
```
india-air-quality/
└── Air_Quality_Data_in_India__2015_-_2020__archive__4_.zip
```

**4. Run**
```bash
python air_quality_preprocessing.py
```

---

## 🛠 Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading, cleaning, transformation |
| `numpy` | Numerical operations |
| `matplotlib` | Base plotting |
| `seaborn` | Statistical visualizations |
| `scikit-learn` | StandardScaler, preprocessing |

---

## 🔮 Next Steps

- [ ] Classification model to predict `Pollution_Level` (Random Forest / XGBoost)
- [ ] Handle class imbalance using SMOTE or class weights
- [ ] Time-series forecasting — predict next day's AQI from past 7 days
- [ ] Model evaluation — accuracy, F1-score, confusion matrix
- [ ] Streamlit dashboard deployment

---

## 👤 Author

**Satish**  
Data Science Student — DSCE Bangalore

[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat-square&logo=github)](https://github.com/YOUR_USERNAME)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/YOUR_PROFILE)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.  
Dataset: [Rohan Rao on Kaggle](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india) | Source: Central Pollution Control Board (CPCB), India.

---

<div align="center">
  <sub>⭐ Star this repo if you found it useful</sub>
</div>
