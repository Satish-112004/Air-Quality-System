# 🇮🇳 India Air Quality Analysis (2015–2020)
### End-to-End Data Preprocessing + EDA | Python · Pandas · Scikit-learn · Seaborn

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Pandas](https://img.shields.io/badge/Pandas-1.5%2B-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-2A9D8F?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-57CC99?style=flat-square)]()

---

## 📌 Project Overview

This project performs a **complete data science preprocessing pipeline** on the India Air Quality dataset (2015–2020), covering **29,531 daily pollution records** across **26 Indian cities**.

The goal is to clean, engineer, encode, and scale the raw data into an ML-ready dataset — while extracting real insights about pollution patterns, seasonal trends, city-wise differences, and the visible effect of the **COVID-19 lockdown** on air quality.

> Built as part of the **Data Science Journey** at DSCE Bangalore — following a structured 10-step preprocessing guide from raw ZIP to ML-ready CSV.

---

## 📊 Key Insights Discovered

| Finding | Detail |
|---|---|
| 🏆 Worst city | **Ahmedabad (AQI 371)** — worse than Delhi on average |
| 📉 India is improving | AQI dropped from **202 (2015) → 112 (2020)**, nearly halved |
| ❄️ Winter is brutal | Winter AQI **(213)** is 46% higher than non-winter (146) |
| 🦠 COVID effect | Delhi AQI crashed visibly in **April 2020** — no traffic, no industry |
| ⚠️ Only 4.5% "Good" days | Most days are **Moderate or worse** across all cities |
| 🔗 PM2.5 ↔ PM10 correlated | Same source — consider dropping one before modeling |

---

## 🗂 Dataset

**Source:** [Kaggle — Air Quality Data in India (2015–2020)](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)

| File | Description |
|---|---|
| `city_day.csv` | Daily city-level averages ✅ **Used in this project** |
| `city_hour.csv` | Hourly city-level data |
| `station_day.csv` | Daily station-level data |
| `station_hour.csv` | Hourly station-level data |
| `stations.csv` | Station metadata |

**Why `city_day.csv`?** It is the cleanest aggregation — one row per city per day. Granular enough for trend analysis, not noisy like hourly station data.

---

## 🏗 Project Structure

```
india-air-quality/
│
├── air_quality_preprocessing.py   ← Main pipeline (Steps 0–10 + Bonus)
├── city_day_clean.csv             ← Cleaned data, original scale (for EDA)
├── city_day_ml.csv                ← Scaled + encoded (ML-ready)
├── India_AQI_EDA.png              ← Full 8-chart EDA dashboard
├── outlier_boxplots.png           ← IQR outlier visualization
├── README.md                      ← You are here
└── dataset/                       ← Extracted ZIP files
    ├── city_day.csv
    ├── city_hour.csv
    ├── station_day.csv
    ├── station_hour.csv
    └── stations.csv
```

---

## ⚙️ Preprocessing Pipeline — Step by Step

### STEP 0 — ZIP Extraction
Raw data arrives as a compressed ZIP. Python's built-in `zipfile` library extracts all 5 CSV files into a `dataset/` folder automatically.

---

### STEP 1 — Load Data
Loaded `city_day.csv` using `pd.read_csv()`. Initial shape: **29,531 rows × 16 columns** covering City, Date, 12 pollutants, AQI, and AQI_Bucket.

---

### STEP 2 — Initial Inspection
Full X-ray using `df.info()`, `df.describe()`, and `df.isnull().sum()`:

| Column | Missing % | Decision |
|---|---|---|
| PM2.5 | 15.6% | `ffill + bfill` per city |
| PM10 | 37.7% | `ffill + bfill` per city |
| NH3 | 35.0% | `ffill + bfill` per city |
| **Xylene** | **61.3%** | **DROPPED** |
| AQI_Bucket | 15.9% | Re-derived from AQI |

---

### STEP 3 — Handle Missing Values

> ⚠️ **Common Mistake:** Using `mean → ffill → median` together. Mean/median ignore time order — a winter reading filled with the annual mean is scientifically wrong.

**Correct approach for time-series environmental data:**

```python
# Sort by City + Date FIRST — time order matters
df.sort_values(["City", "Date"], inplace=True)

# ffill within each city, then bfill for leading NaNs
df[col] = df.groupby("City")[col].transform(lambda x: x.ffill().bfill())
```

- **Xylene (61%)** → Dropped. More than half missing = no sensor coverage.
- **All pollutants** → `ffill()` then `bfill()` within each city group.
- **AQI_Bucket** → Re-derived from AQI using the official CPCB scale.
- **Lucknow PM10** → No sensor installed at all. Filled with global median.

---

### STEP 4 — Duplicates
Checked on `subset=["City", "Date"]` — the logical primary key. Result: **0 duplicates** found.

---

### STEP 5 — Outlier Treatment (IQR Capping)

> ⚠️ **Common Mistake:** Removing outliers in environmental data. AQI = 2049 during Delhi Diwali is **real data** — not an error. Removing it erases the most dangerous days from the dataset.

**Correct approach → IQR Capping (Winsorization):**

```python
def cap_iqr(series, factor=3.0):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    return series.clip(Q1 - factor * IQR, Q3 + factor * IQR)
```

- Factor = **3.0** (not 1.5) — pollution spikes are legitimate, wide fence needed.
- Applied consistently to all pollutant columns + AQI.
- Values are **clipped to the fence**, not deleted.

---

### STEP 6 — Feature Engineering

> ⚠️ **Common Mistake:** `Avg_Pollution = mean(PM2.5, PM10, NO2, SO2, CO)` — these have **different units**. Averaging them is like averaging height (cm) + weight (kg).

**5 engineered features with physical meaning:**

| Feature | Formula | Why |
|---|---|---|
| `Combustion_Index` | `NO2 + CO` | Traffic + industrial burning signal (same units) |
| `Smog_Risk` | `O3 × NO2 / 100` | Photochemical smog forms when sunlight reacts with NOx |
| `PM_Ratio` | `PM2.5 / PM10` | High = vehicle exhaust. Low = dust/construction |
| `Is_Winter` | `Month in {12,1,2}` | Temperature inversion traps pollution near ground |
| `High_Pollution_Day` | `AQI > 200` | Instant label for dangerous air days |

---

### STEP 7 — Encoding

> ⚠️ **Common Mistake:** Using `LabelEncoder` on `AQI_Bucket` — assigns alphabetically: `Good=0, Moderate=3, Poor=2, Satisfactory=4, Severe=1, Very Poor=5`. **Wrong order!**

**Correct approach:**

```python
# Manual ordered mapping — severity matters
bucket_order = {
    "Good": 0, "Satisfactory": 1, "Moderate": 2,
    "Poor": 3, "Very Poor": 4,   "Severe": 5
}
df["Pollution_Level"] = df["AQI_Bucket"].map(bucket_order)

# Season → One-Hot (no natural order between seasons)
df = pd.get_dummies(df, columns=["Season"], prefix="Season", drop_first=True)
```

---

### STEP 8 — Feature Scaling

**StandardScaler** chosen over MinMaxScaler because the data has outliers (pollution spikes). One extreme value would crush MinMaxScaler, pushing all other values toward 0.

```python
# Raw pollutants only — AQI is EXCLUDED (data leakage prevention)
feature_cols = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
                "CO", "SO2", "O3", "Benzene", "Toluene",
                "Combustion_Index", "Smog_Risk", "PM_Ratio"]

scaler = StandardScaler()
df_ml[feature_cols] = scaler.fit_transform(df[feature_cols])
```

> **Never scaled:** Binary columns (`Is_Winter`, `High_Pollution_Day`), OHE Season columns, and the target variable `Pollution_Level`.

---

### STEP 9 — Data Leakage Prevention

**The critical question:** *If you train a model using both AQI and PM2.5/PM10/NO2/SO2/CO/O3 to predict Pollution_Level — what goes wrong?*

**Answer: Data Leakage.** AQI is *mathematically calculated* from those pollutants. Including AQI as a feature means the model sees the answer baked into the input. It will appear to have near-perfect accuracy but will fail completely on real-world data.

**Fix:** AQI is excluded from `feature_cols`. The model learns to infer pollution severity from raw sensor readings only.

---

### STEP 10 — Final Dataset

```
Final ML dataset shape   : 29,531 rows × 23 columns
Missing values           : 0
Duplicate rows           : 0
Target variable          : Pollution_Level (0=Good → 5=Severe)
Outputs                  : city_day_clean.csv + city_day_ml.csv
```

---

## 📈 EDA Dashboard

![India AQI EDA Dashboard](India_AQI_EDA.png)

The dashboard includes:
1. **City Avg AQI Bar** — Ahmedabad leads, not Delhi
2. **AQI Bucket Pie** — Only 4.5% of days are "Good"
3. **Year-wise Trend** — Consistent improvement 2015→2020
4. **Monthly Heatmap** — Dec–Jan worst, July–Aug best (monsoon washes pollution)
5. **PM2.5 Boxplot** — Distribution + spikes per city
6. **Correlation Heatmap** — PM2.5↔PM10 high, O3↔NO2 negative
7. **Seasonal Bar** — Winter 46% worse than non-winter
8. **Delhi Timeline** — COVID lockdown crash clearly visible

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/india-air-quality.git
cd india-air-quality
```

**2. Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

**3. Download the dataset**

Download from [Kaggle](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india) and place the ZIP in the project root:
```
india-air-quality/
└── Air_Quality_Data_in_India__2015_-_2020__archive__4_.zip
```

**4. Run the pipeline**
```bash
python air_quality_preprocessing.py
```

**Outputs generated:**
- `city_day_clean.csv` — clean data, original scale
- `city_day_ml.csv` — scaled + encoded, ML-ready
- `India_AQI_EDA.png` — full EDA dashboard
- `outlier_boxplots.png` — outlier check charts

---

## 🛠 Tech Stack

| Library | Version | Purpose |
|---|---|---|
| `pandas` | 1.5+ | Data loading, cleaning, transformation |
| `numpy` | 1.23+ | Numerical operations |
| `matplotlib` | 3.6+ | Base plotting |
| `seaborn` | 0.12+ | Statistical visualizations |
| `scikit-learn` | 1.1+ | StandardScaler, preprocessing |
| `zipfile` | built-in | ZIP extraction |

---

## 📚 Mistakes Corrected (Learning Log)

| ❌ Mistake | ✅ Fix |
|---|---|
| `mean → ffill → median` imputation mix | `ffill + bfill` within city group (time-series correct) |
| `set_index("Date")` KeyError bug | Convert to datetime first, then set index |
| `set_index` called twice | Call once, check before re-running |
| Removed AQI outliers only, kept PM2.5 | Consistent IQR capping on all pollutant columns |
| Used removal for environmental outliers | IQR Capping — Diwali spikes are real events |
| `LabelEncoder` on AQI_Bucket | Manual ordered mapping (alphabetical ≠ severity) |
| `mean(PM2.5, PM10, NO2, SO2, CO)` | Never average different units — engineered meaningful features instead |
| Included AQI in features | Removed (data leakage — AQI is derived from those pollutants) |

---

## 🔮 Next Steps

- [ ] Build classification model to predict `Pollution_Level` (Random Forest, XGBoost)
- [ ] Handle class imbalance (Very few "Good" days) using SMOTE or class weights
- [ ] Evaluate with accuracy, F1-score, confusion matrix
- [ ] Time-series forecasting — predict next day's AQI from past 7 days
- [ ] Add model evaluation metrics (RMSE, R², accuracy)
- [ ] Deploy as a simple Streamlit dashboard

---

## 👤 Author

**Satish**
Data Science Student — DSCE Bangalore

[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat-square&logo=github)](https://github.com/YOUR_USERNAME)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/YOUR_PROFILE)

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

Dataset credit: [Rohan Rao on Kaggle](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india) | Source: Central Pollution Control Board (CPCB), India.

---

<div align="center">
  <sub>Built with 🔥 dedication | Data Science Journey 2024</sub>
</div>
