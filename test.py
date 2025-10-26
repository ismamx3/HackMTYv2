import pandas as pd
import numpy as np

# For machine learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("[v1] Starting Consumption Prediction Model (fixed & hardened)...")

# -------------------------------
# 1) Load the dataset
# -------------------------------
DATA_URL = "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/%5BHackMTY2025%5D_ConsumptionPrediction_Dataset_v1-YV0ATg1sY9Uc4PHBWDN9lpDJ4mumwK.csv"

print("[v1] Loading dataset from URL...")
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
df = pd.read_csv(DATA_URL)

# -------------------------------
# 2) Early parsing & numeric coerce
# -------------------------------
# Parse Date BEFORE any overview
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Force numeric for critical columns (coerce to NaN if text)
for c in ['Passenger_Count', 'Standard_Specification_Qty', 'Quantity_Consumed',
          'Quantity_Returned', 'Unit_Cost']:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

print(f"[v1] Dataset loaded successfully! Shape: {df.shape}")
print(f"[v1] Columns: {df.columns.tolist()}")

# -------------------------------
# 3) Overview (safe)
# -------------------------------
print("\n" + "="*60)
print("DATASET OVERVIEW")
print("="*60)
print(f"Total records: {len(df)}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"Unique flights: {df['Flight_ID'].nunique()}")
print(f"Unique products: {df['Product_ID'].nunique()}")
# Origins print (cap to avoid flooding console)
origins_list = df['Origin'].dropna().unique()
print(f"Origins count: {len(origins_list)}")
print(f"Origins sample: {origins_list[:20]}")

# -------------------------------
# 4) Preprocessing & Feature Engineering (safe)
# -------------------------------
print("\n[v1] Preprocessing data...")

# Calendar features
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['Month']     = df['Date'].dt.month
df['DayOfMonth']= df['Date'].dt.day

# Safe ratios (avoid divide-by-zero/inf)
spec = df['Standard_Specification_Qty']
cons = df['Quantity_Consumed']
retu = df['Quantity_Returned']

df['Consumption_Rate'] = (cons / spec).replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 1)

pax = df['Passenger_Count'].replace(0, np.nan)
df['Qty_Per_Passenger']      = (spec / pax).replace([np.inf, -np.inf], np.nan).fillna(0)
df['Consumed_Per_Passenger'] = (cons / pax).replace([np.inf, -np.inf], np.nan).fillna(0)

# Crew feedback signals
df['Has_Feedback'] = df['Crew_Feedback'].notna().astype(int)
df['Ran_Out']      = df['Crew_Feedback'].fillna('').str.contains('ran out', case=False).astype(int)
df['Low_Demand']   = df['Crew_Feedback'].fillna('').str.contains('low demand', case=False).astype(int)

# Encode categorical variables (LabelEncoder; árboles lo soportan)
label_encoders = {}
categorical_cols = ['Origin', 'Flight_Type', 'Service_Type', 'Product_ID', 'Product_Name']
for col in categorical_cols:
    le = LabelEncoder()
    df[f'{col}_Encoded'] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

print("[v1] Feature engineering completed!")

# -------------------------------
# 5) EDA (safe)
# -------------------------------
print("\n" + "="*60)
print("KEY INSIGHTS")
print("="*60)

avg_consumption_rate = df['Consumption_Rate'].mean()
print(f"\nAverage Consumption Rate: {avg_consumption_rate:.2%}")
print(f"Median Consumption Rate: {df['Consumption_Rate'].median():.2%}")

total_waste = df['Quantity_Returned'].sum()
total_prepared = df['Standard_Specification_Qty'].sum()
waste_rate = (total_waste / total_prepared) if total_prepared > 0 else 0.0
print(f"\nTotal Waste Rate: {waste_rate:.2%}")
print(f"Total items returned: {total_waste:,.0f}")
print(f"Total items prepared: {total_prepared:,.0f}")

print("\n--- Consumption by Flight Type ---")
flight_type_stats = df.groupby('Flight_Type').agg({
    'Consumption_Rate': 'mean',
    'Quantity_Returned': 'sum',
    'Quantity_Consumed': 'sum'
}).round(3)
print(flight_type_stats)

print("\n--- Consumption by Service Type ---")
service_type_stats = df.groupby('Service_Type').agg({
    'Consumption_Rate': 'mean',
    'Quantity_Returned': 'sum',
    'Quantity_Consumed': 'sum'
}).round(3)
print(service_type_stats)

print("\n--- Consumption by Origin ---")
origin_stats = df.groupby('Origin').agg({
    'Consumption_Rate': 'mean',
    'Quantity_Returned': 'sum',
    'Flight_ID': 'nunique'
}).round(3)
origin_stats.columns = ['Avg_Consumption_Rate', 'Total_Returned', 'Num_Flights']
print(origin_stats)

print("\n--- Top 10 Most Consumed Products ---")
top_products = df.groupby('Product_Name')['Quantity_Consumed'].sum().sort_values(ascending=False).head(10)
print(top_products)

print("\n--- Top 10 Products with Highest Waste ---")
waste_products = df.groupby('Product_Name')['Quantity_Returned'].sum().sort_values(ascending=False).head(10)
print(waste_products)

# -------------------------------
# 6) Modeling — time-based split
# -------------------------------
print("\n" + "="*60)
print("BUILDING PREDICTION MODEL (time-based split)")
print("="*60)

# Feature set (incluye feedback signals y Product_Name_Encoded)
feature_cols = [
    'Origin_Encoded',
    'Flight_Type_Encoded',
    'Service_Type_Encoded',
    'Product_ID_Encoded',
    'Product_Name_Encoded',
    'Passenger_Count',
    'Standard_Specification_Qty',
    'DayOfWeek',
    'Month',
    'Unit_Cost',
    'Qty_Per_Passenger',
    'Has_Feedback',
    'Ran_Out',
    'Low_Demand'
]

# Drop rows with missing essentials in feature/target
needed = feature_cols + ['Quantity_Consumed', 'Date']
df_model = df.dropna(subset=[c for c in needed if c in df.columns]).copy()

# Time-based split: last 20% by Date
df_sorted = df_model.sort_values('Date')
cut = int(len(df_sorted) * 0.8)
train_df = df_sorted.iloc[:cut]
test_df  = df_sorted.iloc[cut:]

X_train, y_train = train_df[feature_cols], train_df['Quantity_Consumed']
X_test,  y_test  = test_df[feature_cols],  test_df['Quantity_Consumed']

print(f"[v1] Training set size: {len(X_train)}")
print(f"[v1] Test set size: {len(X_test)}")

# Train Random Forest model
print("\n[v1] Training Random Forest model...")
rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# -------------------------------
# 7) Evaluation
# -------------------------------
y_pred_train = rf_model.predict(X_train)
y_pred_test  = rf_model.predict(X_test)

def mape_safe(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

print("\n--- Model Performance ---")
train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae  = mean_absolute_error(y_test,  y_pred_test)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse  = np.sqrt(mean_squared_error(y_test,  y_pred_test))
train_r2 = r2_score(y_train, y_pred_train)
test_r2  = r2_score(y_test,  y_pred_test)
train_mape = mape_safe(y_train, y_pred_train)
test_mape  = mape_safe(y_test,  y_pred_test)

print(f"\nTraining Set:")
print(f"  MAE:  {train_mae:.2f} items")
print(f"  RMSE: {train_rmse:.2f} items")
print(f"  R²:   {train_r2:.4f}")
print(f"  MAPE: {train_mape:.2f}%")

print(f"\nTest Set (time-based):")
print(f"  MAE:  {test_mae:.2f} items")
print(f"  RMSE: {test_rmse:.2f} items")
print(f"  R²:   {test_r2:.4f}")
print(f"  MAPE: {test_mape:.2f}%")

# Feature importance
print("\n--- Top 10 Most Important Features ---")
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)
for _, row in feature_importance.head(10).iterrows():
    print(f"{row['Feature']}: {row['Importance']:.4f}")

# -------------------------------
# 8) Prediction examples (safe accuracy)
# -------------------------------
print("\n" + "="*60)
print("PREDICTION EXAMPLES")
print("="*60)

comparison_df = pd.DataFrame({
    'Actual': y_test.values[:10],
    'Predicted': y_pred_test[:10].round(0),
})
comparison_df['Difference'] = (comparison_df['Actual'] - comparison_df['Predicted']).round(0)
comparison_df['Accuracy'] = np.where(
    comparison_df['Actual'] > 0,
    (1 - (comparison_df['Difference'].abs() / comparison_df['Actual'])).clip(0, 1),
    np.nan
)
print("\nSample Predictions (first 10 test cases):")
print(comparison_df.to_string(index=False))

# -------------------------------
# 9) Waste Reduction Potential (aligned indices + guards)
# -------------------------------
print("\n" + "="*60)
print("WASTE REDUCTION POTENTIAL")
print("="*60)

# Align test rows with predictions using labels (NOT positions)
df_test = test_df.copy()  # already aligned with y_test due to the split above

# Ensure Unit_Cost numeric with median imputation
if 'Unit_Cost' in df_test.columns:
    df_test['Unit_Cost'] = pd.to_numeric(df_test['Unit_Cost'], errors='coerce')
    df_test['Unit_Cost'] = df_test['Unit_Cost'].fillna(df_test['Unit_Cost'].median())

df_test['Predicted_Consumption'] = y_pred_test
# Policy: 10% buffer, lower bound 0, and cap to current Standard_Specification_Qty
df_test['Optimal_Specification'] = np.ceil(y_pred_test * 1.10)
df_test['Optimal_Specification'] = np.clip(df_test['Optimal_Specification'], a_min=0, a_max=None)
df_test['Optimal_Specification'] = np.minimum(
    df_test['Optimal_Specification'],
    df_test['Standard_Specification_Qty']
)

current_waste = df_test['Quantity_Returned'].sum()
predicted_waste = (df_test['Optimal_Specification'] - df_test['Quantity_Consumed']).clip(lower=0).sum()
waste_reduction = current_waste - predicted_waste
waste_reduction_pct = (waste_reduction / current_waste) * 100 if current_waste > 0 else 0.0

print(f"\nCurrent waste (test set): {current_waste:,.0f} items")
print(f"Predicted waste with model: {predicted_waste:,.0f} items")
print(f"Potential waste reduction: {waste_reduction:,.0f} items ({waste_reduction_pct:.1f}%)")

# Cost savings
if 'Unit_Cost' in df_test.columns:
    total_cost_current = (df_test['Quantity_Returned'] * df_test['Unit_Cost']).sum()
    total_cost_optimized = ((df_test['Optimal_Specification'] - df_test['Quantity_Consumed']).clip(lower=0) * df_test['Unit_Cost']).sum()
    cost_savings = total_cost_current - total_cost_optimized

    print(f"\nCurrent waste cost: ${total_cost_current:,.2f}")
    print(f"Optimized waste cost: ${total_cost_optimized:,.2f}")
    print(f"Potential cost savings: ${cost_savings:,.2f}")
else:
    print("\nUnit_Cost not available; skipping cost savings computation.")

# -------------------------------
# 10) Actionable recommendations
# -------------------------------
print("\n" + "="*60)
print("ACTIONABLE RECOMMENDATIONS")
print("="*60)

product_analysis = df.groupby('Product_Name').agg({
    'Consumption_Rate': 'mean',
    'Quantity_Returned': 'sum',
    'Standard_Specification_Qty': 'sum',
    'Flight_ID': 'count'
}).round(3)
product_analysis.columns = ['Avg_Consumption_Rate', 'Total_Returned', 'Total_Prepared', 'Num_Flights']
product_analysis['Waste_Rate'] = np.where(
    product_analysis['Total_Prepared'] > 0,
    product_analysis['Total_Returned'] / product_analysis['Total_Prepared'],
    0.0
)

print("\n--- Products with Highest Over-Provisioning (>40% waste) ---")
high_waste = product_analysis[product_analysis['Waste_Rate'] > 0.4].sort_values('Total_Returned', ascending=False)
if len(high_waste) > 0:
    print(high_waste.head(10))
    print("\n💡 Recommendation: Reduce standard specifications for these products")
else:
    print("No products with >40% waste rate found")

print("\n--- Products Running Out (from crew feedback) ---")
ran_out_products = df[df['Ran_Out'] == 1].groupby('Product_Name').size().sort_values(ascending=False)
if len(ran_out_products) > 0:
    print(ran_out_products.head(10))
    print("\n💡 Recommendation: Increase standard specifications for these products")
else:
    print("No products reported as running out")

# -------------------------------
# 11) Model Summary
# -------------------------------
print("\n" + "="*60)
print("MODEL SUMMARY")
print("="*60)
print(f"""
✅ Model trained successfully with {len(X_train)} samples
✅ Test (time-based) R²: {test_r2:.2%}
✅ Test MAE: ±{test_mae:.1f} items | RMSE: {test_rmse:.1f} | MAPE: {test_mape:.2f}%
✅ Potential waste reduction (test): {waste_reduction_pct:.1f}%

Next Steps:
1) Deploy to predict consumption for upcoming flights (API/cron).
2) Adjust standard specs using Recommended (Optimal_Specification).
3) Monitor actual vs predicted and re-entrenar con feedback de tripulación.
4) Agregar Product_Category (Snack/Drink/Meal) y cuantiles (P50/P90) para buffers más finos.
""")

print("\n[v1] Consumption Prediction Model completed successfully!")

import joblib

# Guardar el modelo entrenado y los label encoders
model_data = {
    'model': rf_model,
    'label_encoders': label_encoders,
    'feature_cols': feature_cols
}
joblib.dump(model_data, "modelo_consumo.pkl")
print("✅ Modelo y encoders guardados como 'modelo_consumo.pkl'")

