import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Load the data
df = pd.read_csv('realistic_drug_labels_side_effects.csv')

print("=== DRUG DATA ANALYSIS FOR MACHINE LEARNING APPLICATIONS ===\n")
print(f"Dataset Shape: {df.shape}")
print(f"Total Records: {len(df)}")

# Basic data info
print("\n1. DATA OVERVIEW:")
print(f"- Unique Drugs: {df['drug_name'].nunique()}")
print(f"- Manufacturers: {df['manufacturer'].nunique()}")
print(f"- Drug Classes: {df['drug_class'].nunique()}")
print(f"- Approval Years: {df['approval_year'].min()} - {df['approval_year'].max()}")

# Approval Status Distribution
print("\n2. DRUG APPROVAL PREDICTION - KEY FINDINGS:")
approval_dist = df['approval_status'].value_counts()
print(f"Approval Status Distribution:")
for status, count in approval_dist.items():
    print(f"  - {status}: {count} ({count/len(df)*100:.1f}%)")

# Features for approval prediction
print("\nKey Features for Approval Prediction:")
print("- Drug Class Impact:")
class_approval = df.groupby('drug_class')['approval_status'].apply(lambda x: (x=='Approved').mean()).sort_values(ascending=False)
print(class_approval.head())

print("\n- Manufacturer Success Rate:")
mfg_approval = df.groupby('manufacturer')['approval_status'].apply(lambda x: (x=='Approved').mean()).sort_values(ascending=False)
print(mfg_approval)

print("\n- Side Effect Severity Impact:")
severity_approval = df.groupby('side_effect_severity')['approval_status'].apply(lambda x: (x=='Approved').mean())
print(severity_approval)

# Price Analysis
print("\n3. PRICE PREDICTION - KEY FINDINGS:")
print(f"Price Range: ${df['price_usd'].min():.2f} - ${df['price_usd'].max():.2f}")
print(f"Average Price: ${df['price_usd'].mean():.2f}")
print(f"Median Price: ${df['price_usd'].median():.2f}")

print("\nPrice by Drug Class (Top 5):")
class_price = df.groupby('drug_class')['price_usd'].mean().sort_values(ascending=False)
print(class_price.head())

print("\nPrice by Manufacturer:")
mfg_price = df.groupby('manufacturer')['price_usd'].mean().sort_values(ascending=False)
print(mfg_price)

print("\nPrice by Dosage Correlation:")
dosage_price_corr = df['dosage_mg'].corr(df['price_usd'])
print(f"Dosage-Price Correlation: {dosage_price_corr:.3f}")

# Side Effect Analysis
print("\n4. SIDE EFFECT CLASSIFICATION - KEY FINDINGS:")
severity_dist = df['side_effect_severity'].value_counts()
print("Side Effect Severity Distribution:")
for severity, count in severity_dist.items():
    print(f"  - {severity}: {count} ({count/len(df)*100:.1f}%)")

# Parse side effects
all_side_effects = []
for effects in df['side_effects'].dropna():
    if isinstance(effects, str):
        effects_list = [effect.strip() for effect in effects.split(',')]
        all_side_effects.extend(effects_list)

side_effect_counts = Counter(all_side_effects)
print(f"\nMost Common Side Effects:")
for effect, count in side_effect_counts.most_common(10):
    print(f"  - {effect}: {count}")

print("\nSide Effect Severity by Drug Class:")
severity_class = pd.crosstab(df['drug_class'], df['side_effect_severity'], normalize='index')
print(severity_class.round(3))

# Market Success Analysis
print("\n5. MARKET SUCCESS ANALYSIS - KEY FINDINGS:")

# Define success metrics
df['market_success'] = ((df['approval_status'] == 'Approved') & 
                       (df['price_usd'] > df['price_usd'].median())).astype(int)

success_rate = df['market_success'].mean()
print(f"Overall Market Success Rate: {success_rate:.3f}")

print("\nSuccess Factors:")
print("- Drug Class Success Rate:")
class_success = df.groupby('drug_class')['market_success'].mean().sort_values(ascending=False)
print(class_success.head())

print("\n- Manufacturer Success Rate:")
mfg_success = df.groupby('manufacturer')['market_success'].mean().sort_values(ascending=False)
print(mfg_success)

print("\n- Administration Route Impact:")
route_success = df.groupby('administration_route')['market_success'].mean().sort_values(ascending=False)
print(route_success)

# Feature Engineering Insights
print("\n6. FEATURE ENGINEERING RECOMMENDATIONS:")
print("For Drug Approval Prediction:")
print("- Use drug_class, manufacturer, side_effect_severity as categorical features")
print("- Create binary features for common side effects")
print("- Use dosage_mg as numerical feature")
print("- Consider approval_year for temporal trends")

print("\nFor Price Prediction:")
print("- Strong predictors: drug_class, manufacturer, dosage_mg")
print("- Create interaction features between class and manufacturer")
print("- Use administration_route as categorical feature")
print("- Consider side_effect_severity impact on pricing")

print("\nFor Side Effect Classification:")
print("- Use drug_class, dosage_mg, administration_route as features")
print("- Create features from contraindications and warnings")
print("- Consider manufacturer quality patterns")

print("\nFor Market Success Analysis:")
print("- Combine approval status and price for success metric")
print("- Use drug_class, manufacturer, administration_route")
print("- Include side_effect_severity as risk factor")
print("- Consider temporal trends with approval_year")

# Data Quality Assessment
print("\n7. DATA QUALITY FOR ML:")
print("Missing Values:")
print(df.isnull().sum())

print(f"\nDuplicate Records: {df.duplicated().sum()}")
print(f"Unique Drug Names: {df['drug_name'].nunique()} (vs {len(df)} total records)")

print("\nRecommended Data Preprocessing:")
print("- Handle categorical variables with encoding")
print("- Normalize price_usd and dosage_mg")
print("- Parse and vectorize side_effects text")
print("- Create binary features for contraindications/warnings")
print("- Consider stratified sampling for imbalanced classes")