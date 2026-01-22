"""
Clean messy_surveillance_data.csv and document all transformations.
"""

import pandas as pd
import numpy as np
from pathlib import Path

root = Path(__file__).parent.parent

# Load raw data
print("=" * 60)
print("SURVEILLANCE DATA CLEANING LOG")
print("=" * 60)

df = pd.read_csv(root / 'data/messy_surveillance_data.csv', dtype=str)
print(f"\nLoaded {len(df)} rows from messy_surveillance_data.csv")
print(f"Columns: {list(df.columns)}")

# Track all transformations
transformations = []

# ============================================================================
# 1. CLEAN DATE COLUMN
# ============================================================================
print("\n" + "-" * 60)
print("1. CLEANING DATE COLUMN")
print("-" * 60)

original_dates = df['date'].copy()

# Parse mixed date formats
df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=False)

# Log which rows had non-standard formats
date_changes = []
for i, (orig, new) in enumerate(zip(original_dates, df['date'])):
    expected_format = new.strftime('%Y-%m-%d')
    if orig != expected_format:
        date_changes.append({'row': i+2, 'original': orig, 'cleaned': expected_format})

print(f"Standardized {len(date_changes)} dates to ISO 8601 (YYYY-MM-DD):")
for change in date_changes:
    print(f"  Row {change['row']}: '{change['original']}' → '{change['cleaned']}'")

transformations.append(f"Dates: Converted {len(date_changes)} non-standard formats to ISO 8601")

# ============================================================================
# 2. CLEAN REGION COLUMN
# ============================================================================
print("\n" + "-" * 60)
print("2. CLEANING REGION COLUMN")
print("-" * 60)

original_regions = df['region'].copy()

# Normalize: lowercase, strip whitespace, standardize naming
df['region'] = (df['region']
                .str.strip()
                .str.lower()
                .replace('central region', 'central')
                .str.title())  # Back to title case for consistency

region_changes = []
for i, (orig, new) in enumerate(zip(original_regions, df['region'])):
    if orig != new:
        region_changes.append({'row': i+2, 'original': orig, 'cleaned': new})

print(f"Normalized {len(region_changes)} region values:")
for change in region_changes:
    print(f"  Row {change['row']}: '{change['original']}' → '{change['cleaned']}'")

transformations.append(f"Region: Normalized {len(region_changes)} inconsistent values to 'Central'")

# ============================================================================
# 3. CLEAN CASES COLUMN
# ============================================================================
print("\n" + "-" * 60)
print("3. CLEANING CASES COLUMN")
print("-" * 60)

original_cases = df['cases'].copy()

# Step 3a: Convert missing value representations to NaN
missing_representations = ['NA', 'NULL', 'N/A', 'na', 'null', '', ' ']
df['cases'] = df['cases'].replace(missing_representations, np.nan)

# Step 3b: Convert to numeric
df['cases'] = pd.to_numeric(df['cases'], errors='coerce')

# Log missing value conversions
missing_conversions = []
for i, (orig, new) in enumerate(zip(original_cases, df['cases'])):
    if pd.isna(new) and orig not in [np.nan, None]:
        if str(orig).strip() in ['NA', 'NULL', '']:
            missing_conversions.append({'row': i+2, 'original': orig if orig else '(empty)', 'reason': 'missing value representation'})

print(f"Converted {len(missing_conversions)} missing value representations to NaN:")
for conv in missing_conversions:
    print(f"  Row {conv['row']}: '{conv['original']}' → NaN ({conv['reason']})")

transformations.append(f"Cases: Converted {len(missing_conversions)} missing values (NA/NULL/empty) to NaN")

# Step 3c: Handle outliers
# Flag impossible/implausible values
outlier_log = []

# Negative values (impossible)
negative_mask = df['cases'] < 0
for idx in df[negative_mask].index:
    row_num = idx + 2
    orig_val = df.loc[idx, 'cases']
    outlier_log.append({'row': row_num, 'value': orig_val, 'reason': 'negative (impossible)', 'action': 'set to NaN'})
    df.loc[idx, 'cases'] = np.nan

# Implausible spikes (9999, 8888 look like placeholders)
# Using threshold based on data: max realistic value around 250
placeholder_mask = df['cases'] > 500
for idx in df[placeholder_mask].index:
    row_num = idx + 2
    orig_val = df.loc[idx, 'cases']
    outlier_log.append({'row': row_num, 'value': orig_val, 'reason': 'implausible (likely placeholder)', 'action': 'set to NaN'})
    df.loc[idx, 'cases'] = np.nan

print(f"\nHandled {len(outlier_log)} outliers:")
for out in outlier_log:
    print(f"  Row {out['row']}: {out['value']:.0f} → NaN ({out['reason']})")

transformations.append(f"Cases: Removed {len(outlier_log)} outliers (negative values, placeholder codes >500)")

# ============================================================================
# 4. CLEAN POPULATION COLUMN
# ============================================================================
print("\n" + "-" * 60)
print("4. CLEANING POPULATION COLUMN")
print("-" * 60)

original_pop = df['pop'].copy()

# Step 4a: Handle text value
text_to_num = {'five hundred thousand': '500000'}
df['pop'] = df['pop'].replace(text_to_num)

# Step 4b: Handle missing representations
df['pop'] = df['pop'].replace(missing_representations, np.nan)

# Step 4c: Convert to numeric (handles scientific notation automatically)
df['pop'] = pd.to_numeric(df['pop'], errors='coerce')

pop_changes = []
for i, (orig, new) in enumerate(zip(original_pop, df['pop'])):
    orig_str = str(orig).strip() if pd.notna(orig) else ''
    if orig_str and orig_str != str(int(new)) if pd.notna(new) else True:
        if orig_str in text_to_num:
            pop_changes.append({'row': i+2, 'original': orig, 'cleaned': int(new), 'reason': 'text to numeric'})
        elif 'E' in orig_str.upper():
            pop_changes.append({'row': i+2, 'original': orig, 'cleaned': int(new), 'reason': 'scientific notation'})
        elif orig_str in ['NA', 'NULL', '']:
            pop_changes.append({'row': i+2, 'original': orig if orig else '(empty)', 'cleaned': 'NaN', 'reason': 'missing value'})

print(f"Cleaned {len(pop_changes)} population values:")
for change in pop_changes:
    print(f"  Row {change['row']}: '{change['original']}' → {change['cleaned']} ({change['reason']})")

transformations.append(f"Population: Fixed {len(pop_changes)} values (text, scientific notation, missing)")

# ============================================================================
# 5. FINAL DTYPE CONVERSIONS
# ============================================================================
print("\n" + "-" * 60)
print("5. FINAL DATA TYPES")
print("-" * 60)

# Convert cases to nullable integer (Int64 allows NaN)
df['cases'] = df['cases'].astype('Int64')

# Convert pop to nullable integer
df['pop'] = df['pop'].astype('Int64')

# Format date as string for CSV
df['date'] = df['date'].dt.strftime('%Y-%m-%d')

print(f"  date:   string (ISO 8601)")
print(f"  region: string")
print(f"  cases:  Int64 (nullable integer)")
print(f"  pop:    Int64 (nullable integer)")

# ============================================================================
# 6. SAVE CLEANED DATA
# ============================================================================
print("\n" + "-" * 60)
print("6. SAVING CLEANED DATA")
print("-" * 60)

output_path = root / 'data/cleaned_surveillance.csv'
df.to_csv(output_path, index=False)
print(f"Saved to: {output_path}")

# ============================================================================
# 7. SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("TRANSFORMATION SUMMARY")
print("=" * 60)

for i, t in enumerate(transformations, 1):
    print(f"{i}. {t}")

print(f"\nData quality:")
print(f"  Total rows: {len(df)}")
print(f"  Missing cases: {df['cases'].isna().sum()} ({df['cases'].isna().sum()/len(df)*100:.1f}%)")
print(f"  Missing population: {df['pop'].isna().sum()} ({df['pop'].isna().sum()/len(df)*100:.1f}%)")
print(f"  Valid rows (no missing): {(~df['cases'].isna() & ~df['pop'].isna()).sum()}")

# Show first few rows of cleaned data
print("\n" + "-" * 60)
print("PREVIEW OF CLEANED DATA")
print("-" * 60)
print(df.head(15).to_string(index=False))
