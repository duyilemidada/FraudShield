# ml/eda.py
"""
Run exploratory data analysis on your MongoDB transactions.
Usage: python -m ml.eda
Produces text output and saves a basic CSV summary to ml/reports/eda_summary.csv

This is intentionally simple — no matplotlib (no display on headless servers).
Output is designed to be readable in a terminal or copy-pasted into a notebook.

WHAT TO LOOK FOR:
  - Fraud rate: ideally 2–15%. If it's below 1%, your model will struggle.
  - Missing values: if device_fingerprint is always None, remove it.
  - Amount distribution: is it realistic? Uniform distributions indicate synthetic data.
  - Velocity feature coverage: how many transactions actually have non-zero velocity?
"""

import asyncio
import os
import json
import pandas as pd
import numpy as np
import sys

# Add project root to path so relative imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import database.mongo as mongo_module


async def fetch_all():
    data = []
    async for doc in mongo_module.transaction_collection.find({}, {"_id": 0}):
        data.append(doc)
    return data


def run_eda(df: pd.DataFrame):
    print("=" * 60)
    print("FRAUDSHIELD — EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # ── 1. Basic shape ───────────────────────────────────────
    print(f"\n[1] Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"    Columns: {list(df.columns)}")

    # ── 2. Data types ────────────────────────────────────────
    print("\n[2] Data types:")
    for col, dtype in df.dtypes.items():
        print(f"    {col:<30} {str(dtype)}")

    # ── 3. Missing values ────────────────────────────────────
    print("\n[3] Missing values (% of rows):")
    for col in df.columns:
        n_missing = df[col].isna().sum()
        pct = n_missing / len(df) * 100
        if pct > 0:
            print(f"    {col:<30} {n_missing:>6} missing  ({pct:.1f}%)")
    if df.isna().sum().sum() == 0:
        print("    No missing values found.")

    # ── 4. Target distribution ───────────────────────────────
    if 'is_fraud' in df.columns:
        fraud_count = df['is_fraud'].sum()
        total = len(df)
        fraud_rate = fraud_count / total * 100
        print(f"\n[4] Fraud distribution:")
        print(f"    Legitimate: {total - fraud_count:>6} ({100-fraud_rate:.1f}%)")
        print(f"    Fraud:      {fraud_count:>6} ({fraud_rate:.1f}%)")

        # Interpret fraud rate
        if fraud_rate < 1.0:
            print("    ⚠ Fraud rate below 1% — class imbalance is severe.")
            print("      Use class_weight='balanced' and check recall, not accuracy.")
        elif fraud_rate > 30.0:
            print("    ⚠ Fraud rate above 30% — data may be oversampled or synthetic.")
        else:
            print("    ✓ Fraud rate is in a workable range.")
    else:
        print("\n[4] No 'is_fraud' column found — supervised training not possible with this data.")

    # ── 5. Amount statistics ─────────────────────────────────
    if 'amount' in df.columns:
        amt = df['amount']
        print(f"\n[5] Amount statistics:")
        print(f"    Min:    {amt.min():>15,.2f}")
        print(f"    Median: {amt.median():>15,.2f}")
        print(f"    Mean:   {amt.mean():>15,.2f}")
        print(f"    Max:    {amt.max():>15,.2f}")
        print(f"    Std:    {amt.std():>15,.2f}")

        # Check for uniform distribution (a red flag in synthetic data)
        # A uniform distribution has std/mean close to 0.577
        ratio = amt.std() / amt.mean() if amt.mean() != 0 else 0
        if 0.5 < ratio < 0.62:
            print("    ⚠ Amount distribution looks uniform (std/mean ≈ 0.577).")
            print("      Real transactions follow power-law — reseed with weighted amounts.")
        else:
            print(f"    ✓ Amount distribution looks realistic (std/mean = {ratio:.3f}).")

        # Amount by fraud label
        if 'is_fraud' in df.columns:
            print(f"\n    Mean amount — fraud:   {df[df.is_fraud == True]['amount'].mean():>12,.2f}")
            print(f"    Mean amount — legit:   {df[df.is_fraud == False]['amount'].mean():>12,.2f}")

    # ── 6. Categorical distributions ─────────────────────────
    for col in ['payment_method', 'transaction_type', 'currency']:
        if col in df.columns:
            print(f"\n[6] '{col}' value counts:")
            vc = df[col].value_counts()
            for val, count in vc.items():
                bar = "█" * int(count / len(df) * 40)
                print(f"    {str(val):<20} {count:>6}  {bar}")

    # ── 7. Timestamp analysis ────────────────────────────────
    if 'created_at' in df.columns:
        df['created_at'] = pd.to_datetime(df['created_at'], utc=True, errors='coerce')
        valid_ts = df['created_at'].dropna()
        if len(valid_ts) > 0:
            span_days = (valid_ts.max() - valid_ts.min()).days
            print(f"\n[7] Timestamp span: {span_days} days")
            print(f"    Earliest: {valid_ts.min()}")
            print(f"    Latest:   {valid_ts.max()}")

            # Check for timestamp clustering (another synthetic data flag)
            if span_days < 2:
                print("    ⚠ All timestamps within 2 days — velocity features will be sparse.")
                print("      Reseed with --spread-dates flag or adjust seed script.")
            else:
                print("    ✓ Timestamps span enough time for velocity feature training.")

    # ── 8. Velocity feature coverage ─────────────────────────
    velocity_cols = ['txn_count_5min', 'txn_count_1hr', 'txn_count_24hr',
                     'amount_sum_24hr', 'unique_devices_24hr', 'inbound_senders_1hr']
    present = [c for c in velocity_cols if c in df.columns]
    if present:
        print(f"\n[8] Velocity feature coverage (% of rows with non-zero value):")
        for col in present:
            nonzero = (df[col] != 0).sum()
            pct = nonzero / len(df) * 100
            print(f"    {col:<30} {pct:>5.1f}% non-zero")
    else:
        print("\n[8] No velocity features found. Run add_training_velocity_features() first.")

    # ── 9. Device fingerprint coverage ───────────────────────
    if 'device_fingerprint' in df.columns:
        null_fp = df['device_fingerprint'].isna().sum()
        pct = null_fp / len(df) * 100
        print(f"\n[9] Device fingerprint null rate: {pct:.1f}%")
        if pct > 80:
            print("    ⚠ High null rate — unique_devices_24hr feature may be unreliable.")

    # ── 10. Fraud by payment method (if labelled) ────────────
    if 'is_fraud' in df.columns and 'payment_method' in df.columns:
        print("\n[10] Fraud rate by payment_method:")
        grouped = df.groupby('payment_method')['is_fraud'].agg(['sum', 'count'])
        grouped['fraud_rate'] = grouped['sum'] / grouped['count'] * 100
        for pm, row in grouped.iterrows():
            print(f"     {str(pm):<20} {row['fraud_rate']:>5.1f}% fraud  ({int(row['count'])} txns)")

    print("\n" + "=" * 60)
    print("EDA COMPLETE")
    print("=" * 60)

    return df


def main():
    print("Fetching transactions from MongoDB...")
    data = asyncio.run(fetch_all())
    if not data:
        print("No transactions found. Run: python -m ml.seed_synthetic_data")
        return

    df = pd.DataFrame(data)

    # Save raw report dir
    report_dir = os.path.join(os.path.dirname(__file__), 'reports')
    os.makedirs(report_dir, exist_ok=True)

    run_eda(df)

    # Save a quick CSV summary
    summary_path = os.path.join(report_dir, 'eda_summary.csv')
    df.describe(include='all').to_csv(summary_path)
    print(f"\nNumerical summary saved to: {summary_path}")


if __name__ == '__main__':
    main()