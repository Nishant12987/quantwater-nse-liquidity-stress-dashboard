# -------------------------------------------------
# SAVE MODEL (optional for research)
# -------------------------------------------------

model.save(
    MODEL_DIR / "lstm_stress_forecaster.keras"
)

print("LSTM model + scaler saved successfully.")


# -------------------------------------------------
#  NEW: GENERATE & SAVE PREDICTIONS (IMPORTANT)
# -------------------------------------------------

# Combine train + test for full predictions
full_df = pd.concat([train_df, test_df]).sort_values("DATE").reset_index(drop=True)

# Apply scaler again to full data (IMPORTANT)
full_df_scaled = full_df.copy()
full_df_scaled[FEATURES] = scaler.transform(full_df[FEATURES])

# Build sequences for full dataset
X_full, _ = build_sequences(full_df_scaled, FEATURES, LOOKBACK)

# Generate predictions
lstm_preds = model.predict(X_full).flatten()

# Align with original dataframe
pred_df = full_df.iloc[LOOKBACK:].copy()

pred_df["LSTM_SCORE"] = lstm_preds

# Keep only required columns
pred_df = pred_df[["DATE", "SYMBOL", "LSTM_SCORE"]]

# Save CSV
pred_df.to_csv(
    MODEL_DIR / "lstm_predictions.csv",
    index=False
)

print("✅ LSTM predictions saved at:", MODEL_DIR / "lstm_predictions.csv")
