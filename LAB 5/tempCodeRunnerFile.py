# A1: Single-Feature Linear Regression
X_single = df_clean[['F1','F2','F3','F4']]
y_single = df_clean['CallerAge']
X_train_single, X_test_single, y_train_single, y_test_single = train_test_split(X_single, y_single, test_size=0.2, random_state=42)
lin_reg = LinearRegression()
lin_reg.fit(X_train_single, y_train_single)
y_train_pred_single = lin_reg.predict(X_train_single)
y_test_pred_single = lin_reg.predict(X_test_single)
print(f"Single-Feature Regression Coefficient: {lin_reg.coef_[0]:.6f}")
print(f"Intercept: {lin_reg.intercept_:.6f}")