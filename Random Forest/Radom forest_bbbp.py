import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import joblib

train_df = pd.read_csv('bbbp_gen_test.csv')
valid_df = pd.read_csv('bbbp_gen_valid.csv')
test_df = pd.read_csv('bbbp_gen_train.csv')

feature_cols = ['LogP', 'MW', 'HBD', 'HBA','TPSA', 'AroAtoms']

#DAPO 'LogP', 'MolecularWeight', 'HBD', 'HBA',"Charge",'PolarSurfaceArea', 'AromaticAtoms' AUC-ROC:0.7100 #Best CV AUC-ROC: 0.8630
#GRPO 'LogP', 'MolecularWeight', 'HBD', 'HBA','PolarSurfaceArea', 'AromaticAtoms' AUC-ROC: 0.7183 #Best CV AUC-ROC: 0.8651

X_train = train_df[feature_cols]
X_valid = valid_df[feature_cols]
X_test = test_df[feature_cols]

y_train = train_df['p_np'].astype(int)
y_valid = valid_df['p_np'].astype(int)
y_test = test_df['p_np'].astype(int)


X_full = pd.concat([X_train, X_valid], axis=0)
y_full = pd.concat([y_train, y_valid], axis=0)


param_grid = {
    'n_estimators': [100, 120, 140, 160, 180, 200],
    'max_depth': [10, 12, 14, 16, 18, 20],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=196),
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_input, y_input)

print(f"✅ Best Params: {grid_search.best_params_}")
print(f"✅ Best CV AUC-ROC: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_


y_valid_prob = best_model.predict_proba(X_valid)[:, 1]
y_valid_pred = best_model.predict(X_valid)

val_acc = accuracy_score(y_valid, y_valid_pred)
val_auc = roc_auc_score(y_valid, y_valid_prob)

print(f"📊 Validation Accuracy: {val_acc:.4f}")
print(f"📊 Validation AUC-ROC: {val_auc:.4f}")

# 最佳阈值
fpr_v, tpr_v, thresholds_v = roc_curve(y_valid, y_valid_prob)
opt_idx_v = np.argmax(tpr_v - fpr_v)
opt_thresh_v = thresholds_v[opt_idx_v]
print(f"✨ Optimal Threshold on Validation: {opt_thresh_v:.4f}")


y_test_prob = best_model.predict_proba(X_test)[:, 1]
y_test_pred_opt = (y_test_prob >= opt_thresh_v).astype(int)

test_acc = accuracy_score(y_test, y_test_pred_opt)
test_auc = roc_auc_score(y_test, y_test_prob)

print(f"✅ Test Accuracy (Optimized Threshold): {test_acc:.4f}")
print(f"✅ Test AUC-ROC: {test_auc:.4f}")


fpr_t, tpr_t, _ = roc_curve(y_test, y_test_prob)

plt.figure(figsize=(8,6))
plt.plot(fpr_t, tpr_t, label=f'AUC = {test_auc:.2f}', lw=2)
plt.plot([0,1], [0,1], linestyle='--', color='gray')
plt.scatter(fpr_t[opt_idx_v], tpr_t[opt_idx_v], color='red', label=f'Optimal Threshold ({opt_thresh_v:.2f})')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Test Set)")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()


feat_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': best_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=feat_importance)
plt.title('Feature Importance in BBBP Task')
plt.tight_layout()
plt.savefig('bbbp_attributes_important.pdf', format='pdf', bbox_inches='tight')
plt.show()

joblib.dump(best_model, 'best_rf_model_bbbp.pkl')
print("✅ Model saved to 'best_rf_model_bbbp.pkl'")


