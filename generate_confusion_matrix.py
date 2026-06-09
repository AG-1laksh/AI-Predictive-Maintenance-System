"""
Generate Confusion Matrix visualization from predictions
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (12, 5)

# Load predictions
pred_file = Path("outputs/predictions/test_all_fd_predictions.csv")
df = pd.read_csv(pred_file)

print("Data shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nFirst few rows:")
print(df.head())

# Get actual and predicted values
y_true = df['actual_failure'].values
y_pred = df['predicted_failure'].values

# Calculate confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("\n=== CONFUSION MATRIX ===")
print(cm)

# Get metrics
tn, fp, fn, tp = cm.ravel()
print(f"\nTrue Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Positives (TP): {tp}")

# Calculate rates
print(f"\nAccuracy: {(tp + tn) / (tp + tn + fp + fn):.4f}")
print(f"Precision: {tp / (tp + fp):.4f}")
print(f"Recall/Sensitivity: {tp / (tp + fn):.4f}")
print(f"Specificity: {tn / (tn + fp):.4f}")

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Normal', 'Failure'],
            yticklabels=['Normal', 'Failure'],
            cbar_kws={'label': 'Count'})
axes[0].set_title('Confusion Matrix - All_FD Test Set', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Actual', fontsize=12)
axes[0].set_xlabel('Predicted', fontsize=12)

# Bar chart of confusion components
labels = ['True\nPositive\n(TP)', 'True\nNegative\n(TN)', 'False\nPositive\n(FP)', 'False\nNegative\n(FN)']
values = [tp, tn, fp, fn]
colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']

axes[1].bar(labels, values, color=colors, edgecolor='black', linewidth=1.5)
axes[1].set_title('Confusion Matrix Components', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_ylim(0, max(values) * 1.1)

# Add value labels on bars
for i, (label, val) in enumerate(zip(labels, values)):
    axes[1].text(i, val + 500, str(val), ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: outputs/confusion_matrix.png")

# Classification Report
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_true, y_pred, target_names=['Normal', 'Failure']))

# Save detailed metrics to CSV
metrics_data = {
    'Metric': ['True Positives (TP)', 'True Negatives (TN)', 'False Positives (FP)', 'False Negatives (FN)',
               'Accuracy', 'Precision', 'Recall/Sensitivity', 'Specificity', 'F1-Score'],
    'Value': [
        tp, tn, fp, fn,
        (tp + tn) / (tp + tn + fp + fn),
        tp / (tp + fp),
        tp / (tp + fn),
        tn / (tn + fp),
        2 * (tp / (tp + fp)) * (tp / (tp + fn)) / ((tp / (tp + fp)) + (tp / (tp + fn)))
    ]
}

metrics_df = pd.DataFrame(metrics_data)
metrics_df.to_csv('outputs/confusion_matrix_metrics.csv', index=False)
print("\n✓ Saved: outputs/confusion_matrix_metrics.csv")

print("\n" + "="*50)
print("SUCCESS: Confusion matrix generated!")
print("="*50)
