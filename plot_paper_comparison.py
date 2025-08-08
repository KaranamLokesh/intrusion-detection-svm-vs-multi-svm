import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Your results (from previous message)
methods = [
    "OneVsOne",
    "Pairwise Meta",
    "Balanced Meta",
    "Enhanced Meta",
    "Focused Minority",
]
accuracy = [0.749, 0.751, 0.754, 0.768, 0.775]
precision = [0.785, 0.801, 0.819, 0.787, 0.794]
recall = [0.749, 0.751, 0.754, 0.768, 0.775]
f1 = [0.701, 0.704, 0.726, 0.730, 0.742]
# Per-class recall for R2L (class 3) and U2R (class 4)
r2l_recall = [0.01, 0.01, 0.11, 0.07, 0.10]
u2r_recall = [0.09, 0.10, 0.21, 0.10, 0.15]

# 1. Bar plot: Overall metrics
plt.figure(figsize=(10, 6))
bar_width = 0.18
x = np.arange(len(methods))
plt.bar(x - 1.5 * bar_width, accuracy, width=bar_width, label="Accuracy")
plt.bar(x - 0.5 * bar_width, precision, width=bar_width, label="Precision")
plt.bar(x + 0.5 * bar_width, recall, width=bar_width, label="Recall")
plt.bar(x + 1.5 * bar_width, f1, width=bar_width, label="F1 Score")
plt.xticks(x, methods, rotation=20)
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Overall Performance Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("svm_overall_comparison.png")
plt.show()

# 2. Grouped bar plot: Per-class recall for R2L and U2R
plt.figure(figsize=(10, 6))
plt.bar(x - bar_width / 2, r2l_recall, width=bar_width, label="R2L Recall")
plt.bar(x + bar_width / 2, u2r_recall, width=bar_width, label="U2R Recall")
plt.xticks(x, methods, rotation=20)
plt.ylim(0, 0.25)
plt.ylabel("Recall")
plt.title("Minority Class Recall (R2L & U2R)")
plt.legend()
plt.tight_layout()
plt.savefig("svm_minority_recall.png")
plt.show()

# 3. Show selected extracted paper figures side-by-side with your results
# Pick two relevant images (update filenames as needed)
paper_figs = [
    "extracted_figures/Intrusion_Detection_Mechanism_for_Large_Scale_Networks_using_CNN-LSTM_page5_img1.png",
    "extracted_figures/Classifying_Cover_Crop_Residue_from_RGB_Images_A_Simple_SVM_versus_a_SVM_Ensemble_page3_img1.png",
]

for fig_path in paper_figs:
    if os.path.exists(fig_path):
        img = Image.open(fig_path)
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Extracted Figure: {os.path.basename(fig_path)}")
        plt.tight_layout()
        plt.show()
    else:
        print(f"Image not found: {fig_path}")

print("Charts and paper figures displayed. Saved as PNGs in the current directory.")
