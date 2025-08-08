# %%
from itertools import combinations

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print(
        "Warning: matplotlib/seaborn not available. Install with: pip install matplotlib seaborn"
    )

# Replace RFECV with simpler feature selection
from sklearn.feature_selection import SelectKBest, f_classif

try:
    from imblearn.over_sampling import SMOTE

    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

# %%
# 1. Load and preprocess NSL-KDD dataset (minimal, adapted from ids_svm.py)
COLUMNS = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
]


def load_nsl_kdd():
    train_set = pd.read_csv("KDD/KDDTrain+_2.csv", header=None, names=COLUMNS)
    test_set = pd.read_csv("KDD/KDDTest+_2.csv", header=None, names=COLUMNS)
    return train_set, test_set


# %%
def preprocess_nsl_kdd(train_set: pd.DataFrame, test_set: pd.DataFrame):
    cat_cols = ["protocol_type", "service", "flag"]
    # Label encode categorical columns
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(list(train_set[col]) + list(test_set[col]))
        train_set[col] = le.transform(train_set[col])
        test_set[col] = le.transform(test_set[col])
    # Map labels to integers (0: normal, 1: DoS, 2: Probe, 3: R2L, 4: U2R)
    label_map = {
        "normal": 0,
        "neptune": 1,
        "back": 1,
        "land": 1,
        "pod": 1,
        "smurf": 1,
        "teardrop": 1,
        "mailbomb": 1,
        "apache2": 1,
        "processtable": 1,
        "udpstorm": 1,
        "worm": 1,
        "ipsweep": 2,
        "nmap": 2,
        "portsweep": 2,
        "satan": 2,
        "mscan": 2,
        "saint": 2,
        "ftp_write": 3,
        "guess_passwd": 3,
        "imap": 3,
        "multihop": 3,
        "phf": 3,
        "spy": 3,
        "warezclient": 3,
        "warezmaster": 3,
        "sendmail": 3,
        "named": 3,
        "snmpgetattack": 3,
        "snmpguess": 3,
        "xlock": 3,
        "xsnoop": 3,
        "httptunnel": 3,
        "buffer_overflow": 4,
        "loadmodule": 4,
        "perl": 4,
        "rootkit": 4,
        "ps": 4,
        "sqlattack": 4,
        "xterm": 4,
    }
    train_set["label"] = train_set["label"].replace(label_map)
    test_set["label"] = test_set["label"].replace(label_map)
    # Drop rows with unmapped labels (if any)
    train_set = train_set.dropna(subset=["label"])
    test_set = test_set.dropna(subset=["label"])
    # Split features/labels
    X_train = train_set.drop("label", axis=1).values.astype(np.float64)
    y_train = train_set["label"].values.astype(int)
    X_test = test_set.drop("label", axis=1).values.astype(np.float64)
    y_test = test_set["label"].values.astype(int)
    # Standardize features
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


# %%
def run_onevsone_svm(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
):
    import time

    start_time = time.time()

    clf = OneVsOneClassifier(SVC(kernel="rbf", probability=True, random_state=42))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # Get decision function values for ROC curves
    y_pred_proba = clf.decision_function(X_test)

    training_time = time.time() - start_time

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    print("\n--- OneVsOne SVM Results ---")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Training Time:", f"{training_time:.2f} seconds")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return {
        "model": clf,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "training_time": training_time,
    }


# %%
def run_ensemble_svm(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
):
    classes = np.unique(y_train)
    binary_svms = []
    train_meta_features = []
    test_meta_features = []
    # Train one-vs-rest SVMs for each class
    for c in classes:
        y_binary = (y_train == c).astype(int)
        svm = SVC(kernel="rbf", probability=True, random_state=42)
        svm.fit(X_train, y_binary)
        binary_svms.append(svm)
        train_meta_features.append(svm.predict_proba(X_train)[:, 1])
        test_meta_features.append(svm.predict_proba(X_test)[:, 1])
    train_meta = np.stack(train_meta_features, axis=1)
    test_meta = np.stack(test_meta_features, axis=1)
    # Train meta SVM on these features
    meta_svm = SVC(kernel="rbf", probability=False, random_state=42)
    meta_svm.fit(train_meta, y_train)
    y_pred = meta_svm.predict(test_meta)
    print("\n--- Ensemble SVM (meta) Results ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average="weighted"))
    print("Recall:", recall_score(y_test, y_pred, average="weighted"))
    print("F1 Score:", f1_score(y_test, y_pred, average="weighted"))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return meta_svm


# %%
def run_pairwise_meta_svm(X_train, y_train, X_test, y_test):
    import time

    start_time = time.time()

    classes = np.unique(y_train)
    pairwise_svms = []
    train_meta_features = []
    test_meta_features = []

    # For each pair of classes, train a binary SVM and get probability features
    for i, j in combinations(classes, 2):
        # Select only samples of class i or j
        mask_train = np.logical_or(y_train == i, y_train == j)
        X_pair_train = X_train[mask_train]
        y_pair_train = y_train[mask_train]
        # Relabel: i->0, j->1
        y_pair_train_bin = (y_pair_train == j).astype(int)
        svm = SVC(kernel="rbf", probability=True, random_state=42)
        svm.fit(X_pair_train, y_pair_train_bin)
        pairwise_svms.append(((i, j), svm))
        # For all samples, get probability of class j
        prob_train = svm.predict_proba(X_train)[:, 1]
        prob_test = svm.predict_proba(X_test)[:, 1]
        train_meta_features.append(prob_train)
        test_meta_features.append(prob_test)

    # Stack all meta features horizontally
    X_train_meta = np.hstack(
        [X_train] + [f.reshape(-1, 1) for f in train_meta_features]
    )
    X_test_meta = np.hstack([X_test] + [f.reshape(-1, 1) for f in test_meta_features])

    # Train final SVM on extended features
    meta_svm = SVC(kernel="rbf", probability=False, random_state=42)
    meta_svm.fit(X_train_meta, y_train)
    y_pred = meta_svm.predict(X_test_meta)

    training_time = time.time() - start_time

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    print("\n--- Pairwise Meta SVM Results ---")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Training Time:", f"{training_time:.2f} seconds")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return {
        "model": meta_svm,
        "y_pred": y_pred,
        "y_pred_proba": None,  # Not available for this method
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "training_time": training_time,
    }


# %%
def run_pairwise_meta_svm_balanced(X_train, y_train, X_test, y_test):
    import time

    start_time = time.time()

    classes = np.unique(y_train)
    pairwise_svms = []
    train_meta_features = []
    test_meta_features = []

    # For each pair of classes, train a binary SVM and get probability features
    for i, j in combinations(classes, 2):
        # Select only samples of class i or j
        mask_train = np.logical_or(y_train == i, y_train == j)
        X_pair_train = X_train[mask_train]
        y_pair_train = y_train[mask_train]
        # Relabel: i->0, j->1
        y_pair_train_bin = (y_pair_train == j).astype(int)

        # Use balanced class weights
        svm = SVC(
            kernel="rbf", probability=True, random_state=42, class_weight="balanced"
        )
        svm.fit(X_pair_train, y_pair_train_bin)
        pairwise_svms.append(((i, j), svm))
        # For all samples, get probability of class j
        prob_train = svm.predict_proba(X_train)[:, 1]
        prob_test = svm.predict_proba(X_test)[:, 1]
        train_meta_features.append(prob_train)
        test_meta_features.append(prob_test)

    # Stack all meta features horizontally
    X_train_meta = np.hstack(
        [X_train] + [f.reshape(-1, 1) for f in train_meta_features]
    )
    X_test_meta = np.hstack([X_test] + [f.reshape(-1, 1) for f in test_meta_features])

    # Train final SVM on extended features with balanced weights
    meta_svm = SVC(
        kernel="rbf", probability=False, random_state=42, class_weight="balanced"
    )
    meta_svm.fit(X_train_meta, y_train)
    y_pred = meta_svm.predict(X_test_meta)

    training_time = time.time() - start_time

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    print("\n--- Pairwise Meta SVM (Balanced) Results ---")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Training Time:", f"{training_time:.2f} seconds")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return {
        "model": meta_svm,
        "y_pred": y_pred,
        "y_pred_proba": None,  # Not available for this method
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "training_time": training_time,
    }


# %%
def run_pairwise_meta_svm_smote(X_train, y_train, X_test, y_test):
    classes = np.unique(y_train)
    pairwise_svms = []
    train_meta_features = []
    test_meta_features = []

    # For each pair of classes, train a binary SVM and get probability features
    for i, j in combinations(classes, 2):
        # Select only samples of class i or j
        mask_train = np.logical_or(y_train == i, y_train == j)
        X_pair_train = X_train[mask_train]
        y_pair_train = y_train[mask_train]
        # Relabel: i->0, j->1
        y_pair_train_bin = (y_pair_train == j).astype(int)

        # Apply SMOTE for minority class
        if len(np.unique(y_pair_train_bin)) > 1:
            smote = SMOTE(random_state=42)
            resampled = smote.fit_resample(X_pair_train, y_pair_train_bin)
            X_pair_train_resampled, y_pair_train_bin_resampled = (
                resampled[0],
                resampled[1],
            )
        else:
            X_pair_train_resampled, y_pair_train_bin_resampled = (
                X_pair_train,
                y_pair_train_bin,
            )

        svm = SVC(kernel="rbf", probability=True, random_state=42)
        svm.fit(X_pair_train_resampled, y_pair_train_bin_resampled)
        pairwise_svms.append(((i, j), svm))
        # For all samples, get probability of class j
        prob_train = svm.predict_proba(X_train)[:, 1]
        prob_test = svm.predict_proba(X_test)[:, 1]
        train_meta_features.append(prob_train)
        test_meta_features.append(prob_test)

    # Stack all meta features horizontally
    X_train_meta = np.hstack(
        [X_train] + [f.reshape(-1, 1) for f in train_meta_features]
    )
    X_test_meta = np.hstack([X_test] + [f.reshape(-1, 1) for f in test_meta_features])

    # Train final SVM on extended features
    meta_svm = SVC(
        kernel="rbf", probability=False, random_state=42, class_weight="balanced"
    )
    meta_svm.fit(X_train_meta, y_train)
    y_pred = meta_svm.predict(X_test_meta)
    print("\n--- Pairwise Meta SVM (SMOTE) Results ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average="weighted"))
    print("Recall:", recall_score(y_test, y_pred, average="weighted"))
    print("F1 Score:", f1_score(y_test, y_pred, average="weighted"))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return meta_svm


# %%
def run_pairwise_meta_svm_enhanced(X_train, y_train, X_test, y_test):
    import time

    start_time = time.time()

    classes = np.unique(y_train)
    pairwise_svms = []
    train_meta_features = []
    test_meta_features = []

    # Count samples per class to identify minority classes
    class_counts = np.bincount(y_train)
    minority_classes = np.where(class_counts < np.median(class_counts))[0]

    # For each pair of classes, train a binary SVM and get probability features
    for i, j in combinations(classes, 2):
        # Select only samples of class i or j
        mask_train = np.logical_or(y_train == i, y_train == j)
        X_pair_train = X_train[mask_train]
        y_pair_train = y_train[mask_train]
        # Relabel: i->0, j->1
        y_pair_train_bin = (y_pair_train == j).astype(int)

        # Enhanced parameters for minority classes
        if i in minority_classes or j in minority_classes:
            # Higher C for minority classes, different kernel
            svm = SVC(
                kernel="poly",
                probability=True,
                random_state=42,
                class_weight="balanced",
                C=1000,
                gamma="scale",
            )
        else:
            svm = SVC(
                kernel="rbf",
                probability=True,
                random_state=42,
                class_weight="balanced",
                C=100,
            )

        svm.fit(X_pair_train, y_pair_train_bin)
        pairwise_svms.append(((i, j), svm))
        # For all samples, get probability of class j
        prob_train = svm.predict_proba(X_train)[:, 1]
        prob_test = svm.predict_proba(X_test)[:, 1]
        train_meta_features.append(prob_train)
        test_meta_features.append(prob_test)

    # Stack all meta features horizontally
    X_train_meta = np.hstack(
        [X_train] + [f.reshape(-1, 1) for f in train_meta_features]
    )
    X_test_meta = np.hstack([X_test] + [f.reshape(-1, 1) for f in test_meta_features])

    # Train final SVM with higher penalty for minority classes
    class_weights = {}
    for c in classes:
        if c in minority_classes:
            class_weights[c] = 10.0  # Higher weight for minority classes
        else:
            class_weights[c] = 1.0

    meta_svm = SVC(
        kernel="rbf",
        probability=False,
        random_state=42,
        class_weight=class_weights,
        C=100,
    )
    meta_svm.fit(X_train_meta, y_train)
    y_pred = meta_svm.predict(X_test_meta)

    training_time = time.time() - start_time

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    print("\n--- Enhanced Pairwise Meta SVM Results ---")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Training Time:", f"{training_time:.2f} seconds")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return {
        "model": meta_svm,
        "y_pred": y_pred,
        "y_pred_proba": None,  # Not available for this method
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "training_time": training_time,
    }


# %%
def run_focused_minority_svm(X_train, y_train, X_test, y_test):
    """Focus specifically on improving minority classes (4 and 5)"""
    import time

    start_time = time.time()

    minority_classes = [3, 4]  # R2L and U2R attacks

    # Create specialized features for minority classes
    train_minority_features = []
    test_minority_features = []

    for minority_class in minority_classes:
        # Binary classification: minority_class vs all others
        y_binary = (y_train == minority_class).astype(int)

        # Check if minority class exists in training data
        if len(np.unique(y_binary)) < 2:
            print(f"Warning: Class {minority_class} not found in training data")
            # Create dummy probabilities (all zeros)
            prob_train = np.zeros(len(X_train))
            prob_test = np.zeros(len(X_test))
        else:
            # Use higher C and different kernel for minority classes
            svm_minority = SVC(
                kernel="poly",
                probability=True,
                random_state=42,
                class_weight="balanced",
                C=2000,
                gamma="auto",
            )
            svm_minority.fit(X_train, y_binary)
            # Get probabilities for all samples
            prob_train = svm_minority.predict_proba(X_train)[:, 1]
            prob_test = svm_minority.predict_proba(X_test)[:, 1]

        train_minority_features.append(prob_train)
        test_minority_features.append(prob_test)

    # Combine original features with minority-focused features
    X_train_enhanced = np.hstack(
        [X_train] + [f.reshape(-1, 1) for f in train_minority_features]
    )
    X_test_enhanced = np.hstack(
        [X_test] + [f.reshape(-1, 1) for f in test_minority_features]
    )

    # Train final classifier with very high weights for minority classes
    class_weights = {
        0: 1.0,
        1: 1.0,
        2: 1.0,
        3: 25.0,
        4: 25.0,
    }  # Much higher weights for classes 4,5

    final_svm = SVC(
        kernel="rbf",
        probability=False,
        random_state=42,
        class_weight=class_weights,
        C=100,
    )
    final_svm.fit(X_train_enhanced, y_train)
    y_pred = final_svm.predict(X_test_enhanced)

    training_time = time.time() - start_time

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    print("\n--- Focused Minority SVM Results ---")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Training Time:", f"{training_time:.2f} seconds")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Print specific metrics for minority classes
    recall_per_class = np.array(recall_score(y_test, y_pred, average=None))  # type: ignore
    print(f"\nClass 4 (R2L) - Recall: {recall_per_class[3]:.3f}")
    print(f"Class 5 (U2R) - Recall: {recall_per_class[4]:.3f}")

    return {
        "model": final_svm,
        "y_pred": y_pred,
        "y_pred_proba": None,  # Not available for this method
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "training_time": training_time,
    }


# %%
def plot_confusion_matrix_heatmap(
    y_true, y_pred, title="Confusion Matrix", save_path=None
):
    """Plot confusion matrix as heatmap with publication-quality styling"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.2)

    # Create heatmap with custom styling
    heatmap = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "DoS", "Probe", "R2L", "U2R"],
        yticklabels=["Normal", "DoS", "Probe", "R2L", "U2R"],
        cbar_kws={"label": "Number of Predictions"},
        square=True,
        linewidths=0.5,
        linecolor="white",
    )

    plt.title(title, fontsize=16, fontweight="bold", pad=20)
    plt.ylabel("True Label", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# %%
def plot_performance_comparison(results_dict, save_path=None):
    """Plot performance comparison across different methods with publication-quality styling"""
    methods = list(results_dict.keys())
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]

    # Set up the plot style
    plt.style.use("seaborn-v0_8")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()

    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#8B4513"]

    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = []
        valid_methods = []

        for method in methods:
            if metric in results_dict[method]:
                values.append(results_dict[method][metric])
                valid_methods.append(method)

        if values:  # Only plot if we have valid data
            bars = axes[i].bar(
                valid_methods,
                values,
                color=colors[: len(valid_methods)],
                alpha=0.8,
                edgecolor="black",
                linewidth=1,
            )
            axes[i].set_title(
                f"{label} Comparison", fontsize=14, fontweight="bold", pad=15
            )
            axes[i].set_ylabel(label, fontsize=12, fontweight="bold")
            axes[i].tick_params(axis="x", rotation=45)
            axes[i].set_ylim(0, 1)
            axes[i].grid(True, alpha=0.3)

            # Add value labels on bars
            for j, v in enumerate(values):
                axes[i].text(
                    j,
                    v + 0.01,
                    f"{v:.3f}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    fontsize=10,
                )
        else:
            axes[i].text(
                0.5,
                0.5,
                f"No {label} data available",
                ha="center",
                va="center",
                transform=axes[i].transAxes,
                fontsize=12,
                fontweight="bold",
            )
            axes[i].set_title(
                f"{label} Comparison", fontsize=14, fontweight="bold", pad=15
            )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# %%
def plot_roc_curves(y_true, y_pred_proba, class_names=None, save_path=None):
    """Plot ROC curves for each class"""
    if class_names is None:
        class_names = ["Normal", "DoS", "Probe", "R2L", "U2R"]

    from sklearn.metrics import auc, roc_curve

    plt.figure(figsize=(12, 8))

    # Compute ROC curve and ROC area for each class
    n_classes = len(class_names)
    colors = ["blue", "red", "green", "orange", "purple"]

    for i in range(n_classes):
        # Convert to binary classification for each class
        y_true_binary = (y_true == i).astype(int)
        y_score_binary = y_pred_proba[:, i]

        fpr, tpr, _ = roc_curve(y_true_binary, y_score_binary)
        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            color=colors[i],
            lw=2,
            label=f"{class_names[i]} (AUC = {roc_auc:.3f})",
        )

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=14, fontweight="bold")
    plt.ylabel("True Positive Rate", fontsize=14, fontweight="bold")
    plt.title("ROC Curves for All Classes", fontsize=16, fontweight="bold")
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# %%
def plot_precision_recall_curves(
    y_true, y_pred_proba, class_names=None, save_path=None
):
    """Plot Precision-Recall curves for each class"""
    if class_names is None:
        class_names = ["Normal", "DoS", "Probe", "R2L", "U2R"]

    from sklearn.metrics import average_precision_score, precision_recall_curve

    plt.figure(figsize=(12, 8))

    n_classes = len(class_names)
    colors = ["blue", "red", "green", "orange", "purple"]

    for i in range(n_classes):
        y_true_binary = (y_true == i).astype(int)
        y_score_binary = y_pred_proba[:, i]

        precision, recall, _ = precision_recall_curve(y_true_binary, y_score_binary)
        avg_precision = average_precision_score(y_true_binary, y_score_binary)

        plt.plot(
            recall,
            precision,
            color=colors[i],
            lw=2,
            label=f"{class_names[i]} (AP = {avg_precision:.3f})",
        )

    plt.xlabel("Recall", fontsize=14, fontweight="bold")
    plt.ylabel("Precision", fontsize=14, fontweight="bold")
    plt.title("Precision-Recall Curves for All Classes", fontsize=16, fontweight="bold")
    plt.legend(loc="lower left", fontsize=12)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# %%
def plot_feature_importance(selector, feature_names=None, save_path=None):
    """Plot feature importance scores"""
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(len(selector.scores_))]

    # Get feature scores and sort them
    scores = selector.scores_
    feature_importance = list(zip(feature_names, scores))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    # Plot top 20 features
    top_features = feature_importance[:20]
    names, scores = zip(*top_features)

    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(names)), scores, color="skyblue", edgecolor="black")
    plt.yticks(range(len(names)), names)
    plt.xlabel("F-Score", fontsize=14, fontweight="bold")
    plt.title("Top 20 Most Important Features", fontsize=16, fontweight="bold")
    plt.grid(True, alpha=0.3, axis="x")

    # Add value labels on bars
    for i, score in enumerate(scores):
        plt.text(
            score + max(scores) * 0.01,
            i,
            f"{score:.2f}",
            va="center",
            fontweight="bold",
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# %%
def plot_class_distribution(y_train, y_test, save_path=None):
    """Plot class distribution in training and test sets"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    class_names = ["Normal", "DoS", "Probe", "R2L", "U2R"]
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#8B4513"]

    # Training set distribution
    train_counts = np.bincount(y_train)
    ax1.bar(class_names, train_counts, color=colors, alpha=0.8, edgecolor="black")
    ax1.set_title("Training Set Class Distribution", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Number of Samples", fontsize=12, fontweight="bold")
    ax1.tick_params(axis="x", rotation=45)

    # Add value labels on bars
    for i, count in enumerate(train_counts):
        ax1.text(
            i,
            count + max(train_counts) * 0.01,
            str(count),
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Test set distribution
    test_counts = np.bincount(y_test)
    ax2.bar(class_names, test_counts, color=colors, alpha=0.8, edgecolor="black")
    ax2.set_title("Test Set Class Distribution", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Number of Samples", fontsize=12, fontweight="bold")
    ax2.tick_params(axis="x", rotation=45)

    # Add value labels on bars
    for i, count in enumerate(test_counts):
        ax2.text(
            i,
            count + max(test_counts) * 0.01,
            str(count),
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# %%
def plot_detection_vs_false_alarm(y_true, y_pred, save_path=None):
    """Plot Detection Rate vs False Alarm Rate for each class"""
    class_names = ["Normal", "DoS", "Probe", "R2L", "U2R"]
    colors = ["blue", "red", "green", "orange", "purple"]

    detection_rates = []
    false_alarm_rates = []

    for i in range(5):  # 5 classes
        # True positives and false positives for this class
        tp = np.sum((y_true == i) & (y_pred == i))
        fp = np.sum((y_true != i) & (y_pred == i))
        tn = np.sum((y_true != i) & (y_pred != i))
        fn = np.sum((y_true == i) & (y_pred != i))

        # Calculate rates
        detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
        false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

        detection_rates.append(detection_rate)
        false_alarm_rates.append(false_alarm_rate)

    plt.figure(figsize=(10, 8))
    for i, (dr, far) in enumerate(zip(detection_rates, false_alarm_rates)):
        plt.scatter(
            far,
            dr,
            s=200,
            c=colors[i],
            label=class_names[i],
            alpha=0.7,
            edgecolors="black",
            linewidth=2,
        )
        plt.annotate(
            class_names[i],
            (far, dr),
            xytext=(5, 5),
            textcoords="offset points",
            fontweight="bold",
        )

    plt.xlabel("False Alarm Rate", fontsize=14, fontweight="bold")
    plt.ylabel("Detection Rate", fontsize=14, fontweight="bold")
    plt.title("Detection Rate vs False Alarm Rate", fontsize=16, fontweight="bold")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# %%
def plot_training_time_comparison(methods, training_times, save_path=None):
    """Plot training time comparison across different methods"""
    plt.figure(figsize=(12, 8))

    bars = plt.bar(
        methods,
        training_times,
        color=["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#8B4513"],
        alpha=0.8,
        edgecolor="black",
        linewidth=1,
    )

    plt.title("Training Time Comparison", fontsize=16, fontweight="bold")
    plt.ylabel("Training Time (seconds)", fontsize=14, fontweight="bold")
    plt.xlabel("Methods", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, time in zip(bars, training_times):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(training_times) * 0.01,
            f"{time:.2f}s",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# %%
def create_comprehensive_report(
    X_train, y_train, X_test, y_test, results_dict, save_dir="plots"
):
    """Create a comprehensive visualization report"""
    import os
    from datetime import datetime

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Creating comprehensive visualization report...")

    # 1. Class distribution
    plot_class_distribution(
        y_train,
        y_test,
        save_path=os.path.join(save_dir, f"class_distribution_{timestamp}.png"),
    )

    # 2. Performance comparison
    plot_performance_comparison(
        results_dict,
        save_path=os.path.join(save_dir, f"performance_comparison_{timestamp}.png"),
    )

    # 3. Confusion matrices for each method (if available)
    for method_name in results_dict.keys():
        if "y_pred" in results_dict[method_name]:
            plot_confusion_matrix_heatmap(
                y_test,
                results_dict[method_name]["y_pred"],
                title=f"Confusion Matrix - {method_name}",
                save_path=os.path.join(
                    save_dir, f"confusion_matrix_{method_name}_{timestamp}.png"
                ),
            )

    # 4. Detection vs False Alarm Rate
    if "y_pred" in results_dict.get("OneVsOne", {}):
        plot_detection_vs_false_alarm(
            y_test,
            results_dict["OneVsOne"]["y_pred"],
            save_path=os.path.join(
                save_dir, f"detection_vs_false_alarm_{timestamp}.png"
            ),
        )

    print(f"All plots saved to {save_dir}/")
    print("Report generation completed!")


# %%
def perform_feature_selection(X_train, y_train, X_test, y_test):
    """Perform feature selection using SelectKBest"""
    print("Performing feature selection with SelectKBest...")

    # Use SelectKBest with f_classif for feature selection
    # Select top 50% of features
    n_features = max(1, X_train.shape[1] // 2)
    selector = SelectKBest(score_func=f_classif, k=n_features)

    selector.fit(X_train, y_train)

    print(f"Selected features: {n_features} out of {X_train.shape[1]}")

    # Plot feature selection results if plotting is available
    if PLOTTING_AVAILABLE:
        plt.figure(figsize=(10, 6))
        scores = selector.scores_
        plt.bar(range(len(scores)), scores)
        plt.xlabel("Feature Index")
        plt.ylabel("F-Score")
        plt.title("Feature Selection Scores")
        plt.grid(True)
        plt.show()

    # Get selected features
    selected_features = selector.get_support()
    X_train_selected = X_train[:, selected_features]  # type: ignore
    X_test_selected = X_test[:, selected_features]  # type: ignore

    return X_train_selected, X_test_selected, selected_features, selector


# %%
def run_experiment_with_feature_selection(X_train, y_train, X_test, y_test):
    """Run complete experiment with feature selection"""
    results = {}

    # Perform feature selection
    X_train_selected, X_test_selected, selected_features, rfecv = (
        perform_feature_selection(X_train, y_train, X_test, y_test)
    )

    print(
        f"\nRunning experiments with {np.sum(selected_features)} selected features..."  # type: ignore
    )

    # Test different methods with selected features
    methods = [
        ("OneVsOne", run_onevsone_svm),
        ("Pairwise Meta", run_pairwise_meta_svm),
        ("Balanced Meta", run_pairwise_meta_svm_balanced),
        ("Enhanced Meta", run_pairwise_meta_svm_enhanced),
        ("Focused Minority", run_focused_minority_svm),
    ]

    for method_name, method_func in methods:
        print(f"\n{'=' * 50}")
        print(f"Testing: {method_name}")
        print(f"{'=' * 50}")

        try:
            # Capture the output to extract metrics
            import io
            from contextlib import redirect_stdout

            f = io.StringIO()
            with redirect_stdout(f):
                method_func(X_train_selected, y_train, X_test_selected, y_test)

            output = f.getvalue()
            print(output)

            # Extract metrics from output (simplified - you might want to modify the functions to return metrics)
            # For now, just store that the method completed successfully
            results[method_name] = {"Status": "Completed"}

        except Exception as e:
            print(f"Error in {method_name}: {e}")
            results[method_name] = {"Status": f"Error: {e}"}

    return results, selected_features, rfecv


# %%
def run_simple_experiment(X_train, y_train, X_test, y_test):
    """Run experiments without feature selection"""
    print("Running experiments without feature selection...")

    methods = [
        ("OneVsOne", run_onevsone_svm),
        ("Pairwise Meta", run_pairwise_meta_svm),
        ("Balanced Meta", run_pairwise_meta_svm_balanced),
        ("Enhanced Meta", run_pairwise_meta_svm_enhanced),
        ("Focused Minority", run_focused_minority_svm),
    ]

    results = {}

    for method_name, method_func in methods:
        print(f"\n{'=' * 50}")
        print(f"Testing: {method_name}")
        print(f"{'=' * 50}")

        try:
            method_func(X_train, y_train, X_test, y_test)
            results[method_name] = {"Status": "Completed"}
        except Exception as e:
            print(f"Error in {method_name}: {e}")
            results[method_name] = {"Status": f"Error: {e}"}

    return results


# %%
def run_advanced_minority_svm(X_train, y_train, X_test, y_test):
    """Advanced SVM specifically designed for minority classes (R2L and U2R)"""
    import time

    start_time = time.time()

    minority_classes = [3, 4]  # R2L and U2R attacks

    print("\n--- Advanced Minority SVM for R2L and U2R ---")
    print("Original class distribution:")
    for i, count in enumerate(np.bincount(y_train)):
        print(f"Class {i}: {count} samples")

    # 1. Apply SMOTE for minority classes if available
    if SMOTE_AVAILABLE:
        print("\nApplying SMOTE for minority class balancing...")
        smote = SMOTE(random_state=42, k_neighbors=3)
        resampled = smote.fit_resample(X_train, y_train)
        X_train_balanced, y_train_balanced = resampled[0], resampled[1]
        print("After SMOTE - Class distribution:")
        for i, count in enumerate(np.bincount(y_train_balanced)):
            print(f"Class {i}: {count} samples")
    else:
        X_train_balanced, y_train_balanced = X_train, y_train
        print("SMOTE not available, using original data")

    # 2. Create specialized features for minority classes
    train_minority_features = []
    test_minority_features = []

    for minority_class in minority_classes:
        # Binary classification: minority_class vs all others
        y_binary = (y_train_balanced == minority_class).astype(int)

        # Check if minority class exists in training data
        if len(np.unique(y_binary)) < 2:
            print(f"Warning: Class {minority_class} not found in training data")
            prob_train = np.zeros(len(X_train))
            prob_test = np.zeros(len(X_test))
        else:
            # Use specialized parameters for minority classes
            svm_minority = SVC(
                kernel="rbf",  # Changed from poly to rbf for better generalization
                probability=True,
                random_state=42,
                class_weight="balanced",
                C=5000,  # Increased C for stronger regularization
                gamma="scale",
            )
            svm_minority.fit(X_train_balanced, y_binary)
            prob_train = svm_minority.predict_proba(X_train)[:, 1]
            prob_test = svm_minority.predict_proba(X_test)[:, 1]

        train_minority_features.append(prob_train)
        test_minority_features.append(prob_test)

    # 3. Create additional features using different kernels
    kernel_features_train = []
    kernel_features_test = []

    for minority_class in minority_classes:
        y_binary = (y_train_balanced == minority_class).astype(int)

        if len(np.unique(y_binary)) >= 2:
            # Linear kernel for minority class
            svm_linear = SVC(
                kernel="linear",
                probability=True,
                random_state=42,
                class_weight="balanced",
                C=1000,
            )
            svm_linear.fit(X_train_balanced, y_binary)
            prob_linear_train = svm_linear.predict_proba(X_train)[:, 1]
            prob_linear_test = svm_linear.predict_proba(X_test)[:, 1]

            kernel_features_train.append(prob_linear_train)
            kernel_features_test.append(prob_linear_test)
        else:
            kernel_features_train.append(np.zeros(len(X_train)))
            kernel_features_test.append(np.zeros(len(X_test)))

    # 4. Combine all features
    X_train_enhanced = np.hstack(
        [
            X_train,
            [f.reshape(-1, 1) for f in train_minority_features],
            [f.reshape(-1, 1) for f in kernel_features_train],
        ]
    )
    X_test_enhanced = np.hstack(
        [
            X_test,
            [f.reshape(-1, 1) for f in test_minority_features],
            [f.reshape(-1, 1) for f in kernel_features_test],
        ]
    )

    # 5. Train ensemble of SVMs with different parameters
    ensemble_svms = []
    ensemble_weights = []

    # SVM 1: High C, balanced weights
    svm1 = SVC(
        kernel="rbf",
        probability=True,
        random_state=42,
        class_weight="balanced",
        C=1000,
        gamma="scale",
    )
    svm1.fit(X_train_enhanced, y_train_balanced)
    ensemble_svms.append(svm1)
    ensemble_weights.append(0.4)

    # SVM 2: Very high C for minority classes
    class_weights_high = {0: 1.0, 1: 1.0, 2: 1.0, 3: 30.0, 4: 30.0}
    svm2 = SVC(
        kernel="rbf",
        probability=True,
        random_state=42,
        class_weight=class_weights_high,
        C=2000,
        gamma="auto",
    )
    svm2.fit(X_train_enhanced, y_train_balanced)
    ensemble_svms.append(svm2)
    ensemble_weights.append(0.3)

    # SVM 3: Linear kernel
    svm3 = SVC(
        kernel="linear",
        probability=True,
        random_state=42,
        class_weight="balanced",
        C=500,
    )
    svm3.fit(X_train_enhanced, y_train_balanced)
    ensemble_svms.append(svm3)
    ensemble_weights.append(0.3)

    # 6. Ensemble prediction
    predictions = []
    for svm, weight in zip(ensemble_svms, ensemble_weights):
        pred_proba = svm.predict_proba(X_test_enhanced)
        predictions.append(pred_proba * weight)

    # Weighted ensemble
    ensemble_proba = np.sum(predictions, axis=0)
    y_pred = np.argmax(ensemble_proba, axis=1)

    training_time = time.time() - start_time

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    print("\n--- Advanced Minority SVM Results ---")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Training Time:", f"{training_time:.2f} seconds")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Print specific metrics for minority classes
    recall_per_class = recall_score(y_test, y_pred, average=None)  # type: ignore
    precision_per_class = precision_score(y_test, y_pred, average=None)  # type: ignore
    f1_per_class = f1_score(y_test, y_pred, average=None)  # type: ignore

    print("\nDetailed Minority Class Performance:")
    print(
        f"Class 3 (R2L) - Recall: {recall_per_class[3]:.3f}, Precision: {precision_per_class[3]:.3f}, F1: {f1_per_class[3]:.3f}"  # type: ignore
    )
    print(
        f"Class 4 (U2R) - Recall: {recall_per_class[4]:.3f}, Precision: {precision_per_class[4]:.3f}, F1: {f1_per_class[4]:.3f}"  # type: ignore
    )

    return {
        "model": ensemble_svms,
        "y_pred": y_pred,
        "y_pred_proba": ensemble_proba,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "training_time": training_time,
        "minority_recall": {"R2L": recall_per_class[3], "U2R": recall_per_class[4]},  # type: ignore
    }


# %%
def run_cost_sensitive_svm(X_train, y_train, X_test, y_test):
    """Cost-sensitive SVM with adaptive class weights based on class distribution"""
    import time

    start_time = time.time()

    # Calculate class distribution
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)

    # Calculate adaptive class weights
    class_weights = {}
    for i in range(len(class_counts)):
        if class_counts[i] > 0:
            # Inverse frequency weighting with additional penalty for minority classes
            base_weight = total_samples / (len(class_counts) * class_counts[i])
            if i in [3, 4]:  # R2L and U2R
                class_weights[i] = base_weight * 5.0  # 5x penalty for minority classes
            else:
                class_weights[i] = base_weight
        else:
            class_weights[i] = 1.0

    print(f"Adaptive class weights: {class_weights}")

    # Train cost-sensitive SVM
    svm = SVC(
        kernel="rbf",
        probability=True,
        random_state=42,
        class_weight=class_weights,
        C=1000,
        gamma="scale",
    )

    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    y_pred_proba = svm.predict_proba(X_test)

    training_time = time.time() - start_time

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    print("\n--- Cost-Sensitive SVM Results ---")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Training Time:", f"{training_time:.2f} seconds")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Print minority class performance
    recall_per_class = recall_score(y_test, y_pred, average=None)  # type: ignore
    print("\nMinority Class Performance:")
    print(f"Class 3 (R2L) - Recall: {recall_per_class[3]:.3f}")  # type: ignore
    print(f"Class 4 (U2R) - Recall: {recall_per_class[4]:.3f}")  # type: ignore

    return {
        "model": svm,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "training_time": training_time,
        "minority_recall": {"R2L": recall_per_class[3], "U2R": recall_per_class[4]},  # type: ignore
    }


# %%
def run_hybrid_ensemble_svm(X_train, y_train, X_test, y_test):
    """Hybrid ensemble combining multiple approaches for minority classes"""
    import time

    start_time = time.time()

    minority_classes = [3, 4]  # R2L and U2R

    # 1. Apply SMOTE if available
    if SMOTE_AVAILABLE:
        smote = SMOTE(random_state=42, k_neighbors=3)
        resampled = smote.fit_resample(X_train, y_train)
        X_train_balanced, y_train_balanced = resampled[0], resampled[1]
    else:
        X_train_balanced, y_train_balanced = X_train, y_train

    # 2. Create multiple specialized classifiers
    classifiers = []

    # Classifier 1: Standard SVM with balanced weights
    clf1 = SVC(
        kernel="rbf",
        probability=True,
        random_state=42,
        class_weight="balanced",
        C=100,
    )
    clf1.fit(X_train_balanced, y_train_balanced)
    classifiers.append(("Balanced_SVM", clf1, 0.3))

    # Classifier 2: High C SVM for minority classes
    class_weights_high = {0: 1.0, 1: 1.0, 2: 1.0, 3: 30.0, 4: 30.0}
    clf2 = SVC(
        kernel="rbf",
        probability=True,
        random_state=42,
        class_weight=class_weights_high,
        C=2000,
        gamma="auto",
    )
    clf2.fit(X_train_balanced, y_train_balanced)
    classifiers.append(("HighC_SVM", clf2, 0.4))

    # Classifier 3: Linear SVM
    clf3 = SVC(
        kernel="linear",
        probability=True,
        random_state=42,
        class_weight="balanced",
        C=500,
    )
    clf3.fit(X_train_balanced, y_train_balanced)
    classifiers.append(("Linear_SVM", clf3, 0.3))

    # 3. Ensemble prediction with weighted voting
    predictions = []
    for name, clf, weight in classifiers:
        pred_proba = clf.predict_proba(X_test)
        predictions.append(pred_proba * weight)

    ensemble_proba = np.sum(predictions, axis=0)
    y_pred = np.argmax(ensemble_proba, axis=1)

    training_time = time.time() - start_time

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    print("\n--- Hybrid Ensemble SVM Results ---")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Training Time:", f"{training_time:.2f} seconds")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Print minority class performance
    recall_per_class = recall_score(y_test, y_pred, average=None)  # type: ignore
    print("\nMinority Class Performance:")
    print(f"Class 3 (R2L) - Recall: {recall_per_class[3]:.3f}")  # type: ignore
    print(f"Class 4 (U2R) - Recall: {recall_per_class[4]:.3f}")  # type: ignore

    return {
        "model": classifiers,
        "y_pred": y_pred,
        "y_pred_proba": ensemble_proba,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "training_time": training_time,
        "minority_recall": {"R2L": recall_per_class[3], "U2R": recall_per_class[4]},  # type: ignore
    }


# %%
# Main execution cell
if __name__ == "__main__":
    train_set, test_set = load_nsl_kdd()
    X_train, X_test, y_train, y_test = preprocess_nsl_kdd(train_set, test_set)

    print("NSL-KDD Intrusion Detection with SVM Methods")
    print("=" * 50)

    # Store results for visualization
    results_dict = {}
    training_times = []
    methods = []

    # Run individual methods and collect results
    print("\nRunning individual SVM methods...")

    # 1. OneVsOne SVM
    try:
        result_ovo = run_onevsone_svm(
            np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
        )
        results_dict["OneVsOne"] = result_ovo
        training_times.append(result_ovo["training_time"])
        methods.append("OneVsOne")

        # Plot confusion matrix for OneVsOne
        if PLOTTING_AVAILABLE:
            plot_confusion_matrix_heatmap(
                np.array(y_test),
                result_ovo["y_pred"],
                title="OneVsOne SVM Confusion Matrix",
            )

            # Plot ROC curves
            plot_roc_curves(np.array(y_test), result_ovo["y_pred_proba"])

    except Exception as e:
        print(f"OneVsOne SVM failed: {e}")

    # 2. Pairwise Meta SVM
    try:
        result_pairwise = run_pairwise_meta_svm(
            np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
        )
        results_dict["Pairwise Meta"] = result_pairwise
        training_times.append(result_pairwise["training_time"])
        methods.append("Pairwise Meta")
    except Exception as e:
        print(f"Pairwise Meta SVM failed: {e}")

    # 3. Balanced Meta SVM
    try:
        result_balanced = run_pairwise_meta_svm_balanced(
            np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
        )
        results_dict["Balanced Meta"] = result_balanced
        training_times.append(result_balanced["training_time"])
        methods.append("Balanced Meta")
    except Exception as e:
        print(f"Balanced Meta SVM failed: {e}")

    # 4. Enhanced Meta SVM
    try:
        result_enhanced = run_pairwise_meta_svm_enhanced(
            np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
        )
        results_dict["Enhanced Meta"] = result_enhanced
        training_times.append(result_enhanced["training_time"])
        methods.append("Enhanced Meta")
    except Exception as e:
        print(f"Enhanced Meta SVM failed: {e}")

    # 5. Focused Minority SVM
    try:
        result_focused = run_focused_minority_svm(
            np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
        )
        results_dict["Focused Minority"] = result_focused
        training_times.append(result_focused["training_time"])
        methods.append("Focused Minority")
    except Exception as e:
        print(f"Focused Minority SVM failed: {e}")

    # 6. Advanced Minority SVM (NEW - specifically for R2L and U2R)
    try:
        result_advanced = run_advanced_minority_svm(
            np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
        )
        results_dict["Advanced Minority"] = result_advanced
        training_times.append(result_advanced["training_time"])
        methods.append("Advanced Minority")
    except Exception as e:
        print(f"Advanced Minority SVM failed: {e}")

    # 7. Cost-Sensitive SVM (NEW - adaptive class weights)
    try:
        result_cost = run_cost_sensitive_svm(
            np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
        )
        results_dict["Cost-Sensitive"] = result_cost
        training_times.append(result_cost["training_time"])
        methods.append("Cost-Sensitive")
    except Exception as e:
        print(f"Cost-Sensitive SVM failed: {e}")

    # 8. Hybrid Ensemble SVM (NEW - multiple approaches)
    try:
        result_hybrid = run_hybrid_ensemble_svm(
            np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
        )
        results_dict["Hybrid Ensemble"] = result_hybrid
        training_times.append(result_hybrid["training_time"])
        methods.append("Hybrid Ensemble")
    except Exception as e:
        print(f"Hybrid Ensemble SVM failed: {e}")

    # Create comprehensive visualizations
    if PLOTTING_AVAILABLE and results_dict:
        print("\nCreating comprehensive visualizations...")

        # 1. Class distribution
        plot_class_distribution(np.array(y_train), np.array(y_test))

        # 2. Performance comparison
        plot_performance_comparison(results_dict)

        # 3. Training time comparison
        if training_times:
            plot_training_time_comparison(methods, training_times)

        # 4. Detection vs False Alarm Rate
        if "OneVsOne" in results_dict:
            plot_detection_vs_false_alarm(
                np.array(y_test), results_dict["OneVsOne"]["y_pred"]
            )

        # 5. Create comprehensive report
        try:
            create_comprehensive_report(
                np.array(X_train),
                np.array(y_train),
                np.array(X_test),
                np.array(y_test),
                results_dict,
            )
        except Exception as e:
            print(f"Report generation failed: {e}")

    print("\nExperiment completed!")
    print(f"Successfully tested {len(results_dict)} methods")

# %%
