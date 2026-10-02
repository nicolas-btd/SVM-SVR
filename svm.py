import os
import time
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as skdata
import sklearn.preprocessing as skprep
import sklearn.svm as sksvm
import sklearn.model_selection as sksel
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error

# Set matplotlib styling for clean scientific publication quality
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['figure.titlesize'] = 13


def plot_frontiere(clf, window, ax=None):
    """
    Plots decision boundary and decision function margins for a classifier.
    """
    if ax is None:
        ax = plt.gca()
    x0s = np.linspace(window[0], window[1], 200)
    x1s = np.linspace(window[2], window[3], 200)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    y_decision = clf.decision_function(X).reshape(x0.shape)
    
    # Decision regions
    ax.contourf(x0, x1, y_pred, alpha=0.15, cmap='coolwarm')
    # Decision boundary and margins (0, +1, -1)
    ax.contour(x0, x1, y_decision, levels=[-1, 0, 1], linestyles=['--', '-', '--'],
               colors=['#444444', '#111111', '#444444'], linewidths=[1.0, 1.8, 1.0])


def plot_svm_regression(svm_reg, X, y, axes, ax=None):
    """
    Plots SVR fit, epsilon tube, and highlights support vectors.
    """
    if ax is None:
        ax = plt.gca()
    x1s = np.linspace(axes[0], axes[1], 200).reshape(-1, 1)
    y_pred = svm_reg.predict(x1s)
    ax.plot(x1s, y_pred, "r-", linewidth=2, label=r"$\hat{y}$ (SVR prediction)")
    ax.plot(x1s, y_pred + svm_reg.epsilon, "k--", linewidth=1.2, label=r"$\varepsilon$-tube bounds")
    ax.plot(x1s, y_pred - svm_reg.epsilon, "k--", linewidth=1.2)
    
    # Highlight support vectors (points on or outside the tube)
    if hasattr(svm_reg, "support_") and len(svm_reg.support_) > 0:
        ax.scatter(X[svm_reg.support_], y[svm_reg.support_], s=120,
                   facecolors='none', edgecolors='crimson', linewidths=1.5,
                   label=f"Support Vectors ({len(svm_reg.support_)})")
    ax.scatter(X, y, color='#1f77b4', s=30, alpha=0.8, edgecolors='k', linewidths=0.5, label="Data points")
    ax.set_xlim(axes[0], axes[1])
    ax.set_ylim(axes[2], axes[3])
    ax.legend(loc="upper left", frameon=True, fontsize=8)


# =============================================================================
# 1. SVM Classification on Synthetic Data (Moons)
# =============================================================================
def explore_synthetic_svm():
    print("\n--- 1. SVM Classification on Synthetic Data (make_moons) ---")
    X1, y1 = skdata.make_moons(n_samples=400, noise=0.15, random_state=42)
    scaler = skprep.StandardScaler()
    X1norm = scaler.fit_transform(X1)

    minX = np.floor(X1norm.min(axis=0)) - 0.5
    maxX = np.ceil(X1norm.max(axis=0)) + 0.5
    window = [minX[0], maxX[0], minX[1], maxX[1]]

    models = [
        ("Linear (C=1.0)", sksvm.SVC(kernel='linear', C=1.0)),
        ("Poly (d=3, C=1, r=1)", sksvm.SVC(kernel='poly', degree=3, coef0=1, C=1.0)),
        ("Poly (d=10, C=100, r=1)", sksvm.SVC(kernel='poly', degree=10, coef0=1, C=100.0)),
        ("RBF (γ=0.5, C=1.0)", sksvm.SVC(kernel='rbf', gamma=0.5, C=1.0)),
        ("RBF (γ=5.0, C=1.0)", sksvm.SVC(kernel='rbf', gamma=5.0, C=1.0)),
        ("RBF (γ=5.0, C=100.0)", sksvm.SVC(kernel='rbf', gamma=5.0, C=100.0)),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()

    for idx, (title, model) in enumerate(models):
        model.fit(X1norm, y1)
        acc = accuracy_score(y1, model.predict(X1norm))
        n_sv = len(model.support_)
        
        plot_frontiere(model, window, ax=axes[idx])
        scatter = axes[idx].scatter(X1norm[:, 0], X1norm[:, 1], c=y1, cmap='coolwarm',
                                    edgecolors='k', linewidths=0.5, s=30)
        axes[idx].set_title(f"{title}\nAcc: {acc:.2%} | SV: {n_sv}")
        axes[idx].set_xlim(window[0], window[1])
        axes[idx].set_ylim(window[2], window[3])

    plt.tight_layout()
    plt.savefig('svm_synthetic_comparison.png', dpi=300)
    plt.close()
    print("Saved 'svm_synthetic_comparison.png'")


# =============================================================================
# 2. SVR on 1D Synthetic Data
# =============================================================================
def explore_synthetic_svr():
    print("\n--- 2. SVR on 1D Synthetic Data ---")
    np.random.seed(42)
    n = 60
    X2 = 6 * np.random.rand(n, 1)
    y2 = (1.5 * np.sin(X2) + X2 + np.random.randn(n, 1) * 0.7).ravel()

    minX = 0
    maxX = 6
    minY = np.floor(np.min(y2)) - 1
    maxY = np.ceil(np.max(y2)) + 1
    axes_limits = [minX, maxX, minY, maxY]

    # Comparison 1: Kernel comparison
    svr_linear = sksvm.SVR(kernel='linear', C=10.0, epsilon=0.2).fit(X2, y2)
    svr_poly = sksvm.SVR(kernel='poly', degree=3, coef0=1.0, C=10.0, epsilon=0.2).fit(X2, y2)
    svr_rbf = sksvm.SVR(kernel='rbf', gamma=0.5, C=10.0, epsilon=0.2).fit(X2, y2)

    xtest = np.linspace(minX, maxX, 200).reshape(-1, 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: Kernels
    axes[0].scatter(X2, y2, color='#1f77b4', edgecolors='k', linewidths=0.5, s=35, label='Data')
    axes[0].plot(xtest, svr_linear.predict(xtest), 'r-', linewidth=1.8, label=f'Linear (MSE={mean_squared_error(y2, svr_linear.predict(X2)):.2f})')
    axes[0].plot(xtest, svr_poly.predict(xtest), 'g-', linewidth=1.8, label=f'Poly d=3 (MSE={mean_squared_error(y2, svr_poly.predict(X2)):.2f})')
    axes[0].plot(xtest, svr_rbf.predict(xtest), 'm-', linewidth=2.0, label=f'RBF (MSE={mean_squared_error(y2, svr_rbf.predict(X2)):.2f})')
    axes[0].set_title("Kernel Comparison for SVR (C=10, ε=0.2)")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("y")
    axes[0].legend(frameon=True)

    # Subplot 2: Epsilon tube effect on RBF
    svr_rbf_tube = sksvm.SVR(kernel='rbf', gamma=0.5, C=10.0, epsilon=0.5).fit(X2, y2)
    plot_svm_regression(svr_rbf_tube, X2, y2, axes_limits, ax=axes[1])
    axes[1].set_title(r"SVR RBF Tube ($\varepsilon=0.5, C=10$) & Support Vectors")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("y")

    plt.tight_layout()
    plt.savefig('svr_synthetic_regression.png', dpi=300)
    plt.close()
    print("Saved 'svr_synthetic_regression.png'")


# =============================================================================
# 3. SVR for Image Inpainting on USPS Handwritten Digits
# =============================================================================
def explore_usps_inpainting(data_dir='USPS'):
    print("\n--- 3. SVR Image Inpainting (USPS Digits) ---")
    uspsXapp = np.load(os.path.join(data_dir, 'uspsXapp.npy'))
    uspsYapp = np.load(os.path.join(data_dir, 'uspsYapp.npy')).ravel()

    # Select representative image (e.g. digit 3, index 456)
    nimg = 456
    im = uspsXapp[:, nimg]
    lab = uspsYapp[nimg] - 1

    # Coordinate grid: (x1, x2) in {0..15} x {0..15}
    x1, x2 = np.meshgrid(np.arange(16), np.arange(16))
    coords = np.column_stack((x1.ravel(), x2.ravel()))  # (256, 2)
    pixel_values = im  # (256,)

    ratios = [0.10, 0.30, 0.50, 0.80]
    fig, axes = plt.subplots(len(ratios), 3, figsize=(10, 11))

    for row_idx, test_ratio in enumerate(ratios):
        xtrain, xtest, ytrain, ytest = sksel.train_test_split(
            coords, pixel_values, test_size=test_ratio, random_state=42
        )

        # Optimize SVR with Gaussian kernel
        svr = sksvm.SVR(kernel='rbf', C=10.0, gamma=0.1, epsilon=0.01)
        svr.fit(xtrain, ytrain)
        y_pred_test = svr.predict(xtest)

        # Construct images:
        # 1. Original
        im_orig = im.reshape((16, 16))

        # 2. Degraded (missing pixels filled with background white +1.0)
        im_degraded = np.ones((16, 16))
        for pt, val in zip(xtrain, ytrain):
            im_degraded[pt[1], pt[0]] = val

        # 3. Reconstructed (combine training pixels + predicted missing pixels)
        im_reconstructed = im_degraded.copy()
        for pt, val in zip(xtest, y_pred_test):
            im_reconstructed[pt[1], pt[0]] = val

        mse = mean_squared_error(ytest, y_pred_test)

        # Plotting
        axes[row_idx, 0].imshow(im_orig, cmap='gray', vmin=-1, vmax=1)
        axes[row_idx, 0].set_title(f"Original (Digit {lab})" if row_idx == 0 else "")
        axes[row_idx, 0].axis('off')

        axes[row_idx, 1].imshow(im_degraded, cmap='gray', vmin=-1, vmax=1)
        axes[row_idx, 1].set_title(f"Degraded ({int(test_ratio*100)}% missing)" if row_idx == 0 else f"{int(test_ratio*100)}% missing")
        axes[row_idx, 1].axis('off')

        axes[row_idx, 2].imshow(im_reconstructed, cmap='gray', vmin=-1, vmax=1)
        axes[row_idx, 2].set_title(f"Reconstructed (MSE={mse:.4f})")
        axes[row_idx, 2].axis('off')

    plt.tight_layout()
    plt.savefig('usps_inpainting.png', dpi=300)
    plt.close()
    print("Saved 'usps_inpainting.png'")


# =============================================================================
# 4. Binary SVM Classification on Real USPS Digits (Digit 3 vs Digit 8)
# =============================================================================
def explore_usps_classification(data_dir='USPS'):
    print("\n--- 4. Binary Classification on USPS Digits ---")
    uspsXapp = np.load(os.path.join(data_dir, 'uspsXapp.npy'))
    uspsYapp = np.load(os.path.join(data_dir, 'uspsYapp.npy')).ravel()
    uspsXtest = np.load(os.path.join(data_dir, 'uspsXtest.npy'))
    uspsYtest = np.load(os.path.join(data_dir, 'uspsYtest.npy')).ravel()

    # Select Class 3 (digit 2/3) and Class 8 (digit 7/8 in 1-based indexing)
    # USPS classes: 1->0, 2->1, 3->2, 4->3, 5->4, 6->5, 7->6, 8->7, 9->8, 10->9
    # We choose label 4 ('3') vs label 9 ('8')
    class_a = 4  # Digit 3
    class_b = 9  # Digit 8
    digit_name_a = "3"
    digit_name_b = "8"

    # Training set: 300 of class A and 300 of class B
    idx_a_train = np.where(uspsYapp == class_a)[0]
    idx_b_train = np.where(uspsYapp == class_b)[0]
    np.random.seed(42)
    np.random.shuffle(idx_a_train)
    np.random.shuffle(idx_b_train)

    train_sel_a = idx_a_train[:300]
    train_sel_b = idx_b_train[:300]
    train_rem_a = idx_a_train[300:]
    train_rem_b = idx_b_train[300:]

    idx_a_test = np.where(uspsYtest == class_a)[0]
    idx_b_test = np.where(uspsYtest == class_b)[0]

    X_train = np.vstack([uspsXapp[:, train_sel_a].T, uspsXapp[:, train_sel_b].T])
    y_train = np.hstack([np.zeros(300, dtype=int), np.ones(300, dtype=int)])

    # Test set: remaining training samples + official test samples for these classes
    X_test_extra = np.vstack([uspsXapp[:, train_rem_a].T, uspsXapp[:, train_rem_b].T])
    y_test_extra = np.hstack([np.zeros(len(train_rem_a), dtype=int), np.ones(len(train_rem_b), dtype=int)])

    X_test_official = np.vstack([uspsXtest[:, idx_a_test].T, uspsXtest[:, idx_b_test].T])
    y_test_official = np.hstack([np.zeros(len(idx_a_test), dtype=int), np.ones(len(idx_b_test), dtype=int)])

    X_test = np.vstack([X_test_extra, X_test_official])
    y_test = np.hstack([y_test_extra, y_test_official])

    print(f"Dataset: Digit '{digit_name_a}' vs Digit '{digit_name_b}'")
    print(f"Train samples: {X_train.shape[0]} ({np.sum(y_train==0)} vs {np.sum(y_train==1)})")
    print(f"Test samples: {X_test.shape[0]} ({np.sum(y_test==0)} vs {np.sum(y_test==1)})")

    # GridSearch CV for (C, gamma)
    param_grid = {
        'C': [0.1, 1, 5, 10, 50, 100],
        'gamma': [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    }
    grid = sksel.GridSearchCV(sksvm.SVC(kernel='rbf'), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)

    best_C = grid.best_params_['C']
    best_gamma = grid.best_params_['gamma']
    best_model = grid.best_estimator_

    train_acc = accuracy_score(y_train, best_model.predict(X_train))
    test_acc = accuracy_score(y_test, best_model.predict(X_test))
    print(f"Optimal Parameters: C={best_C}, gamma={best_gamma}")
    print(f"CV Best Score: {grid.best_score_:.4f}")
    print(f"Train Accuracy: {train_acc:.4f} | Test Accuracy: {test_acc:.4f}")

    # Plot misclassified test digits
    y_pred_test = best_model.predict(X_test)
    mis_idx = np.where(y_pred_test != y_test)[0]
    print(f"Total misclassified test samples: {len(mis_idx)} / {len(y_test)} ({len(mis_idx)/len(y_test):.2%})")

    if len(mis_idx) > 0:
        n_show = min(12, len(mis_idx))
        fig, axes = plt.subplots(2, 6, figsize=(12, 4.5))
        axes = axes.ravel()
        for i in range(n_show):
            idx = mis_idx[i]
            img = X_test[idx].reshape((16, 16))
            true_label = digit_name_a if y_test[idx] == 0 else digit_name_b
            pred_label = digit_name_a if y_pred_test[idx] == 0 else digit_name_b
            axes[i].imshow(img, cmap='gray', vmin=-1, vmax=1)
            axes[i].set_title(f"True: {true_label} | Pred: {pred_label}", color='crimson', fontsize=9)
            axes[i].axis('off')
        for j in range(n_show, len(axes)):
            axes[j].axis('off')
        plt.suptitle(f"Misclassified Test USPS Digits (SVM RBF: C={best_C}, γ={best_gamma})", fontsize=12)
        plt.tight_layout()
        plt.savefig('misclassified_digits.png', dpi=300)
        plt.close()
        print("Saved 'misclassified_digits.png'")

    return X_train, y_train, X_test, y_test, best_C, best_gamma


# =============================================================================
# 5. Sensitivity Analysis: Impact of C and Gamma
# =============================================================================
def explore_sensitivity_analysis(X_train, y_train, X_test, y_test, best_C, best_gamma):
    print("\n--- 5. Parameter Sensitivity & Execution Time Analysis ---")
    
    # 1. Vary gamma with fixed best_C
    gammas = np.logspace(-4, 0, 15)
    train_scores_gamma = []
    test_scores_gamma = []
    times_gamma = []

    for g in gammas:
        t0 = time.time()
        clf = sksvm.SVC(kernel='rbf', C=best_C, gamma=g)
        clf.fit(X_train, y_train)
        fit_time = time.time() - t0
        
        train_scores_gamma.append(accuracy_score(y_train, clf.predict(X_train)))
        test_scores_gamma.append(accuracy_score(y_test, clf.predict(X_test)))
        times_gamma.append(fit_time)

    # 2. Vary C with fixed best_gamma
    c_values = np.logspace(-2, 3, 15)
    train_scores_c = []
    test_scores_c = []
    times_c = []

    for c in c_values:
        t0 = time.time()
        clf = sksvm.SVC(kernel='rbf', C=c, gamma=best_gamma)
        clf.fit(X_train, y_train)
        fit_time = time.time() - t0
        
        train_scores_c.append(accuracy_score(y_train, clf.predict(X_train)))
        test_scores_c.append(accuracy_score(y_test, clf.predict(X_test)))
        times_c.append(fit_time)

    # Plotting Sensitivity Curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Gamma - Accuracy
    axes[0, 0].semilogx(gammas, train_scores_gamma, 'o-', color='#1f77b4', label='Train Accuracy')
    axes[0, 0].semilogx(gammas, test_scores_gamma, 's-', color='#2ca02c', label='Test Accuracy')
    axes[0, 0].axvline(best_gamma, color='crimson', linestyle='--', label=fr'Optimal $\gamma$={best_gamma}')
    axes[0, 0].set_title(r"Accuracy vs $\gamma$ (fixed $C=" + f"{best_C}$)")
    axes[0, 0].set_xlabel(r"$\gamma$ (log scale)")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].legend()

    # Gamma - Training Time
    axes[0, 1].loglog(gammas, times_gamma, '^-', color='#d62728')
    axes[0, 1].axvline(best_gamma, color='crimson', linestyle='--', label=fr'Optimal $\gamma$={best_gamma}')
    axes[0, 1].set_title(r"Training Time vs $\gamma$ (fixed $C=" + f"{best_C}$)")
    axes[0, 1].set_xlabel(r"$\gamma$ (log scale)")
    axes[0, 1].set_ylabel("Fit Time (s)")
    axes[0, 1].legend()

    # C - Accuracy
    axes[1, 0].semilogx(c_values, train_scores_c, 'o-', color='#1f77b4', label='Train Accuracy')
    axes[1, 0].semilogx(c_values, test_scores_c, 's-', color='#2ca02c', label='Test Accuracy')
    axes[1, 0].axvline(best_C, color='crimson', linestyle='--', label=f'Optimal $C$={best_C}')
    axes[1, 0].set_title(r"Accuracy vs $C$ (fixed $\gamma=" + f"{best_gamma}$)")
    axes[1, 0].set_xlabel(r"$C$ (log scale)")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].legend()

    # C - Training Time
    axes[1, 1].loglog(c_values, times_c, '^-', color='#d62728')
    axes[1, 1].axvline(best_C, color='crimson', linestyle='--', label=f'Optimal $C$={best_C}')
    axes[1, 1].set_title(r"Training Time vs $C$ (fixed $\gamma=" + f"{best_gamma}$)")
    axes[1, 1].set_xlabel(r"$C$ (log scale)")
    axes[1, 1].set_ylabel("Fit Time (s)")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig('svm_sensitivity_analysis.png', dpi=300)
    plt.close()
    print("Saved 'svm_sensitivity_analysis.png'")


# =============================================================================
# Main Execution Pipeline
# =============================================================================
def main():
    print("================================================================")
    print("  SUPPORT VECTOR MACHINES & REGRESSION (SVM / SVR) BENCHMARK   ")
    print("================================================================")
    
    # 1. Synthetic Classification
    explore_synthetic_svm()
    
    # 2. Synthetic Regression
    explore_synthetic_svr()
    
    # 3. USPS Inpainting with SVR
    explore_usps_inpainting()
    
    # 4. USPS Digit Binary Classification
    X_train, y_train, X_test, y_test, best_C, best_gamma = explore_usps_classification()
    
    # 5. Sensitivity & Performance Analysis
    explore_sensitivity_analysis(X_train, y_train, X_test, y_test, best_C, best_gamma)
    
    print("\nAll experiments successfully completed and visualizations exported.")


if __name__ == '__main__':
    main()
