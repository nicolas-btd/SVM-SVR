import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as skdata
import sklearn.preprocessing as skprep
import sklearn.svm as sksvm
import sklearn.model_selection as sksel

def plot_frontiere(clf, window):
    x0s = np.linspace(window[0], window[1], 100)
    x1s = np.linspace(window[2], window[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    y_decision = clf.decision_function(X).reshape(x0.shape)
    plt.contourf(x0, x1, y_pred,  alpha=0.2)
    plt.contourf(x0, x1, y_decision,  alpha=0.1)

if __name__ == '__main__':
    # 1. Pratique sur des données synthétiques - SVM
    # Génération des données
    X1, y1 = skdata.make_moons(n_samples=400, noise=0.15, random_state=42)
    
    # Standardisation
    scaler = skprep.StandardScaler()
    X1norm = scaler.fit_transform(X1)
    
    minX = np.floor(X1norm.min(axis=0))
    maxX = np.ceil(X1norm.max(axis=0))
    
    # Création du modèle SVM (noyau linéaire)
    svm_model = sksvm.SVC(kernel='linear', C=1.0)
    svm_model.fit(X1norm, y1)
    
    plt.figure(figsize=(8, 6))
    plot_frontiere(svm_model, [minX[0], maxX[0], minX[1], maxX[1]])
    
    # Affichage des points
    plt.scatter(X1norm[:, 0], X1norm[:, 1], c=y1, cmap='winter', edgecolors='k')
    plt.title("SVM Frontière de décision (Noyau linéaire)")
    plt.savefig('svm_lineaire.png')
    plt.close()
