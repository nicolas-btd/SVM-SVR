# Support Vector Machines (SVM) et Regression (SVR)

Ce projet explore l'utilisation des machines à vecteurs de support (SVM) pour des problèmes de classification et de régression (SVR). 

## Classification SVM sur données synthétiques

Nous commençons par étudier l'algorithme SVM sur un jeu de données synthétique généré à l'aide de `make_moons`.
Les données ont d'abord été standardisées (moyenne de 0, écart-type de 1) afin d'assurer que toutes les caractéristiques contribuent de manière équivalente à la construction de la marge.

### Modèle Linéaire
Avec un noyau linéaire classique et une constante de régularisation par défaut (`C=1.0`), la frontière de décision est une simple droite qui ne parvient pas à séparer efficacement ce jeu de données fondamentalement non-linéaire.
