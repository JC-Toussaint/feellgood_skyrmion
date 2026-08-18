# Générateur de maillage et d'ansatz de skyrmion pour feeLLGood

## Présentation

Ce projet permet de générer automatiquement :

- un **maillage 3D cylindrique** au format **Gmsh** (`.msh`) ;
- un **ansatz micromagnétique** proche de la solution d'équilibre d'un skyrmion ;
- un fichier de magnétisation `sol.in` directement exploitable par **feeLLGood**.

L'objectif est de fournir une configuration initiale de haute qualité afin de faciliter et d'accélérer les simulations micromagnétiques réalisées avec **feeLLGood**, en démarrant à partir d'un état déjà proche du minimum d'énergie.

## Fonctionnalités

Le programme propose les fonctionnalités suivantes :

- minimisation variationnelle du profil radial d'un skyrmion axisymétrique ;
- recherche automatique d'un minimum d'énergie par plusieurs initialisations de la chiralité ;
- génération d'un maillage volumique cylindrique avec **Gmsh** ;
- création des groupes physiques nécessaires à la simulation ;
- projection de la solution 1D sur le maillage 3D ;
- génération du fichier `sol.in` contenant le champ d'aimantation initial ;
- interface graphique développée avec **PyQt5** ;
- visualisation interactive :
  - du profil radial d'aimantation ;
  - du maillage 3D ;
  - du champ d'aimantation sous forme de glyphes ;
  - de coupes interactives grâce à **PyVista/VTK**.

## Principe

Le programme calcule tout d'abord un profil radial de skyrmion par minimisation de l'énergie micromagnétique.

Cette solution axisymétrique est ensuite projetée sur un maillage tridimensionnel cylindrique afin de construire une distribution d'aimantation complète.

Le résultat constitue un **ansatz** très proche de la solution micromagnétique recherchée, ce qui réduit le temps de relaxation nécessaire dans **feeLLGood**.

## Fichiers générés

Selon les paramètres choisis, le programme produit notamment :

- `cylinderXXXX.msh` : maillage 3D au format Gmsh ;
- `sol.in` : distribution initiale d'aimantation compatible avec feeLLGood.

## Dépendances

Le projet est écrit en Python et utilise notamment :

- NumPy
- SciPy
- Gmsh
- PyQt5
- Matplotlib
- PyVista
- VTK

## Installation

Installer les dépendances Python :

```bash
pip install numpy scipy gmsh pyqt5 matplotlib pyvista pyvistaqt vtk
```

## Utilisation

Lancer l'application :

```bash
python fg-skyrmion.py
```

Depuis l'interface graphique, il est possible de :

1. définir les paramètres physiques (échange, DMI, anisotropie, dimensions du cylindre, etc.) ;
2. choisir la discrétisation et les paramètres du maillage ;
3. lancer le calcul ;
4. visualiser le profil obtenu ;
5. exporter automatiquement le maillage et le fichier `sol.in`.

## Utilisation avec feeLLGood

Les fichiers générés sont destinés à être utilisés comme données d'entrée pour **feeLLGood** :

- le maillage définit la géométrie du problème ;
- le fichier `sol.in` fournit une aimantation initiale déjà proche de la solution micromagnétique, ce qui favorise une convergence plus rapide des calculs.

## Auteur

**Jean-Christophe Toussaint**  
Grenoble INP

## Référence

Ce programme est destiné à préparer des simulations pour le logiciel **feeLLGood** :

https://feellgood.neel.cnrs.fr/

## Licence

Copyright (C) 2026 Jean-Christophe Toussaint.

FeeLLGood_skyrmion is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

The libraries used by feeLLGood_skyrmion are distributed under different licenses, and this is documented in their respective Web sites.
