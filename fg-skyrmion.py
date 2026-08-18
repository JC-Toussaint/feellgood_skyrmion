#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
APPLICATION DE CALCUL D'AIMANTATION DE SKYRMION ET DE GÉNÉRATION DE MAILLAGE 3D
================================================================================
Description :
  - Minimisation variationnelle 1D du profil d'aimantation d'un skyrmion
    axisymétrique (angle polaire theta(r) et angle azimutal phi).
  - Condition aux limites de Brown généralisée évaluée au bord (r = R).
  - Génération d'un maillage 3D d'un cylindre via l'API OpenCASCADE de Gmsh
    avec définition du volume et des régions de surface :
      * Volume_Cylindre (Volume 3D)
      * surface_bas     (Surface Z = -t/2)
      * surface_haut    (Surface Z = +t/2)
      * surface_laterale (Surface cylindrique R = R_max)
  - Export des nœuds et des composantes d'aimantation (mx, my, mz) dans sol.in.
  - Interface graphique PyQt5 avec visualisation du profil 1D (Matplotlib) et
    rendu 3D interactif accéléré matériellement via PyVista/VTK :
      * Surface externe du cylindre (maillage triangulaire) avec contrôle
        d'opacité.
      * Champ de vecteurs d'aimantation affiché sous forme de glyphes
        (flèches) colorés par composante mz, avec sous-échantillonnage
        réglable pour rester fluide sur de gros maillages.
      * Widget de "coupe" interactif (plan de coupe / slab) permettant de
        trancher le volume 3D et de visualiser l'aimantation à l'intérieur
        du cylindre, pas seulement en surface.

Dépendances supplémentaires nécessaires à la visualisation 3D :
    pip install pyvista pyvistaqt vtk
================================================================================
"""

import sys
import os
import math
import socket
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import gmsh

# ------------------------------------------------------------------------------
# Dépendances PyQt5 & Matplotlib pour l'interface graphique et le rendu 3D
# ------------------------------------------------------------------------------
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QFormLayout, QDoubleSpinBox, QSpinBox, QPushButton, QPlainTextEdit,
    QGroupBox, QSplitter, QLabel, QProgressBar, QMessageBox, QTabWidget,
    QCheckBox, QComboBox, QSlider
)

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas
)
from matplotlib.figure import Figure

# ------------------------------------------------------------------------------
# PyVista / VTK pour le rendu 3D interactif (surface + glyphes + plan de coupe)
# ------------------------------------------------------------------------------
import pyvista as pv
from pyvistaqt import QtInteractor


# ==============================================================================
# QUADRATURE DE GAUSS À 2 POINTS (Intégration numérique sur [-1, 1])
# ==============================================================================
# XI : Points d'évaluation (racines du polynôme de Legendre de degré 2)
# WG : Poids associés de la quadrature de Gauss
XI = np.array([-1.0 / np.sqrt(3.0), +1.0 / np.sqrt(3.0)])
WG = np.array([1.0, 1.0])


# ==============================================================================
# NETTOYAGE DU RÉPERTOIRE AU LANCEMENT DU PROGRAMME
# ==============================================================================
def clean_working_directory():
    """
    Supprime les anciens fichiers de maillage (cylinder*.msh) et de solution (sol.in)
    situés dans le même répertoire que le script Python au moment du lancement.
    """
    try:
        try:
            script_dir = Path(__file__).resolve().parent
        except NameError:
            script_dir = Path.cwd()

        # Suppression de tous les fichiers cylinder*.msh
        for mesh_file in script_dir.glob("cylinder*.msh"):
            try:
                mesh_file.unlink()
                print(f"[Nettoyage] Fichier supprimé : {mesh_file.name}")
            except Exception as e:
                print(f"[Nettoyage] Impossible de supprimer {mesh_file.name} : {e}")

        # Suppression du fichier sol.in
        sol_file = script_dir / "sol.in"
        if sol_file.exists():
            try:
                sol_file.unlink()
                print(f"[Nettoyage] Fichier supprimé : {sol_file.name}")
            except Exception as e:
                print(f"[Nettoyage] Impossible de supprimer {sol_file.name} : {e}")

    except Exception as e:
        print(f"[Nettoyage] Erreur globale lors du nettoyage : {e}")


# ==============================================================================
# MODÈLE PHYSIQUE / NUMÉRIQUE DE SKYRMION 1D AXISYMÉTRIQUE
# ==============================================================================
class SkyrmionModel:
    """
    Modélise l'énergie micromagnétique 1D d'un skyrmion axisymétrique.
    Variables d'état :
      - theta(x) : Profil d'orientation de l'aimantation (0 <= x <= 1)
      - phi      : Angle de chirality (constante globale)
    """
    def __init__(self, N, d, k):
        """
        :param N: Nombre de subdivisions spatiales du domaine 1D [0, 1]
        :param d: Constante adimensionnée de DMI (D * R / A)
        :param k: Constante adimensionnée d'anisotropie (Ku * R^2 / A)
        """
        self.N = N
        self.d = d
        self.k = k
        self.x = np.linspace(0.0, 1.0, N + 1)  # Grille adimensionnée x = r / R
        self.h = 1.0 / N                       # Pas d'espace h

    def energy_and_gradient(self, q):
        """
        Calcule l'énergie totale adimensionnée et son gradient analytique par rapport à q.
        
        q contient :
          - q[:N] : Valeurs de theta aux nœuds internes et au bord r=R (theta[0] est fixé à 0)
          - q[N]  : Angle azimutal phi
        
        Densité d'énergie adimensionnée :
          f = x*(dtheta/dx)^2 + sin^2(theta)/x + d*sin(phi)*(x*dtheta/dx + sin(theta)*cos(theta)) - k*x*cos^2(theta)
        """
        N = self.N
        h = self.h
        x = self.x
        d = self.d
        k = self.k

        # Reconstruction du vecteur theta (Condition au centre theta(0) = 0 imposée)
        theta = np.zeros(N + 1)
        theta[1:] = q[:N]
        phi = q[N]

        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)

        grad_theta = np.zeros(N + 1)
        grad_phi = 0.0
        E = 0.0

        # Discrétisation par éléments finis P1 (linéaires)
        theta_L = theta[:-1]
        theta_R = theta[1:]
        theta_x = (theta_R - theta_L) / h  # Dérivée dtheta/dx par élément

        # Intégration sur chaque élément [x_i, x_{i+1}] via la quadrature de Gauss
        for uk, wk in zip(XI, WG):
            # Fonctions de forme linéaires sur [-1, 1]
            N1 = (1.0 - uk) / 2.0
            N2 = (1.0 + uk) / 2.0

            # Position et valeur de theta au point de Gauss
            xg = 0.5 * (x[:-1] + x[1:]) + 0.5 * uk * h
            th = N1 * theta_L + N2 * theta_R

            s = np.sin(th)
            c = np.cos(th)

            # Termes de la densité d'énergie : Échange, DMI, Anisotropie
            f_exchange = xg * theta_x**2 + s**2 / xg
            f_DMI = d * sin_phi * (xg * theta_x + s * c)
            f_anis = -k * xg * c**2

            f = f_exchange + f_DMI + f_anis
            E += np.sum(wk * h / 2.0 * f)

            # Dérivées partielles pour l'assemblage du gradient
            df_dtheta = (
                2.0 * s * c / xg
                + d * sin_phi * np.cos(2.0 * th)
                + 2.0 * k * xg * s * c
            )
            df_dtheta_x = 2.0 * xg * theta_x + d * sin_phi * xg

            contribution_L = (df_dtheta * N1 - df_dtheta_x / h) * (wk * h / 2.0)
            contribution_R = (df_dtheta * N2 + df_dtheta_x / h) * (wk * h / 2.0)

            grad_theta[:-1] += contribution_L
            grad_theta[1:]  += contribution_R

            # Dérivée par rapport à l'angle phi
            df_dphi = d * cos_phi * (xg * theta_x + s * c)
            grad_phi += np.sum(wk * h / 2.0 * df_dphi)

        # Extraction du gradient sans le premier degré de liberté (theta(0) fixed)
        gradient = np.concatenate([grad_theta[1:], [grad_phi]])
        return E, gradient

    def initial_profile_from_formula(self, formula_str):
        """
        Évalue l'expression analytique saisie par l'utilisateur pour le profil initial theta(x).
        """
        local_dict = {
            "x": self.x, "np": np, "pi": np.pi, "sin": np.sin,
            "cos": np.cos, "tan": np.tan, "tanh": np.tanh,
            "exp": np.exp, "sqrt": np.sqrt, "abs": np.abs,
        }
        result = eval(formula_str, {"__builtins__": None}, local_dict)
        theta = np.asarray(result, dtype=float)
        if theta.ndim == 0:
            theta = np.full_like(self.x, theta, dtype=float)
        theta[0] = 0.0  # Respect de la condition d'axe theta(0) = 0
        return theta

    def minimize_configuration(self, theta0, phi_initial, maxiter):
        """
        Exécute l'algorithme L-BFGS-B pour trouver le minimum d'énergie local.
        """
        N = self.N
        q0 = np.concatenate([theta0[1:], [phi_initial]])

        # Bornes physiques : theta in [0, pi], phi in [-pi, pi]
        bounds = [(0.0, np.pi) for _ in range(N)]
        bounds.append((-np.pi, np.pi))

        result = minimize(
            fun=lambda q: self.energy_and_gradient(q)[0],
            x0=q0,
            jac=lambda q: self.energy_and_gradient(q)[1],
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": maxiter, "ftol": 1e-12,
                "gtol": 1e-9, "maxls": 50, "maxcor": 20
            },
        )

        theta = np.zeros(N + 1)
        theta[1:] = result.x[:N]
        phi = result.x[N]
        return result, theta, phi


# Angles d'initialisation de phi pour explorer les différentes chiralités
PHI_INITIALIZATIONS = np.linspace(-1, 1, 5) * np.pi

def run_skyrmion_1d(R, A, D, Ku, N_disc, maxiter, formula_str, progress_callback=None, step_callback=None):
    """
    Pilote le calcul 1D :
      1. Adimensionnement des grandeurs physiques.
      2. Minimisation multi-départ sur phi_0 pour garantir la recherche du minimum global.
      3. Calcul de la condition aux limites de Brown généralisée en r = R :
         [ 2 * dtheta/dx + d * sin(phi) ] = 0
    """
    d = D * R / A
    k = Ku * R**2 / A

    model = SkyrmionModel(N_disc, d, k)
    theta_init = model.initial_profile_from_formula(formula_str)

    solutions = []
    for i, phi0 in enumerate(PHI_INITIALIZATIONS):
        if progress_callback:
            progress_callback(f"Minimisation {i + 1}/{len(PHI_INITIALIZATIONS)} (phi_0={phi0:.2f}) ...")

        res, theta_sol, phi_sol = model.minimize_configuration(theta_init, phi0, maxiter)
        solutions.append((res.fun, theta_sol, phi_sol, res))

        if step_callback:
            step_callback(i + 1)

    # Tri par énergie minimale décroissante (le premier élément est le minimum global)
    solutions.sort(key=lambda item: item[0])
    Emin, theta, phi, _ = solutions[0]

    r_nm = model.x * R * 1e9  # Conversion du rayon en nanomètres

    # Calcul numérique de dtheta/dx en r = R par différence finie d'ordre 2 au bord droit
    h = 1.0 / N_disc
    dtheta_dx_R = (3.0 * theta[-1] - 4.0 * theta[-2] + theta[-3]) / (2.0 * h)
    
    # Évaluation du terme de la condition aux limites de Brown :
    brown_cond = 2.0 * dtheta_dx_R + d * np.sin(phi)

    return {
        "x": model.x,
        "r_nm": r_nm,
        "theta": theta,
        "phi": phi,
        "E": Emin,
        "d": d,
        "k": k,
        "brown_cond": brown_cond
    }


# ==============================================================================
# GÉNÉRATION DU FICHIER SOL.IN (0-INDEXÉ)
# ==============================================================================
def generate_sol_file(node_tags, node_coords, r_profile, theta_profile, phi_val, R_val, filename="sol.in"):
    """
    Projete la solution 1D theta(r) sur la géométrie 3D et écrit le fichier sol.in.
    Indexation des nœuds : 0-indexée pour compatibilité avec les solveurs FEM/FFT.
    """
    num_nodes = len(node_tags)
    coords = np.asarray(node_coords, dtype=np.float64).reshape(-1, 3)

    px = coords[:, 0]
    py = coords[:, 1]

    # Calcul du rayon polaire et de l'angle azimutal pour chaque nœud du maillage 3D
    r_nodes = np.sqrt(px**2 + py**2)
    x_normalized = np.clip(r_nodes / R_val, 0.0, 1.0)
    chi = np.arctan2(py, px)

    # Interpolation du profil 1D theta(r) sur chaque nœud 3D
    theta_nodes = np.interp(x_normalized, r_profile, theta_profile)

    # Composantes locales de l'aimantation dans le repère cylindrique
    m_r = np.sin(theta_nodes) * np.cos(phi_val)
    m_phi = np.sin(theta_nodes) * np.sin(phi_val)
    m_z = np.cos(theta_nodes)

    # Conversion des composantes dans le repère cartésien 3D (mx, my, mz)
    mx = m_r * np.cos(chi) - m_phi * np.sin(chi)
    my = m_r * np.sin(chi) + m_phi * np.cos(chi)
    mz = m_z

    m = np.column_stack((mx, my, mz))

    # En-tête normalisé
    now = datetime.now().isoformat()
    hostname = socket.gethostname()

    header_lines = [
        "## generator: app_cylinder_skyrmion",
        f"## hostname: {hostname}",
        f"## real-world time: {now}",
        "## time: 0",
        "## columns: idx\tmx\tmy\tmz\tphi"
    ]

    # Convertir les identifiants Gmsh (1-indexés) en indices 0-indexés
    idx = (np.array(node_tags, dtype=np.int64) - 1)[:, np.newaxis]
    phi_col = np.zeros((num_nodes, 1))
    tab = np.hstack((idx, m, phi_col))

    # Tri explicite par identifiant de nœud croissant
    tab = tab[np.argsort(tab[:, 0])]

    with open(filename, "w", encoding="utf-8") as f:
        for line in header_lines:
            f.write(line + "\n")
        np.savetxt(f, tab, fmt=["%d", "%+.7e", "%+.7e", "%+.7e", "%.7e"], delimiter="\t")


# ==============================================================================
# WORKER EN THREAD SÉPARÉ (Empêche le gel du GUI PyQt5 durant le maillage)
# ==============================================================================
class SimulationWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(str)
    progress_step = QtCore.pyqtSignal(int)
    finished_ok = QtCore.pyqtSignal(dict)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.params = params

    def run(self):
        try:
            p = self.params

            # Étape 1 : Résolution du profil 1D
            res_1d = run_skyrmion_1d(
                R=p["R"], A=p["A"], D=p["D"], Ku=p["Ku"],
                N_disc=p["N_disc"], maxiter=p["maxiter"], formula_str=p["formula_str"],
                progress_callback=self.progress.emit, step_callback=self.progress_step.emit
            )

            # Étape 2 : Construction de la géométrie et maillage 3D via Gmsh
            self.progress.emit("Génération du maillage 3D du cylindre avec Gmsh...")

            gmsh.clear()
            gmsh.option.setNumber("General.Terminal", 0)  # Silence console Gmsh

            # Activation du multi-threading sur le processeur hôte
            num_threads = os.cpu_count() or 1
            gmsh.option.setNumber("General.NumThreads", num_threads)

            gmsh.model.add("cylinder")

            R_nm = p["R"] * 1e9
            t_nm = p["t"] * 1e9

            # Création du cylindre volumique centré en Z (-t/2 à +t/2) via le noyau OpenCASCADE
            cylinder = gmsh.model.occ.addCylinder(0.0, 0.0, -t_nm / 2.0, 0.0, 0.0, t_nm, R_nm)
            gmsh.model.occ.synchronize()

            # --- DÉFINITION DES GROUPES PHYSIQUES (VOLUME & SURFACES) ---
            # 1. Groupe physique 3D : Volume principal
            gmsh.model.addPhysicalGroup(3, [cylinder], name="volume_cylindre")

            # 2. Identification géométrique des surfaces de bord (2D)
            surfaces = gmsh.model.getBoundary([(3, cylinder)], oriented=False)
            
            lat_surf = []
            top_surf = []
            bot_surf = []
            eps = 1e-3 * t_nm  # Tolérance sur l'altitude Z

            for dim, tag in surfaces:
                com = gmsh.model.occ.getCenterOfMass(dim, tag)
                z_c = com[2]
                if abs(z_c - (-t_nm / 2.0)) < eps:
                    bot_surf.append(tag)      # Surface inférieure Z = -t/2
                elif abs(z_c - (t_nm / 2.0)) < eps:
                    top_surf.append(tag)      # Surface supérieure Z = +t/2
                else:
                    lat_surf.append(tag)      # Surface cylindrique latérale

            # 3. Création des groupes physiques surfaciques 2D
            if bot_surf:
                gmsh.model.addPhysicalGroup(2, bot_surf, name="surface_bas")
            if top_surf:
                gmsh.model.addPhysicalGroup(2, top_surf, name="surface_haut")
            if lat_surf:
                gmsh.model.addPhysicalGroup(2, lat_surf, name="surface_laterale")

            # Réglages des finesses de maillage
            gmsh.option.setNumber("Mesh.MeshSizeMin", p["hmin"])
            gmsh.option.setNumber("Mesh.MeshSizeMax", p["hmax"])
            gmsh.model.mesh.generate(3)

            # Extraction des nœuds et des coordonnées
            node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
            num_nodes = len(node_tags)

            try:
                script_dir = Path(__file__).resolve().parent
            except NameError:
                script_dir = Path.cwd()

            mesh_filename = script_dir / f"cylinder{num_nodes}.msh"
            gmsh.write(str(mesh_filename))

            # Étape 3 : Écriture du fichier de solution d'aimantation sol.in
            sol_filename = script_dir / "sol.in"
            self.progress.emit(f"Génération du fichier {sol_filename.name} (numérotation 0-indexée)...")

            generate_sol_file(
                node_tags=node_tags,
                node_coords=node_coords,
                r_profile=res_1d["x"],
                theta_profile=res_1d["theta"],
                phi_val=res_1d["phi"],
                R_val=R_nm,
                filename=str(sol_filename)
            )

            # Étape 4 : Extraction de la topologie 2D de surface pour le rendu graphique Matplotlib
            coords = np.asarray(node_coords, dtype=np.float64).reshape(-1, 3)
            tag_to_index = np.full(np.max(node_tags) + 1, -1, dtype=int)
            tag_to_index[node_tags] = np.arange(len(node_tags))

            # Récupération des éléments 2D (Triangles constituant la peau externe du cylindre)
            elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
            triangles_list = []
            for etype, enodes in zip(elem_types, elem_node_tags):
                if etype == 2:  # Triangle à 3 nœuds
                    tris = enodes.reshape(-1, 3)
                    triangles_list.append(tag_to_index[tris])
                elif etype == 3:  # Quadrangle à 4 nœuds (re-découpés en 2 triangles)
                    quads = tag_to_index[enodes.reshape(-1, 4)]
                    triangles_list.append(quads[:, [0, 1, 2]])
                    triangles_list.append(quads[:, [0, 2, 3]])

            if triangles_list:
                triangles = np.vstack(triangles_list)
            else:
                triangles = np.empty((0, 3), dtype=int)

            # Calcul du champ vectoriel complet sur tous les nœuds
            r_nodes = np.sqrt(coords[:, 0]**2 + coords[:, 1]**2)
            x_norm = np.clip(r_nodes / R_nm, 0.0, 1.0)
            chi = np.arctan2(coords[:, 1], coords[:, 0])
            theta_nodes = np.interp(x_norm, res_1d["x"], res_1d["theta"])

            phi_val = res_1d["phi"]
            m_r = np.sin(theta_nodes) * np.cos(phi_val)
            m_phi = np.sin(theta_nodes) * np.sin(phi_val)
            m_z = np.cos(theta_nodes)

            mx = m_r * np.cos(chi) - m_phi * np.sin(chi)
            my = m_r * np.sin(chi) + m_phi * np.cos(chi)
            mz = m_z

            m_all = np.column_stack((mx, my, mz))

            gmsh.finalize()

            res_1d["mesh_filename"] = mesh_filename.name
            res_1d["sol_filename"] = sol_filename.name
            res_1d["num_nodes"] = num_nodes
            res_1d["coords"] = coords
            res_1d["triangles"] = triangles
            res_1d["m_all"] = m_all

            self.finished_ok.emit(res_1d)

        except Exception:
            if gmsh.isInitialized():
                gmsh.finalize()
            self.failed.emit(traceback.format_exc())


# ==============================================================================
# ONGLET 1 : PANNEAU GRAPHIQUE PROFIL 1D
# ==============================================================================
class MagnetizationPlotPanel(QtWidgets.QFrame):
    """
    Panneau 2D traçant les composantes mx(r), my(r), mz(r) le long du rayon.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self.figure = Figure(constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        layout.addWidget(self.canvas)

        self.ax = self.figure.add_subplot(111)
        self._init_axes()

    def _init_axes(self):
        self.ax.clear()
        self.ax.set_xlabel("r (nm)")
        self.ax.set_ylabel("Composantes de m")
        self.ax.set_title(r"Distribution d'aimantation $m_x, m_y, m_z(r)$")
        self.ax.axhline(0, linestyle="--", linewidth=1, color="gray")
        self.ax.set_ylim(-1.05, 1.05)
        self.ax.grid(True)

    def update_plot(self, r_nm, theta, phi):
        self._init_axes()
        mx = np.sin(theta) * np.cos(phi)
        my = np.sin(theta) * np.sin(phi)
        mz = np.cos(theta)

        self.ax.plot(r_nm, mx, linewidth=2, label=r"$m_x$", color="tab:red")
        self.ax.plot(r_nm, my, linewidth=2, label=r"$m_y$", color="tab:green")
        self.ax.plot(r_nm, mz, linewidth=2, label=r"$m_z$", color="tab:blue")
        self.ax.legend(loc="best")

        r_min, r_max = r_nm.min(), r_nm.max()
        marge = 0.05 * (r_max - r_min) if r_max > r_min else 1.0
        self.ax.set_xlim(r_min - marge, r_max + marge)
        self.canvas.draw()


# ==============================================================================
# ONGLET 1 : CONDITION DE BROWN
# ==============================================================================
class PickedValuePanel(QtWidgets.QFrame):
    """
    Affiche l'angle phi optimal retenu et la valeur numérique calculée pour la condition au bord r = R.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        group = QGroupBox("Résultats physiques retenus & Vérification au bord (r = R)")
        group_layout = QVBoxLayout(group)

        self.label_phi = QLabel("Angle azimutal optimal φ : -")
        self.label_phi.setAlignment(QtCore.Qt.AlignCenter)

        self.label_brown = QLabel("Condition de Brown généralisée en r = R : -")
        self.label_brown.setAlignment(QtCore.Qt.AlignCenter)

        mono_font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        mono_font.setPointSize(10)
        self.label_phi.setFont(mono_font)
        self.label_brown.setFont(mono_font)

        group_layout.addWidget(self.label_phi)
        group_layout.addWidget(self.label_brown)
        layout.addWidget(group)

    def set_results(self, phi_val, brown_val):
        phi_deg = np.degrees(phi_val)
        self.label_phi.setText(f"Angle azimutal optimal φ : {phi_val:.6f} rad ({phi_deg:.2f}°)")

        self.label_brown.setText(
            f"Condition de Brown généralisée en r = R : [ 2·dθ/dx + d·sin(φ) ] = {brown_val:.4e}"
        )
        
        # Coloration en rouge si le module dépasse 1e-3, sinon couleur par défaut
        if abs(brown_val) > 1e-3:
            self.label_brown.setStyleSheet("color: red; font-weight: bold;")
        else:
            self.label_brown.setStyleSheet("")


# ==============================================================================
# ONGLET 2 : REPRÉSENTATION 3D INTERACTIVE (PYVISTA/VTK) — SURFACE ET GLYPHES
#            VECTORIELS
# ==============================================================================
class Surface3DPlotPanel(QtWidgets.QWidget):
    """
    Affichage 3D interactif et accéléré matériellement (VTK, via PyVista) du
    cylindre maillé et du champ d'aimantation.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        # --------------------------------------------------------------------
        # Barre de réglages interactifs
        # --------------------------------------------------------------------
        control_layout = QHBoxLayout()

        control_layout.addWidget(QLabel("Longueur des flèches :"))
        self.spin_arrow_len = QDoubleSpinBox()
        self.spin_arrow_len.setRange(0.1, 100.0)
        self.spin_arrow_len.setValue(4.0)
        self.spin_arrow_len.setSingleStep(0.5)
        self.spin_arrow_len.setSuffix(" nm")
        control_layout.addWidget(self.spin_arrow_len)

        control_layout.addSpacing(15)

        control_layout.addWidget(QLabel("Max flèches 3D :"))
        self.spin_max_arrows = QSpinBox()
        self.spin_max_arrows.setRange(100, 20000)
        self.spin_max_arrows.setValue(1500)
        self.spin_max_arrows.setSingleStep(100)
        control_layout.addWidget(self.spin_max_arrows)

        control_layout.addSpacing(15)

        control_layout.addWidget(QLabel("Opacité surface :"))
        self.slider_opacity = QSlider(QtCore.Qt.Horizontal)
        self.slider_opacity.setRange(0, 100)
        self.slider_opacity.setValue(100)
        self.slider_opacity.setFixedWidth(110)
        control_layout.addWidget(self.slider_opacity)

        control_layout.addStretch()
        layout.addLayout(control_layout)

        control_layout_2 = QHBoxLayout()

        self.chk_show_surface = QCheckBox("Afficher la surface")
        self.chk_show_surface.setChecked(True)
        control_layout_2.addWidget(self.chk_show_surface)

        self.chk_show_arrows = QCheckBox("Afficher les flèches (surface)")
        self.chk_show_arrows.setChecked(True)
        control_layout_2.addWidget(self.chk_show_arrows)

        control_layout_2.addWidget(QLabel("Colorer surface par :"))
        self.combo_color_by = QComboBox()
        self.combo_color_by.addItems(["m_z", "m_x", "m_y"])
        control_layout_2.addWidget(self.combo_color_by)

        control_layout_2.addStretch()
        layout.addLayout(control_layout_2)

        # --------------------------------------------------------------------
        # Zone de rendu 3D interactive PyVista (widget Qt intégré, VTK natif)
        # --------------------------------------------------------------------
        self.plotter = QtInteractor(self)
        self.plotter.set_background("white")
        layout.addWidget(self.plotter.interactor)

        # Données brutes issues du worker de simulation
        self.coords = None
        self.triangles = None
        self.m_all = None

        # Références vers les acteurs VTK affichés (pour mise à jour ciblée)
        self._surface_actor = None
        self._arrow_actor = None

        # Connexion des signaux de mise à jour dynamique
        self.spin_arrow_len.valueChanged.connect(self.redraw_plot)
        self.spin_max_arrows.valueChanged.connect(self.redraw_plot)
        self.slider_opacity.valueChanged.connect(self._on_opacity_changed)
        self.chk_show_surface.toggled.connect(self.redraw_plot)
        self.chk_show_arrows.toggled.connect(self.redraw_plot)
        self.combo_color_by.currentIndexChanged.connect(self.redraw_plot)

    def set_data(self, coords, triangles, m_all, tetra=None):
        self.coords = np.asarray(coords, dtype=np.float64)
        self.triangles = np.asarray(triangles, dtype=np.int64) if triangles is not None else None
        self.m_all = np.asarray(m_all, dtype=np.float64)

        self.redraw_plot(reset_camera=True)

    def _scalar_field(self, m_all):
        choice = self.combo_color_by.currentText()
        if choice == "m_z":
            return m_all[:, 2], "m_z"
        elif choice == "m_x":
            return m_all[:, 0], "m_x"
        else:
            return m_all[:, 1], "m_y"

    def _on_opacity_changed(self):
        if self._surface_actor is not None:
            self._surface_actor.GetProperty().SetOpacity(self.slider_opacity.value() / 100.0)
            self.plotter.render()

    def redraw_plot(self, reset_camera=False):
        self.plotter.clear()
        self._surface_actor = None
        self._arrow_actor = None

        if self.coords is None or self.m_all is None or len(self.coords) == 0:
            self.plotter.add_text("Aucune donnée à afficher", font_size=12, color="black")
            self.plotter.render()
            return

        coords = self.coords
        triangles = self.triangles
        m_all = self.m_all

        scalars, scalar_name = self._scalar_field(m_all)

        # 1. SURFACE EXTERNE DU CYLINDRE
        if self.chk_show_surface.isChecked() and triangles is not None and len(triangles) > 0:
            n_tri = len(triangles)
            faces = np.hstack([np.full((n_tri, 1), 3, dtype=np.int64), triangles]).ravel()
            surf_mesh = pv.PolyData(coords, faces)
            surf_mesh["scalars"] = scalars

            self._surface_actor = self.plotter.add_mesh(
                surf_mesh,
                scalars="scalars",
                cmap="coolwarm",
                opacity=self.slider_opacity.value() / 100.0,
                show_edges=True,
                edge_color="#1f4e78",
                line_width=0.4,
                smooth_shading=True,
                scalar_bar_args={"title": scalar_name},
            )

        # 2. GLYPHES VECTORIELS (flèches noires)
        if self.chk_show_arrows.isChecked():
            if triangles is not None and len(triangles) > 0:
                surface_node_indices = np.unique(triangles)
            else:
                surface_node_indices = np.arange(len(coords))

            max_arrows = self.spin_max_arrows.value()
            arrow_length = self.spin_arrow_len.value()

            n_pts = len(surface_node_indices)
            if n_pts > max_arrows:
                step = max(1, n_pts // max_arrows)
                sub_idx = surface_node_indices[::step]
            else:
                sub_idx = surface_node_indices

            arrow_cloud = pv.PolyData(coords[sub_idx])
            arrow_cloud["vectors"] = m_all[sub_idx]
            arrow_cloud.set_active_vectors("vectors")

            glyphs = arrow_cloud.glyph(
                orient="vectors",
                scale=False,
                factor=arrow_length,
                geom=pv.Arrow(),
            )

            self._arrow_actor = self.plotter.add_mesh(
                glyphs,
                color="black",
                show_scalar_bar=False,
                lighting=True,
            )

        self.plotter.add_axes()
        if reset_camera:
            self.plotter.reset_camera()
        self.plotter.render()


# ==============================================================================
# FENÊTRE PRINCIPALE DU PROGRAMME
# ==============================================================================
class MainWindow(QMainWindow):
    """
    Fenêtre principale regroupant la saisie des paramètres,
    le suivi d'exécution du worker et les panneaux de résultats.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Générateur de Maillage Cylindrique & Aimantation Skyrmion")
        self.resize(1200, 750)

        self.worker = None

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        splitter = QSplitter(QtCore.Qt.Horizontal)
        main_layout.addWidget(splitter)

        # ----------------------------------------------------------------------
        # PANNEAU DE GAUCHE : FORMULAIRE DE PARAMÈTRES ET CONSOLE
        # ----------------------------------------------------------------------
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        param_group = QGroupBox("Paramètres de la Simulation")
        form_layout = QFormLayout()

        self.spin_R = QDoubleSpinBox()
        self.spin_R.setRange(1.0, 1000.0); self.spin_R.setValue(50.0); self.spin_R.setSuffix(" nm")

        self.spin_t = QDoubleSpinBox()
        self.spin_t.setRange(0.1, 1000.0); self.spin_t.setValue(10.0); self.spin_t.setSuffix(" nm")

        self.spin_A = QDoubleSpinBox()
        self.spin_A.setRange(0.01, 100.0); self.spin_A.setValue(1.0)
        form_layout.addRow("Échange A (x1e-11 J/m):", self.spin_A)

        self.spin_D = QDoubleSpinBox()
        self.spin_D.setRange(-50.0, 50.0); self.spin_D.setValue(2.0)
        form_layout.addRow("DMI D (x1e-3 J/m^2):", self.spin_D)

        self.spin_Ku = QDoubleSpinBox()
        self.spin_Ku.setRange(-100.0, 100.0); self.spin_Ku.setValue(5.0)
        form_layout.addRow("Anisotropie Ku (x1e5 J/m^3):", self.spin_Ku)

        self.spin_N_disc = QSpinBox()
        self.spin_N_disc.setRange(100, 5000); self.spin_N_disc.setValue(1000)

        self.spin_hmin = QDoubleSpinBox()
        self.spin_hmin.setRange(0.1, 100.0); self.spin_hmin.setValue(2.0); self.spin_hmin.setSuffix(" nm")

        self.spin_hmax = QDoubleSpinBox()
        self.spin_hmax.setRange(0.1, 100.0); self.spin_hmax.setValue(4.0); self.spin_hmax.setSuffix(" nm")

        self.edit_formula = QtWidgets.QLineEdit("0 * x")

        form_layout.addRow("Rayon (R):", self.spin_R)
        form_layout.addRow("Épaisseur (t):", self.spin_t)
        form_layout.addRow("Discrétisation 1D (N):", self.spin_N_disc)
        form_layout.addRow("Taille maillage min:", self.spin_hmin)
        form_layout.addRow("Taille maillage max:", self.spin_hmax)
        form_layout.addRow("Profil initial theta(x):", self.edit_formula)

        param_group.setLayout(form_layout)
        left_layout.addWidget(param_group)

        self.btn_run = QPushButton("🚀 Générer le maillage & sol.in")
        self.btn_run.setStyleSheet("font-weight: bold; font-size: 13px; padding: 8px;")
        self.btn_run.clicked.connect(self.start_simulation)
        left_layout.addWidget(self.btn_run)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, len(PHI_INITIALIZATIONS))
        self.progress_bar.setValue(0)
        left_layout.addWidget(self.progress_bar)

        left_layout.addWidget(QLabel("Journal d'exécution :"))
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        mono_font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        mono_font.setPointSize(9)
        self.log_text.setFont(mono_font)
        left_layout.addWidget(self.log_text)

        left_panel.setMaximumWidth(400)
        splitter.addWidget(left_panel)

        # ----------------------------------------------------------------------
        # PANNEAU DE DROITE : DEUX ONGLETS (PROFIL 1D ET VUE 3D OPAQUE)
        # ----------------------------------------------------------------------
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)

        self.tabs = QTabWidget()

        # Onglet 1 : Profil 1D et valeur au bord (Condition de Brown)
        tab1 = QWidget()
        tab1_layout = QVBoxLayout(tab1)
        self.mag_panel = MagnetizationPlotPanel()
        self.picked_panel = PickedValuePanel()
        tab1_layout.addWidget(self.mag_panel, stretch=1)
        tab1_layout.addWidget(self.picked_panel, stretch=0)
        self.tabs.addTab(tab1, "📈 Profil 1D & Condition de Brown")

        # Onglet 2 : Rendu 3D interactif PyVista/VTK (surface, glyphes)
        self.surface_3d_panel = Surface3DPlotPanel()
        self.tabs.addTab(self.surface_3d_panel, "🌐 Surface 3D & Aimantation")

        right_layout.addWidget(self.tabs)
        splitter.addWidget(right_container)
        splitter.setSizes([380, 820])

    def start_simulation(self):
        if self.worker is not None and self.worker.isRunning():
            return

        # Initialisation explicite de l'API Gmsh sur le thread principal UI
        if not gmsh.isInitialized():
            gmsh.initialize()

        params = {
            "R": self.spin_R.value() * 1e-9,
            "t": self.spin_t.value() * 1e-9,
            "A": self.spin_A.value() * 1e-11,
            "D": self.spin_D.value() * 1e-3,
            "Ku": self.spin_Ku.value() * 1e5,
            "N_disc": self.spin_N_disc.value(),
            "maxiter": 5000,
            "hmin": self.spin_hmin.value(),
            "hmax": self.spin_hmax.value(),
            "formula_str": self.edit_formula.text().strip(),
        }

        self.btn_run.setEnabled(False)
        self.log_text.clear()
        self.progress_bar.setValue(0)

        # Lancement du calcul dans un thread séparé
        self.worker = SimulationWorker(params)
        self.worker.progress.connect(self.log_text.appendPlainText)
        self.worker.progress_step.connect(self.progress_bar.setValue)
        self.worker.finished_ok.connect(self.on_simulation_finished)
        self.worker.failed.connect(self.on_simulation_failed)
        self.worker.start()

    def on_simulation_finished(self, results):
        self.btn_run.setEnabled(True)

        phi_deg = np.degrees(results['phi'])

        self.log_text.appendPlainText("\n=== Bilan d'exécution ===")
        self.log_text.appendPlainText(f"• Maillage généré : {results['mesh_filename']}")
        self.log_text.appendPlainText(f"• Fichier sol généré : {results['sol_filename']}")
        self.log_text.appendPlainText(f"• Nombre total de nœuds : {results['num_nodes']}")
        self.log_text.appendPlainText(f"• Énergie minimale E : {results['E']:.6f}")
        self.log_text.appendPlainText(f"• Angle azimutal optimal φ : {results['phi']:.6f} rad ({phi_deg:.2f}°)")
        self.log_text.appendPlainText(f"• Condition de Brown (r=R) : {results['brown_cond']:.4e}")

        # Mise à jour des graphiques dans l'onglet 1
        self.mag_panel.update_plot(results["r_nm"], results["theta"], results["phi"])
        self.picked_panel.set_results(results["phi"], results["brown_cond"])

        # Mise à jour de la vue 3D dans l'onglet 2
        if "coords" in results and "triangles" in results and "m_all" in results:
            self.surface_3d_panel.set_data(
                results["coords"],
                results["triangles"],
                results["m_all"]
            )

    def on_simulation_failed(self, error_message):
        self.btn_run.setEnabled(True)
        QMessageBox.critical(self, "Erreur", f"Une erreur s'est produite :\n\n{error_message}")


# ==============================================================================
# POINT D'ENTRÉE PRINCIPAL DE L'APPLICATION
# ==============================================================================
def main():
    # Nettoyage sécurisé des anciens fichiers cylinder*.msh et sol.in au démarrage
    clean_working_directory()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()