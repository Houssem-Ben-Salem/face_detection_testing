# Suite de Tests pour Algorithmes de Détection de Visages

Une plateforme complète pour évaluer et comparer différents algorithmes de détection de visages sous diverses conditions.

## Auteurs
- Ben Salem Houssem
- Sauvé Catherine

## Aperçu

Ce projet fournit une interface conviviale pour tester et comparer divers algorithmes de détection de visages dans différentes conditions. Il comprend à la fois des approches traditionnelles et des approches modernes basées sur l'apprentissage profond, permettant aux utilisateurs d'évaluer les performances des algorithmes dans différents scénarios tels que les variations de pose et les conditions d'éclairage.

## Fonctionnalités

### Algorithmes Implémentés
1. Méthodes Traditionnelles
   - Viola-Jones (Cascades de Haar OpenCV)
   - HOG + SVM (Implémentation DLib)

2. Méthodes Basées sur l'Apprentissage Automatique Moderne
   - MediaPipe Face
   - MTCNN (Multi-task Cascaded CNN)
   - RetinaFace

3. Détection d'Objets Générale
   - YOLOv8 (avec modèle personnalisé pour la détection de visages)

### Conditions de Test
- **Variations de Pose**
  - Lacet (Yaw) : -90° à +90°
  - Tangage (Pitch) : -45° à +45°
  - Roulis (Roll) : -30° à +30°

- **Conditions d'Éclairage**
  - Illumination : 1-1000 lux
  - Direction : Multidirectionnelle
  - Température de couleur : 2000K-7000K

### Caractéristiques Principales
- Tests en temps réel via webcam
- Traitement de fichiers vidéo
- Vue comparative côte à côte
- Optimisation des performances avec saut d'images
- Guide de référence complet des algorithmes
- Traitement multi-thread avec mise en mémoire tampon des images

## Installation

1. Cloner le dépôt :
```bash
git clone https://github.com/yourusername/face-detection-testing.git
cd face-detection-testing
```

2. Créer et activer un environnement virtuel :
```bash
python -m venv myenv
source myenv/bin/activate  # Sur Windows utiliser : myenv\Scripts\activate
```

3. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## Structure du Projet
```
face_detection_testing/
├── app.py                 # Application Gradio principale
├── algorithm_reference.py # Référence de comparaison des algorithmes
├── requirements.txt       # Dépendances du projet
├── config/
│   └── algorithms.yaml    # Configurations des algorithmes
├── algorithms/
│   ├── __init__.py
│   ├── base.py           # Classe de base des algorithmes
│   ├── traditional/
│   │   ├── viola_jones.py
│   │   └── hog_svm.py
│   ├── ml_based/
│   │   ├── mediapipe.py
│   │   ├── mtcnn.py
│   │   └── retinaface.py
│   └── object_detection/
│       └── yolo.py
└── utils/
    ├── variations.py      # Variations de pose et d'éclairage
    └── __init__.py       
```

## Utilisation

1. Démarrer l'application :
```bash
python app.py
```

2. Ouvrir votre navigateur web et naviguer vers :
```
http://localhost:7860
```

3. Sélectionner les options souhaitées :
   - Choisir un algorithme dans le menu déroulant
   - Sélectionner la méthode d'entrée (webcam ou téléchargement vidéo)
   - Appliquer des variations si désiré
   - Ajuster les paramètres de performance selon les besoins

## Comparaison des Algorithmes

L'application inclut un guide de référence complet comparant les algorithmes selon plusieurs dimensions :
- Support des Variations de Pose
- Gestion des Conditions d'Éclairage
- Support des Occlusions
- Caractéristiques de Performance
- Fonctionnalités Spéciales

Chaque algorithme est évalué sur divers aspects avec des explications détaillées de leurs forces et limites.

## Considérations de Performance

- Le saut d'images peut être ajusté pour optimiser les performances
- L'accélération GPU est disponible pour les algorithmes supportés
- L'utilisation de la mémoire varie significativement entre les algorithmes
- La vitesse de traitement dépend de l'algorithme sélectionné et de la résolution d'entrée

## Remerciements

- L'équipe OpenCV pour l'implémentation de Viola-Jones
- Les développeurs de DLib pour l'implémentation HOG+SVM
- Google pour le framework MediaPipe
- L'équipe InsightFace pour l'implémentation de RetinaFace
- Ultralytics pour YOLOv8
- Tous les autres contributeurs open-source

## Citation

Si vous utilisez ce projet dans votre recherche, veuillez citer :
```
@software{face_detection_testing,
  author = {Ben Salem, Houssem and Sauvé, Catherine},
  title = {Suite de Tests pour Algorithmes de Détection de Visages},
  year = {2025},
  url = {https://github.com/yourusername/face-detection-testing}
}
```