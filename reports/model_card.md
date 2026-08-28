# Model Card - Detecteur de Masque Facial

## 1. Informations generales

| Champ | Valeur |
|-------|--------|
| Nom du modele | FaceMaskDetector |
| Version | 1.0 |
| Date | 2026 |
| Type | Classification binaire (Avec masque / Sans masque) |
| Framework | TensorFlow / Keras |

## 2. Finalite du systeme

Detecter automatiquement le port du masque facial dans les espaces publics pour assister la surveillance sanitaire (transports, commerces, etablissements scolaires).

## 3. Utilisateurs cibles

- Personnel de securite
- Gestionnaires d'etablissements
- Autorites sanitaires

## 4. Personnes affectees

- Usagers des espaces publics
- Personnel de surveillance
- Personnes ne pouvant pas porter de masque (raisons medicales)

## 5. Donnees

| Caracteristique | Description |
|-----------------|-------------|
| Source | Dataset d'images de visages |
| Nombre total | 304 images |
| Avec masque | 104 images (34.2%) |
| Sans masque | 200 images (65.8%) |
| Taille des images | 128 x 128 pixels |
| Pre-traitement | Normalisation /255,0 |

## 6. Modeles compares

| Modele | Architecture | Parametres | Accuracy | ROC-AUC |
|--------|--------------|------------|----------|---------|
| ANN | 3 couches denses | 25,297,921 | 88.52% | 0.975 |
| CNN | 3 blocs Conv + Dense | 8,710,817 | 80.33% | 0.929 |

**Modele retenu : ANN** (meilleure performance sur ce dataset)

## 7. Performance detaillee (ANN)

| Metrique | Avec masque | Sans masque |
|----------|-------------|-------------|
| Precision | 0.89 | 0.88 |
| Recall | 0.76 | 0.95 |
| F1-score | 0.82 | 0.92 |

## 8. Limitations

- Performance limitee sur visages de profil
- Non teste sur enfants ou personnes avec accessoires
- Dataset de taille modeste (304 images)
- Biais potentiel sur certaines populations
- Ne detecte pas le port correct du masque (nez decouvert)

## 9. Cas d'erreur et consequences

| Type d'erreur | Consequence |
|---------------|-------------|
| Faux positif (sans masque detecte alors que masque porte) | Sanction injustifiee |
| Faux negatif (masque detecte alors que sans masque) | Risque sanitaire |

## 10. Recommandations

- Utiliser avec supervision humaine
- Ne pas utiliser comme seule preuve pour des sanctions
- Prevoir un mecanisme de recours
- Anonymiser les images (RGPD)

## 11. Explicabilite

- Le modele fournit une probabilite de detection
- Seuil a 0.5 pour la decision binaire
- Interpretation possible via les probabilites de sortie

## 12. Consentement et RGPD

- Les images doivent etre anonymisees
- Le systeme ne doit pas stocker d'images identifiables
- Information et consentement des personnes filmees
- Droit a l'oubli et mecanisme de recours