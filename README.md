# Bank Churn MLOps - Workshop

Ce projet implémente une API de prédiction de churn client (défection) avec les bonnes pratiques MLOps, déployée sur Microsoft Azure.

## 📁 Structure du Projet

```
bank-churn-mlops/
├── app/
│   ├── __init__.py
│   ├── main.py          # API FastAPI
│   ├── models.py         # Schémas Pydantic
│   └── drift_detect.py   # Détection de data drift
├── data/
│   └── bank_churn.csv    # Dataset d'entraînement
├── model/
│   └── churn_model.pkl   # Modèle entraîné
├── tests/
│   └── test_api.py       # Tests unitaires
├── .github/
│   └── workflows/
│       └── ci-cd.yml     # Pipeline CI/CD
├── Dockerfile
├── requirements.txt
├── generate_data.py      # Génération du dataset
├── train_model.py        # Entraînement du modèle
├── drift_data_gen.py     # Génération de données avec drift
└── deploy_azure.sh       # Script de déploiement Azure
```

## 🚀 Démarrage Rapide

### 1. Créer l'environnement virtuel

```bash
cd bank-churn-mlops
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Générer le dataset

```bash
python generate_data.py
```

### 4. Entraîner le modèle

```bash
python train_model.py
```

### 5. Lancer l'API en local

```bash
uvicorn app.main:app --reload --port 8000
```

### 6. Tester l'API

- **Documentation Swagger**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Prédiction**: 

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "CreditScore": 650,
    "Age": 35,
    "Tenure": 5,
    "Balance": 50000,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 75000,
    "Geography_Germany": 0,
    "Geography_Spain": 1
  }'
```

## 🐳 Docker

### Build de l'image

```bash
docker build -t bank-churn-api:v1 .
```

### Lancer le conteneur

```bash
docker run -d -p 8000:8000 --name churn-api bank-churn-api:v1
```

## ☁️ Déploiement Azure

### Prérequis

- Azure CLI installé (`az --version`)
- Docker Desktop en cours d'exécution
- Compte Azure avec abonnement actif

### Déploiement

```bash
chmod +x deploy_azure.sh
./deploy_azure.sh
```

## 🧪 Tests

```bash
pytest tests/ -v --cov=app
```

## 📊 Monitoring & Drift Detection

### Générer des données de production avec drift

```bash
python drift_data_gen.py
```

### Vérifier le drift via l'API

```bash
curl -X POST "http://localhost:8000/drift/check"
```

## 🔄 CI/CD

Le pipeline GitHub Actions:
1. Exécute les tests
2. Build l'image Docker
3. Push vers Azure Container Registry
4. Déploie sur Azure Container Apps

### Configuration des Secrets GitHub

| Secret | Description |
|--------|-------------|
| `AZURE_CREDENTIALS` | JSON avec clientId, clientSecret, subscriptionId, tenantId |
| `ACR_USERNAME` | Nom d'utilisateur ACR |
| `ACR_PASSWORD` | Mot de passe ACR |

## 🧹 Nettoyage

Pour supprimer toutes les ressources Azure:

```bash
az group delete --name rg-mlops-bank-churn --yes --no-wait
```

## 📝 Endpoints API

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/` | GET | Informations sur l'API |
| `/health` | GET | Health check |
| `/docs` | GET | Documentation Swagger |
| `/predict` | POST | Prédiction pour un client |
| `/predict/batch` | POST | Prédictions pour plusieurs clients |
| `/drift/check` | POST | Vérification du data drift |
| `/drift/alert` | POST | Alerte manuelle de drift |
