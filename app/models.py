from pydantic import BaseModel, Field
from typing import Optional


class CustomerFeatures(BaseModel):
    """Schéma des features pour un client"""
    CreditScore: int = Field(..., ge=300, le=850, description="Score de crédit (300-850)")
    Age: int = Field(..., ge=18, le=100, description="Âge du client")
    Tenure: int = Field(..., ge=0, le=10, description="Ancienneté en années")
    Balance: float = Field(..., ge=0, description="Solde du compte")
    NumOfProducts: int = Field(..., ge=1, le=4, description="Nombre de produits")
    HasCrCard: int = Field(..., ge=0, le=1, description="Possède une carte de crédit (0/1)")
    IsActiveMember: int = Field(..., ge=0, le=1, description="Membre actif (0/1)")
    EstimatedSalary: float = Field(..., ge=0, description="Salaire estimé")
    Geography_Germany: int = Field(..., ge=0, le=1, description="Client allemand (0/1)")
    Geography_Spain: int = Field(..., ge=0, le=1, description="Client espagnol (0/1)")

    class Config:
        json_schema_extra = {
            "example": {
                "CreditScore": 650,
                "Age": 35,
                "Tenure": 5,
                "Balance": 50000.0,
                "NumOfProducts": 2,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 75000.0,
                "Geography_Germany": 0,
                "Geography_Spain": 1
            }
        }


class PredictionResponse(BaseModel):
    """Schéma de réponse pour une prédiction"""
    churn_probability: float = Field(..., description="Probabilité de churn (0-1)")
    prediction: int = Field(..., description="Prédiction binaire (0: reste, 1: part)")
    risk_level: str = Field(..., description="Niveau de risque (Low/Medium/High)")


class HealthResponse(BaseModel):
    """Schéma de réponse pour le health check"""
    status: str = Field(..., description="État de l'API")
    model_loaded: bool = Field(..., description="Modèle chargé ou non")


class BatchPredictionResponse(BaseModel):
    """Schéma de réponse pour les prédictions batch"""
    predictions: list = Field(..., description="Liste des prédictions")
    count: int = Field(..., description="Nombre de prédictions")
