import streamlit as st
import requests
import pandas as pd
import json

# Configuration de la page
st.set_page_config(
    page_title="Bank Churn API Tester",
    page_icon="🏦",
    layout="wide"
)

# Sidebar pour la configuration de l'URL
st.sidebar.header("⚙️ Configuration")
api_mode = st.sidebar.radio(
    "Mode de l'API",
    ["Local (Docker)", "Azure (Production)", "Custom URL"]
)

if api_mode == "Local (Docker)":
    API_BASE_URL = "http://localhost:8000"
elif api_mode == "Azure (Production)":
    API_BASE_URL = st.sidebar.text_input(
        "URL Azure Container Apps",
        placeholder="https://bank-churn.xxxxx.azurecontainerapps.io"
    )
else:
    API_BASE_URL = st.sidebar.text_input(
        "URL personnalisée",
        value="http://localhost:8000"
    )

PREDICT_URL = f"{API_BASE_URL}/predict"
BATCH_URL = f"{API_BASE_URL}/predict/batch"
HEALTH_URL = f"{API_BASE_URL}/health"
DRIFT_URL = f"{API_BASE_URL}/drift/check"

# Titre de l'application
st.title("🏦 Bank Churn Prediction API Tester")
st.markdown("Testez les prédictions de défection client via votre API FastAPI.")

# Afficher l'URL actuelle
st.info(f"🔗 API URL: **{API_BASE_URL}**")

# Section 1 : Vérification de l'état de l'API
st.header("📡 1. Vérification de l'API")

col_health1, col_health2 = st.columns([1, 3])
with col_health1:
    if st.button("🩺 Vérifier la santé", type="primary"):
        try:
            response = requests.get(HEALTH_URL, timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                st.success(f"✅ API en ligne - Modèle chargé : {health_data.get('model_loaded', 'N/A')}")
            else:
                st.error(f"❌ API retourne une erreur : {response.status_code}")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Impossible de joindre l'API : {e}")

# Section 2 : Prédiction individuelle
st.header("👤 2. Prédiction pour un client unique")
st.markdown("Remplissez les caractéristiques d'un client pour obtenir une prédiction.")

# Création de deux colonnes pour l'organisation
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Informations démographiques")
    credit_score = st.slider("Credit Score", 300, 850, 650)
    age = st.slider("Âge", 18, 100, 35)
    tenure = st.slider("Ancienneté (années)", 0, 10, 5)
    
    st.subheader("🌍 Informations géographiques")
    geography = st.selectbox("Pays", ["France", "Allemagne", "Espagne"])
    geography_germany = 1 if geography == "Allemagne" else 0
    geography_spain = 1 if geography == "Espagne" else 0

with col2:
    st.subheader("💰 Informations financières")
    balance = st.number_input("Solde du compte (€)", min_value=0.0, value=50000.0, step=1000.0)
    num_products = st.slider("Nombre de produits", 1, 4, 2)
    estimated_salary = st.number_input("Salaire estimé (€)", min_value=0.0, value=75000.0, step=1000.0)
    
    st.subheader("📋 Statut client")
    has_cr_card = st.checkbox("Possède une carte de crédit", value=True)
    is_active_member = st.checkbox("Membre actif", value=True)

# Préparation des données pour l'API
customer_data = {
    "CreditScore": credit_score,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_products,
    "HasCrCard": 1 if has_cr_card else 0,
    "IsActiveMember": 1 if is_active_member else 0,
    "EstimatedSalary": estimated_salary,
    "Geography_Germany": geography_germany,
    "Geography_Spain": geography_spain
}

# Affichage des données JSON
with st.expander("📄 Voir les données envoyées à l'API (format JSON)"):
    st.json(customer_data)

# Bouton de prédiction individuelle
if st.button("🔍 Prédire le risque de churn", type="primary"):
    with st.spinner("Envoi de la requête à l'API..."):
        try:
            response = requests.post(PREDICT_URL, json=customer_data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                
                # Affichage des résultats
                st.success("✅ Prédiction obtenue avec succès !")
                
                # Métriques
                col_metric1, col_metric2, col_metric3 = st.columns(3)
                
                with col_metric1:
                    st.metric(
                        label="Probabilité de Churn",
                        value=f"{result['churn_probability']*100:.1f}%"
                    )
                
                with col_metric2:
                    prediction_label = "🚨 VA PARTIR" if result['prediction'] == 1 else "✅ RESTE"
                    st.metric(
                        label="Prédiction",
                        value=prediction_label
                    )
                
                with col_metric3:
                    risk_colors = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
                    risk_emoji = risk_colors.get(result['risk_level'], "⚪")
                    st.metric(
                        label="Niveau de Risque",
                        value=f"{risk_emoji} {result['risk_level']}"
                    )
                
                # Afficher la réponse JSON complète
                with st.expander("📋 Réponse JSON complète"):
                    st.json(result)
                    
            elif response.status_code == 422:
                st.error(f"❌ Données invalides : {response.json()}")
            elif response.status_code == 503:
                st.error("❌ Modèle non disponible sur le serveur")
            else:
                st.error(f"❌ Erreur API : {response.status_code} - {response.text}")
                
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Impossible de joindre l'API : {e}")

# Section 3 : Prédiction par lot (Batch)
st.header("👥 3. Prédiction par lot (Batch)")
st.markdown("Téléchargez un fichier CSV pour faire des prédictions sur plusieurs clients.")

uploaded_file = st.file_uploader("📁 Choisir un fichier CSV", type=['csv'])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write(f"📊 **{len(df)} clients** chargés")
        st.dataframe(df.head(10))
        
        required_columns = [
            'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
            'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
            'Geography_Germany', 'Geography_Spain'
        ]
        
        missing = [col for col in required_columns if col not in df.columns]
        
        if missing:
            st.warning(f"⚠️ Colonnes manquantes : {missing}")
        else:
            if st.button("🚀 Lancer les prédictions batch", type="primary"):
                with st.spinner("Envoi des données à l'API..."):
                    try:
                        batch_data = df[required_columns].to_dict(orient='records')
                        response = requests.post(BATCH_URL, json=batch_data, timeout=120)
                        
                        if response.status_code == 200:
                            result = response.json()
                            predictions = result['predictions']
                            
                            # Ajouter les prédictions au DataFrame
                            df['Churn_Probability'] = [p['churn_probability'] for p in predictions]
                            df['Prediction'] = [p['prediction'] for p in predictions]
                            df['Risk_Label'] = df['Prediction'].map({0: '✅ Reste', 1: '🚨 Part'})
                            
                            st.success(f"✅ {result['count']} prédictions effectuées !")
                            
                            # Statistiques
                            col_stat1, col_stat2, col_stat3 = st.columns(3)
                            with col_stat1:
                                churn_rate = df['Prediction'].mean() * 100
                                st.metric("Taux de Churn prédit", f"{churn_rate:.1f}%")
                            with col_stat2:
                                st.metric("Clients à risque", f"{df['Prediction'].sum()}")
                            with col_stat3:
                                st.metric("Clients fidèles", f"{len(df) - df['Prediction'].sum()}")
                            
                            # Afficher les résultats
                            st.dataframe(df)
                            
                            # Télécharger les résultats
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="📥 Télécharger les résultats CSV",
                                data=csv,
                                file_name="predictions_churn.csv",
                                mime="text/csv"
                            )
                        else:
                            st.error(f"❌ Erreur API : {response.status_code}")
                            
                    except Exception as e:
                        st.error(f"❌ Erreur : {e}")
                        
    except Exception as e:
        st.error(f"❌ Erreur de lecture du fichier : {e}")

# Section 4 : Détection du Drift
st.header("📈 4. Détection du Data Drift")
st.markdown("Vérifiez si les données de production ont dévié par rapport aux données d'entraînement.")

col_drift1, col_drift2 = st.columns([1, 3])
with col_drift1:
    threshold = st.number_input("Seuil p-value", min_value=0.01, max_value=0.10, value=0.05, step=0.01)

if st.button("🔬 Vérifier le Drift", type="secondary"):
    with st.spinner("Analyse du drift en cours..."):
        try:
            response = requests.post(f"{DRIFT_URL}?threshold={threshold}", timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                
                features_analyzed = result.get('features_analyzed', 0)
                features_drifted = result.get('features_drifted', 0)
                drift_pct = (features_drifted / features_analyzed * 100) if features_analyzed > 0 else 0
                
                col_d1, col_d2, col_d3 = st.columns(3)
                with col_d1:
                    st.metric("Features analysées", features_analyzed)
                with col_d2:
                    st.metric("Features avec drift", features_drifted)
                with col_d3:
                    risk = "🟢 LOW" if drift_pct < 20 else "🟡 MEDIUM" if drift_pct < 50 else "🔴 HIGH"
                    st.metric("Niveau de risque", risk)
                
                if 'results' in result:
                    with st.expander("📊 Détails par feature"):
                        for feature, data in result['results'].items():
                            drift_status = "🔴 DRIFT" if data['drift_detected'] else "🟢 OK"
                            st.write(f"**{feature}**: {drift_status} (p-value: {data['p_value']:.4f})")
                
            elif response.status_code == 404:
                st.warning("⚠️ Fichiers de données non trouvés. Générez d'abord les données de production.")
            else:
                st.error(f"❌ Erreur API : {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Impossible de joindre l'API : {e}")

# Section 5 : Documentation
st.header("📚 5. Documentation API")
st.markdown(f"""
### Endpoints disponibles

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/` | GET | Informations sur l'API |
| `/health` | GET | Vérification de santé |
| `/docs` | GET | Documentation Swagger |
| `/predict` | POST | Prédiction individuelle |
| `/predict/batch` | POST | Prédictions par lot |
| `/drift/check` | POST | Vérification du drift |

### Liens rapides
- 📖 [Documentation Swagger]({API_BASE_URL}/docs)
- 📘 [Documentation ReDoc]({API_BASE_URL}/redoc)
""")

# Footer
st.markdown("---")
st.markdown("🏦 **Bank Churn MLOps** | Workshop Azure ML | © 2026")
