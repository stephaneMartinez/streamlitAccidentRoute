import streamlit as st
import streamlit_antd_components as sac
from Librairies.utils import *
from Librairies.machineLearning import *
import Librairies.texte as lb
import Librairies.constant as constant
import Librairies.exploration as exploration
import Librairies.texte as lb
import pandas as pd
import pydeck as pdk

from datetime import date
from sklearn.metrics import classification_report                # Performance accuracy, f1-score
from sklearn.metrics import roc_auc_score                        # performance en utilisant la courbe ROC et le score AUC
from sklearn.metrics import log_loss                             # Perte logarithmique
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import make_scorer, f1_score, recall_score

from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt

TAB_METHODOLOGIE=" Méthodologie"
TAB_RESULTATS=" Synthèse des résultats"
TAB_SIMULATION=" Simulation"
EXPENDEUR_RESULTAT_INTERP_SHAP_BEESWARM = "Graphe de Beeswarm"
EXPENDEUR_RESULTAT_INTERP_SHAP_DEPENDANCE = "Graphe de dépendance"
EXPENDEUR_RESULTAT_INTERP_LIME = "Graphe de Lime"


EXPENDEUR_ESTIMATION_MODELE = "Performances du modèle"

model_choices = {
    "Regression logistique (*) - classification multi classes - rééquilibrage Random Under Sampler": {
        'MODEL_FILE' : constant.CHEMIN_MODELE + "streamlit_LR_under_multi.pkl",
        'TRAIN_RESULT' : constant.CHEMIN_IMAGE + "Resultats_LR_under_multi.png",
        'TEST_RESULT' : constant.CHEMIN_IMAGE + "Resultats_LR_under_multi_test.png",
        'BINARY_MODEL' : False
    },
    
}

def main():
    # ------
    # HEADER
    # ------
    st.subheader('Interprétabilité avec Shap et Lime', anchor=False)
    #st.markdown('''<p style="text-align:justify;">

    # ---------------------
    # CHARGEMENT DES DONNES
    # ---------------------
    #data = exploration.load_data()

    # ----------------------
    # MENU ONGLETS PRINCIPAL
    # ----------------------
    tabs = sac.tabs([
        sac.TabsItem(TAB_METHODOLOGIE),
        sac.TabsItem(TAB_RESULTATS), 
        sac.TabsItem(TAB_SIMULATION)
        ], size='sm')
    
    # TRANSFORMER
    if tabs == TAB_METHODOLOGIE :
        # OBJECTIF ET METHODOLOGIE
        st.markdown(lb.INTERP_INTRODUCTION, unsafe_allow_html=True)    
        show_space(1)
        st.markdown(lb.ML_RESULTATS_INTERPRETABILITE, unsafe_allow_html=True)

    elif tabs== TAB_RESULTATS:
        with st.expander(EXPENDEUR_RESULTAT_INTERP_SHAP_BEESWARM, False) :
            # BEESWARM - REGRESSION LOGISTIQUE
            col1, col2 = st.columns([0.2, 0.8])
            with col1 :
                show_image(constant.CHEMIN_IMAGE + "Interp_RL_titre.png")

            with col2 :
                st.markdown (lb.INTERP_TOP10, unsafe_allow_html=True) 
                show_image(constant.CHEMIN_IMAGE + "Interp_Beeswarm_RL_Top10.png")

                st.markdown (lb.INTERP_FLOP10, unsafe_allow_html=True) 
                show_image(constant.CHEMIN_IMAGE + "Interp_Beeswarm_RL_Flop10.png")
            
            # BEESWARM - XGBOOST
            col1, col2 = st.columns([0.2, 0.8])
            with col1 :
                show_image(constant.CHEMIN_IMAGE + "Interp_XGBoost_titre.png")

            with col2 :
                st.markdown (lb.INTERP_TOP10, unsafe_allow_html=True) 
                show_image(constant.CHEMIN_IMAGE + "Interp_Beeswarm_XGBoost_Top10.png")

                st.markdown (lb.INTERP_FLOP10, unsafe_allow_html=True) 
                show_image(constant.CHEMIN_IMAGE + "Interp_Beeswarm_XGBoost_Flop10.png")

            # BEESWARM - DNN
            col1, col2 = st.columns([0.2, 0.8])
            with col1 :
                show_image(constant.CHEMIN_IMAGE + "Interp_DNN_titre.png")

            with col2 :
                st.markdown (lb.INTERP_TOP10, unsafe_allow_html=True) 
                show_image(constant.CHEMIN_IMAGE + "Interp_Beeswarm_DNN_Top10.png")

                st.markdown (lb.INTERP_FLOP10, unsafe_allow_html=True) 
                show_image(constant.CHEMIN_IMAGE + "Interp_Beeswarm_DNN_Flop10.png")

        with st.expander(EXPENDEUR_RESULTAT_INTERP_SHAP_DEPENDANCE):
            # DEPENDANCE - REGRESSION LOGISTIQUE
            col1, col2 = st.columns([0.2, 0.8])
            with col1 :
                show_image(constant.CHEMIN_IMAGE + "Interp_RL_titre.png")

            with col2 :
                show_image(constant.CHEMIN_IMAGE + "Interp_GrapheDependance_RL_user_secu1.png")

            # DEPENDANCE - XGBOOST
            col1, col2 = st.columns([0.2, 0.8])
            with col1 :
                show_image(constant.CHEMIN_IMAGE + "Interp_XGBoost_titre.png")

            with col2 :
                show_image(constant.CHEMIN_IMAGE + "Interp_GrapheDependance_XGBoost_user_secu1.png")

            # DEPENDANCE - DNN
            col1, col2 = st.columns([0.2, 0.8])
            with col1 :
                show_image(constant.CHEMIN_IMAGE + "Interp_DNN_titre.png")

            with col2 :
                show_image(constant.CHEMIN_IMAGE + "Interp_GrapheDependance_DNN_user_secu1.png")
 
        with st.expander(EXPENDEUR_RESULTAT_INTERP_LIME):
            # LIME - REGRESSION LOGISTIQUE
            col1, col2 = st.columns([0.2, 0.8])
            with col1 :
                show_image(constant.CHEMIN_IMAGE + "Interp_RL_titre.png")

            with col2 :
                show_image(constant.CHEMIN_IMAGE + "InterpLime_RL.png")
            
            # LIME - XGBOOST
            col1, col2 = st.columns([0.2, 0.8])
            with col1 :
                show_image(constant.CHEMIN_IMAGE + "Interp_XGBoost_titre.png")

            with col2 :
                show_image(constant.CHEMIN_IMAGE + "InterpLime_XGBoost.png")

            # LIME - DNN
            col1, col2 = st.columns([0.2, 0.8])
            with col1 :
                show_image(constant.CHEMIN_IMAGE + "Interp_DNN_titre.png")

            with col2 :
                st.markdown (lb.INTERP_TOP10, unsafe_allow_html=True) 
                show_image(constant.CHEMIN_IMAGE + "InterpLime_DNN.png")

            st.markdown (lb.INTERP_FORCE, unsafe_allow_html=True)
            show_image(constant.CHEMIN_IMAGE + "InterpDiagrammeForce_DNN.png")

    elif tabs==TAB_SIMULATION :
        show_error_404()    
            
