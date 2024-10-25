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
EXPENDEUR_RESULTAT_ML_UNDER_SAMPLER_MULTI = "Résultats rééquilibrage Under Sampling - multiclasses"
EXPENDEUR_RESULTAT_ML_SMOTE_MULTI = "Résultats rééquilibrage SMOTE - multiclasses"
EXPENDEUR_RESULTAT_ML_UNDER_SAMPLER_BINARY = "Résultats rééquilibrage Under Sampling - Classification binaire"
EXPENDEUR_RESULTAT_ML_SMOTE_BINARY = "Résultats rééquilibrage SMOTE - Classification binaire"

EXPENDEUR_ESTIMATION_MODELE = "Performances du modèle"

model_choices = {
    "Regression logistique (*) - classification multi classes - rééquilibrage Random Under Sampler": {
        'MODEL_FILE' : constant.CHEMIN_MODELE + "streamlit_LR_under_multi.pkl",
        'TRAIN_RESULT' : constant.CHEMIN_IMAGE + "Resultats_LR_under_multi.png",
        'TEST_RESULT' : constant.CHEMIN_IMAGE + "Resultats_LR_under_multi_test.png",
        'BINARY_MODEL' : False
    },
    "SVC - classification multi classes - rééquilibrage Random Under Sampler" : {
        'MODEL_FILE' : None,
        'TRAIN_RESULT' :  None,
        'TEST_RESULT' : constant.CHEMIN_IMAGE + "Resultats_SVC_under_multi_test.png",
        'BINARY_MODEL' : False
    }, 
    "RAMDOM FOREST - classification multi classes - rééquilibrage Random Under Sampler" : {
        'MODEL_FILE' : None,
        'TRAIN_RESULT' :  constant.CHEMIN_IMAGE + "Resultats_RForest_under_multi.png",
        'TEST_RESULT' : constant.CHEMIN_IMAGE + "Resultats_RForest_under_multi_test.png",
        'BINARY_MODEL' : False
    }, 
    "XGBoost (*) - classification multi classes - rééquilibrage Random Under Sampler" : {
        'MODEL_FILE' : constant.CHEMIN_MODELE + "streamlit_XGBoost_under_multi.pkl",
        'TRAIN_RESULT' :  constant.CHEMIN_IMAGE + "Resultats_XGBoost_under_multi.png",
        'TEST_RESULT' : constant.CHEMIN_IMAGE + "Resultats_XGBoost_under_multi_test.png",
        'BINARY_MODEL' : False
    }, 
    "GradientBoosting - classification multi classes - rééquilibrage Random Under Sampler" : {
        'MODEL_FILE' : None,
        'TRAIN_RESULT' :  constant.CHEMIN_IMAGE + "Resultats_GradientBoosting_under_multi.png",
        'TEST_RESULT' : constant.CHEMIN_IMAGE + "Resultats_GradientBoosting_under_multi_test.png",
        'BINARY_MODEL' : False
    }, 
    "AdaBoost - classification multi classes - rééquilibrage Random Under Sampler" : {
        'MODEL_FILE' : None,
        'TRAIN_RESULT' :  constant.CHEMIN_IMAGE + "Resultats_AdaBoost_under_multi.png",
        'TEST_RESULT' : constant.CHEMIN_IMAGE + "Resultats_AdaBoost_under_multi_test.png",
        'BINARY_MODEL' : False
    }, 
    "AdaBoost - classification multi classes - rééquilibrage SMOTE" : {
        'MODEL_FILE' : None,
        'TRAIN_RESULT' :  constant.CHEMIN_IMAGE + "Resultats_AdaBoost_SMOTE_multi.png",
        'TEST_RESULT' : constant.CHEMIN_IMAGE + "Resultats_AdaBoost_SMOTE_multi_test.png",
        'BINARY_MODEL' : False
    }, 
    "AdaBoost (*) - classification binaire - rééquilibrage Random Under Sampler" : {
        'MODEL_FILE' : constant.CHEMIN_MODELE + "streamlit_AdaBoost_under_binary.pkl",
        'TRAIN_RESULT' :  constant.CHEMIN_IMAGE + "Resultats_AdaBoost_under_binary.png",
        'TEST_RESULT' : constant.CHEMIN_IMAGE + "Resultats_AdaBoost_under_binary_test.png",
        'BINARY_MODEL' : True
    }, 
    "AdaBoost (*) - classification binaire - rééquilibrage SMOTE" : {
        'MODEL_FILE' : constant.CHEMIN_MODELE + "streamlit_AdaBoost_SMOTE_binary.pkl",
        'TRAIN_RESULT' :  constant.CHEMIN_IMAGE + "Resultats_AdaBoost_SMOTE_binary.png",
        'TEST_RESULT' : constant.CHEMIN_IMAGE + "Resultats_AdaBoost_SMOTE_binary_test.png",
        'BINARY_MODEL' : True
    }, 
    "Bagging+Decision Tree (*) - classification multi classes - rééquilibrage Random under Sampling" : {
        'MODEL_FILE' : constant.CHEMIN_MODELE + "streamlit_Bagging_DTree_under_multi.pkl",
        'TRAIN_RESULT' :  constant.CHEMIN_IMAGE + "Resultats_Bagging_DTree_under_multi.png",
        'TEST_RESULT' : constant.CHEMIN_IMAGE + "Resultats_Bagging_DTree_under_multi_test.png",
        'BINARY_MODEL' : False
    }, 
    "Bagging+Decision Tree (*) - classification multi classes - rééquilibrage SMOTE" : {
        'MODEL_FILE' : constant.CHEMIN_MODELE + "streamlit_Bagging_DTree_SMOTE_multi.pkl",
        'TRAIN_RESULT' :  constant.CHEMIN_IMAGE + "Resultats_Bagging_DTree_SMOTE_multi.png",
        'TEST_RESULT' : constant.CHEMIN_IMAGE + "Resultats_Bagging_DTree_SMOTE_multi_test.png",
        'BINARY_MODEL' : False
    }, 
    "Bagging+Decision Tree (*) - classification binaire - rééquilibrage Random under Sampling" : {
        'MODEL_FILE' : constant.CHEMIN_MODELE + "streamlit_Bagging_DTree_under_binary.pkl",
        'TRAIN_RESULT' :  constant.CHEMIN_IMAGE + "Resultats_Bagging_DTree_under_binary.png",
        'TEST_RESULT' : constant.CHEMIN_IMAGE + "Resultats_Bagging_DTree_under_binary_test.png",
        'BINARY_MODEL' : True
    }, 
    "Bagging+Decision Tree (*) - classification binaire - rééquilibrage SMOTE" : {
        'MODEL_FILE' : constant.CHEMIN_MODELE + "streamlit_Bagging_DTree_SMOTE_binary.pkl",
        'TRAIN_RESULT' :  constant.CHEMIN_IMAGE + "Resultats_Bagging_DTree_SMOTE_binary.png",
        'TEST_RESULT' : constant.CHEMIN_IMAGE + "Resultats_Bagging_DTree_SMOTE_binary_test.png",
        'BINARY_MODEL' : True
    }, 

    
}

def main():
    # ------
    # HEADER
    # ------
    st.subheader('Modélisation simple en machine learning', anchor=False)
    st.markdown('''<p style="text-align:justify;">
                Utilisation de la librairie scikit-learn.        
                </p>''', unsafe_allow_html=True)

    # ---------------------
    # CHARGEMENT DES DONNES
    # ---------------------
    data = exploration.load_data()

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
        st.markdown(lb.PREPROCESSING_METHODOLOGIE_TYPE_PROBLEME, unsafe_allow_html=True)    

        sac.alert(label='Premières constatations', description=lb.PREPROCESSING_METHODOLOGIE_PREMIERE_CONSTATATION, color='orange', icon=True, closable=False)

        show_space(1)
        st.markdown(lb.PREPROCESSING_METHODOLOGIE_AJUSTEMENT_PROBLEME, unsafe_allow_html=True)    
    
    elif tabs== TAB_RESULTATS:
        with st.expander(EXPENDEUR_RESULTAT_ML_SMOTE_MULTI, False) :
            show_image(constant.CHEMIN_IMAGE + "ML_Resultats_SMOTE_multiClasses.png")

        with st.expander(EXPENDEUR_RESULTAT_ML_UNDER_SAMPLER_MULTI):
            show_image(constant.CHEMIN_IMAGE + "ML_Resultats_UnderSampling_multiClasses.png")
 
        with st.expander(EXPENDEUR_RESULTAT_ML_SMOTE_BINARY):
            show_image(constant.CHEMIN_IMAGE + "ML_Resultats_SMOTE_binary.png")

        with st.expander(EXPENDEUR_RESULTAT_ML_UNDER_SAMPLER_BINARY):
            show_image(constant.CHEMIN_IMAGE + "ML_Resultats_UnderSampling_binary.png")

        st.markdown(lb.ML_RESULTATS_BINAIRE, unsafe_allow_html=True)

    elif tabs==TAB_SIMULATION :
        selected_model = st.selectbox ("Choisissez un modèle", list(model_choices.keys()))
        train_results = model_choices[selected_model]['TRAIN_RESULT']
        test_results = model_choices[selected_model]['TEST_RESULT']
        binary_model = model_choices[selected_model]['BINARY_MODEL']
        #if selected_model ==:
        with st.expander(EXPENDEUR_ESTIMATION_MODELE, True) :
            show_image (train_results)
            show_image (test_results)
            
        # Charger le modèle sélectionné
        model_name = model_choices[selected_model]['MODEL_FILE']
        if model_name :
            with open(model_name, 'rb') as file:
                model = load_model(file)

            # Interface utilisateur pour entrer les features
            st.header(f'Demande de Prédiction')
            
            # Critères de sélection
            col1, col2 = st.columns(2)
            with col1 :
                #st.markdown(lb.SIMULATEUR_CRITERE_CARACTERISTIQUE, unsafe_allow_html=True)
                carac_agg_option = exploration.get_list_from_dict(constant.FEATURES[constant.CARAC_AGG][constant.FEATURE_MODALITE])
                carac_atm_option = exploration.get_list_from_dict(constant.FEATURES[constant.CARAC_ATM][constant.FEATURE_MODALITE])
                carac_col_option = exploration.get_list_from_dict(constant.FEATURES[constant.CARAC_COL][constant.FEATURE_MODALITE])
                carac_lum_option = exploration.get_list_from_dict(constant.FEATURES[constant.CARAC_LUM][constant.FEATURE_MODALITE])            
                
                carac_agg_default = constant.FEATURES[constant.CARAC_AGG][constant.FEATURE_DEFAULT]
                carac_atm_default = constant.FEATURES[constant.CARAC_ATM][constant.FEATURE_DEFAULT]
                carac_col_default = constant.FEATURES[constant.CARAC_COL][constant.FEATURE_DEFAULT]
                carac_lum_default = constant.FEATURES[constant.CARAC_LUM][constant.FEATURE_DEFAULT]            
                
                carac_agg = st.selectbox(constant.CARAC_AGG, carac_agg_option, index=carac_agg_default)
                carac_atm = st.selectbox(constant.CARAC_ATM, carac_atm_option, index=carac_atm_default)
                carac_col = st.selectbox(constant.CARAC_COL, carac_col_option, index=carac_col_default)
                carac_lum = st.selectbox(constant.CARAC_LUM, carac_lum_option, index=carac_lum_default)
                
                show_space(1)
                sac.divider(align='center', variant='dashed', color='gray')
                show_space(1)
                
                agg_catv_perso_option = exploration.get_list_from_dict(constant.FEATURES[constant.AGG_CATV_PERSO][constant.FEATURE_MODALITE])
                vehi_choc_option = exploration.get_list_from_dict(constant.FEATURES[constant.VEHI_CHOC][constant.FEATURE_MODALITE])
                vehi_manv_option = exploration.get_list_from_dict(constant.FEATURES[constant.VEHI_MANV][constant.FEATURE_MODALITE])
                vehi_motor_option = exploration.get_list_from_dict(constant.FEATURES[constant.VEHI_MOTOR][constant.FEATURE_MODALITE])
                vehi_obs_option = exploration.get_list_from_dict(constant.FEATURES[constant.VEHI_OBS][constant.FEATURE_MODALITE])
                vehi_obsm_option = exploration.get_list_from_dict(constant.FEATURES[constant.VEHI_OBSM][constant.FEATURE_MODALITE])            
                
                agg_catv_perso_default = constant.FEATURES[constant.AGG_CATV_PERSO][constant.FEATURE_DEFAULT]
                vehi_choc_default = constant.FEATURES[constant.VEHI_CHOC][constant.FEATURE_DEFAULT]
                vehi_manv_default = constant.FEATURES[constant.VEHI_MANV][constant.FEATURE_DEFAULT]
                vehi_motor_default = constant.FEATURES[constant.VEHI_MOTOR][constant.FEATURE_DEFAULT]
                vehi_obsm_default = constant.FEATURES[constant.VEHI_OBSM][constant.FEATURE_DEFAULT]
                vehi_obs_default = constant.FEATURES[constant.VEHI_OBS][constant.FEATURE_DEFAULT]
                
                agg_catv_perso = st.selectbox(constant.AGG_CATV_PERSO, agg_catv_perso_option, agg_catv_perso_default)
                vehi_choc = st.selectbox(constant.VEHI_CHOC, vehi_choc_option, vehi_choc_default)
                vehi_manv = st.selectbox(constant.VEHI_MANV, vehi_manv_option, vehi_manv_default)
                vehi_motor = st.selectbox(constant.VEHI_MOTOR, vehi_motor_option, vehi_motor_default)
                vehi_obs = st.selectbox(constant.VEHI_OBS, vehi_obs_option, vehi_obs_default)
                vehi_obsm = st.selectbox(constant.VEHI_OBSM, vehi_obsm_option, vehi_obsm_default)
            
            with col2 :
                lieu_catr_option = exploration.get_list_from_dict(constant.FEATURES[constant.LIEU_CATR][constant.FEATURE_MODALITE])
                lieu_circ_option = exploration.get_list_from_dict(constant.FEATURES[constant.LIEU_CIRC][constant.FEATURE_MODALITE])
                lieu_plan_option = exploration.get_list_from_dict(constant.FEATURES[constant.LIEU_PLAN][constant.FEATURE_MODALITE])
                lieu_situ_option = exploration.get_list_from_dict(constant.FEATURES[constant.LIEU_SITU][constant.FEATURE_MODALITE])            
            
                lieu_catr_default = constant.FEATURES[constant.LIEU_CATR][constant.FEATURE_DEFAULT]
                lieu_circ_default = constant.FEATURES[constant.LIEU_CIRC][constant.FEATURE_DEFAULT]
                lieu_plan_default = constant.FEATURES[constant.LIEU_PLAN][constant.FEATURE_DEFAULT]
                lieu_situ_default = constant.FEATURES[constant.LIEU_SITU][constant.FEATURE_DEFAULT]            
            
                lieu_catr = st.selectbox(constant.LIEU_CATR, lieu_catr_option, index=lieu_catr_default)
                lieu_circ = st.selectbox(constant.LIEU_CIRC, lieu_circ_option, index=lieu_circ_default)
                lieu_plan = st.selectbox(constant.LIEU_PLAN, lieu_plan_option, index=lieu_plan_default)
                lieu_situ = st.selectbox(constant.LIEU_SITU, lieu_situ_option, index=lieu_situ_default)
            
                show_space(1)
                sac.divider(align='center', variant='dashed', color='gray', key="divide2")
                show_space(1)            
            
                user_catu_option = exploration.get_list_from_dict(constant.FEATURES[constant.USER_CATU][constant.FEATURE_MODALITE])
                user_secu1_option = exploration.get_list_from_dict(constant.FEATURES[constant.USER_SECU1][constant.FEATURE_MODALITE])
                user_secu2_option = exploration.get_list_from_dict(constant.FEATURES[constant.USER_SECU2][constant.FEATURE_MODALITE])
                user_secu3_option = exploration.get_list_from_dict(constant.FEATURES[constant.USER_SECU3][constant.FEATURE_MODALITE])
                user_trajet_option = exploration.get_list_from_dict(constant.FEATURES[constant.USER_TRAJET][constant.FEATURE_MODALITE])
                zone_geo_option = exploration.get_list_from_dict(constant.FEATURE_ZONE_GEO[constant.FEATURE_MODALITE], keys=True)
            
                user_catu_default = constant.FEATURES[constant.USER_CATU][constant.FEATURE_DEFAULT]
                user_secu1_default = constant.FEATURES[constant.USER_SECU1][constant.FEATURE_DEFAULT]
                user_secu2_default = constant.FEATURES[constant.USER_SECU2][constant.FEATURE_DEFAULT]
                user_secu3_default = constant.FEATURES[constant.USER_SECU3][constant.FEATURE_DEFAULT]
                user_trajet_default = constant.FEATURES[constant.USER_TRAJET][constant.FEATURE_DEFAULT]
            
                user_catu = st.selectbox(constant.USER_CATU, user_catu_option, index=user_catu_default)
                user_secu1 = st.selectbox(constant.USER_SECU1, user_secu1_option, index=user_secu1_default)
                user_secu2 = st.selectbox(constant.USER_SECU2, user_secu2_option, index=user_secu2_default)
                user_secu3 = st.selectbox(constant.USER_SECU3, user_secu3_option, index=user_secu3_default)
                user_trajet = st.selectbox(constant.USER_TRAJET, user_trajet_option, index=user_trajet_default)
                zone_geo = st.selectbox(constant.ZONE_GEO, zone_geo_option, index=7)
            
            affiche_data_test = sac.switch(label="Afficher les données", align='start', size='sm')        
            
            # LANCER UNE PREDICTION
            # ---------------------
            if st.button('Faire une prédiction'):
                # Générer le jeu de données
                data = init_df(
                    carac_an = date.today().year,
                    carac_mois = date.today().month,
                    carac_jour = date.today().day,
                    carac_agg = next((cle for cle, valeur in constant.FEATURES[constant.CARAC_AGG][constant.FEATURE_MODALITE].items() if valeur == carac_agg), -2),
                    carac_atm = next((cle for cle, valeur in constant.FEATURES[constant.CARAC_ATM][constant.FEATURE_MODALITE].items() if valeur == carac_atm), -2),
                    carac_col = next((cle for cle, valeur in constant.FEATURES[constant.CARAC_COL][constant.FEATURE_MODALITE].items() if valeur == carac_col), -2),
                    carac_lum = next((cle for cle, valeur in constant.FEATURES[constant.CARAC_LUM][constant.FEATURE_MODALITE].items() if valeur == carac_lum), -2),
                    carac_lat = constant.FEATURE_ZONE_GEO[constant.FEATURE_MODALITE][zone_geo]['Latitude'],
                    carac_long = constant.FEATURE_ZONE_GEO[constant.FEATURE_MODALITE][zone_geo]['Longitude'],
                    lieu_catr = next((cle for cle, valeur in constant.FEATURES[constant.LIEU_CATR][constant.FEATURE_MODALITE].items() if valeur == lieu_catr), -2),
                    lieu_circ = next((cle for cle, valeur in constant.FEATURES[constant.LIEU_CIRC][constant.FEATURE_MODALITE].items() if valeur == lieu_circ), -2),
                    lieu_plan = next((cle for cle, valeur in constant.FEATURES[constant.LIEU_PLAN][constant.FEATURE_MODALITE].items() if valeur == lieu_plan), -2),
                    lieu_situ = next((cle for cle, valeur in constant.FEATURES[constant.LIEU_SITU][constant.FEATURE_MODALITE].items() if valeur == lieu_situ), -2),
                    agg_catv_perso = next((cle for cle, valeur in constant.FEATURES[constant.AGG_CATV_PERSO][constant.FEATURE_MODALITE].items() if valeur == agg_catv_perso), -2),
                    vehi_choc = next((cle for cle, valeur in constant.FEATURES[constant.VEHI_CHOC][constant.FEATURE_MODALITE].items() if valeur == vehi_choc), -2),
                    vehi_manv = next((cle for cle, valeur in constant.FEATURES[constant.VEHI_MANV][constant.FEATURE_MODALITE].items() if valeur == vehi_manv), -2),
                    vehi_motor = next((cle for cle, valeur in constant.FEATURES[constant.VEHI_MOTOR][constant.FEATURE_MODALITE].items() if valeur == vehi_motor), -2),
                    vehi_obsm = next((cle for cle, valeur in constant.FEATURES[constant.VEHI_OBSM][constant.FEATURE_MODALITE].items() if valeur == vehi_obsm), -2),
                    vehi_obs =  next((cle for cle, valeur in constant.FEATURES[constant.VEHI_OBS][constant.FEATURE_MODALITE].items() if valeur == vehi_obs), -2),
                    user_catu = next((cle for cle, valeur in constant.FEATURES[constant.USER_CATU][constant.FEATURE_MODALITE].items() if valeur == user_catu), -2),
                    user_secu1 = next((cle for cle, valeur in constant.FEATURES[constant.USER_SECU1][constant.FEATURE_MODALITE].items() if valeur == user_secu1), -2),
                    user_secu2 = next((cle for cle, valeur in constant.FEATURES[constant.USER_SECU2][constant.FEATURE_MODALITE].items() if valeur == user_secu2), -2),
                    user_secu3 = next((cle for cle, valeur in constant.FEATURES[constant.USER_SECU3][constant.FEATURE_MODALITE].items() if valeur == user_secu3), -2),
                    user_trajet = next((cle for cle, valeur in constant.FEATURES[constant.USER_TRAJET][constant.FEATURE_MODALITE].items() if valeur == user_trajet), -2),
                )
            
                if affiche_data_test:
                    st.write(data[constant.LISTE_COLUMNS])
            
                prediction = model.predict(data)
                prediction_proba = model.predict_proba(data)
            
                if binary_model == True :
                    liste_gravite = constant.LISTE_GRAVITE_BINAIRE

                else:
                    liste_gravite = constant.LISTE_GRAVITE

                st.success(f"La prédiction avec {selected_model} est : {prediction[0]} = '{liste_gravite[prediction[0]]}'")       # {constant.FEATURES[constant.USER_GRAVITE][constant.FEATURE_MODALITE][prediction[0]]}
                #st.success(f"La prédiction avec {selected_model} est : {prediction_proba}")
                probs = prediction_proba[0]
                
                # Création du graphique à barres
                fig, ax = plt.subplots(figsize=(10, 3))
                bars = ax.bar(range(len(probs)), probs, alpha=0.7)
                for bar, prob in zip(bars, probs):
                    yval = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, 
                            yval + 0.01, 
                            f'{prob * 100:.2f}%', 
                            ha='center', 
                            va='bottom' 
                    )
                ax.set_xticks(range(len(probs)))
                ax.set_xticklabels(liste_gravite, rotation=45)
                ax.set_title(f'Probabilités prédites')
                ax.set_ylabel('Probabilité')
                ax.set_xlabel("Gravité de l'accident")
                ax.set_ylim(0, 1)
                
                st.pyplot(fig.figure)
                
            
        else :
            # Aucun modèle disponible pour lancer une simulation
            sac.alert(label='Simulation', description = "Modèle non disponible pour lancer une simulation.",  size='sm', variant='light', )