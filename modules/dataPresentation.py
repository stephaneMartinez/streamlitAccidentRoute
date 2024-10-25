
import streamlit as st
import streamlit_antd_components as sac
from Librairies.utils import *
import Librairies.texte as lb
import Librairies.constant as constant
import pandas as pd

image_volume_df = "donnees/images/Nb_Accidents_2005_2022.png"
image_MCD = "donnees/images/MCD_AccidentsRoutiers.png"

# Constantes
TAB_REGLES = 'Règles de gestion'
TAB_FORMAT = "Format des données"
TAB_MERGE = "Consolidation des données"

#@st.cache_data(persist=False)
def load_data () :
    data = pd.read_csv("donnees/agg_usagers_sample.csv")

    return data

def main():
    data = load_data()
    # ------
    # HEADER
    # ------
    st.subheader('Présentation des données', anchor=False)
    st.markdown(lb.PRESENTATION_HEADER, unsafe_allow_html=True)

    # ----------------------
    # MENU ONGLETS PRINCIPAL
    # ----------------------
    tabs = sac.tabs([sac.TabsItem(TAB_REGLES), sac.TabsItem(TAB_FORMAT), sac.TabsItem(TAB_MERGE)], size='sm')
    if tabs == TAB_REGLES:
        # ONGLET 1 - REGLE DE GESTION DE SAISIE DES DONNEES
        show_space(1)
        st.markdown(lb.PRESENTATION_REGLE_GESTION, unsafe_allow_html=True)
        
        show_space(1)
        
        sac.alert(label="Exemple d'évolution de règle", description=lb.PRESENTATION_REGLE_GESTION_ALERT_1, icon=True, closable=False)
        
        show_space(2)

    elif tabs == TAB_FORMAT:
        st.markdown(lb.PRESENTATION_FORMAT_DONNEES_1, unsafe_allow_html=True)
        show_image(image_MCD)
        
        show_space(1)

        st.markdown(lb.PRESENTATION_FORMAT_DONNEES_2, unsafe_allow_html=True)

        with st.expander('Lieux', False):
            cols_lieux = st.columns([2.5, 1])
            # Afficher les modalités de lieux O/N            
            with cols_lieux[-1]:
                label_lieux = sac.switch(label='détail', align='center', size='md', key=constant.FEATURE_LIEU)

            with cols_lieux[0]:
                st.markdown(lb.PRESENTATION_FORMAT_DONNEES_DETAIL_LIEU, unsafe_allow_html=True)
                
                for feature_key in constant.FEATURES_BY_TYPE[constant.FEATURE_LIEU] :
                    feature_value = constant.FEATURES[feature_key][constant.FEATURE_MODALITE]
                    # Afficher chaque variable
                    st.markdown(f'- __{feature_key}__')

                    if label_lieux==True and len(feature_value)>0 :
                        text = "<ul><ul>"
                        for modalite_key, modalite_value in feature_value.items() :
                            # Afficher les modalités de chaque variable
                            text +=  f"<li>{modalite_key} - {modalite_value} </li><br />"
                            pass
                        text += "</ul></ul>"
                          
                        sac.alert(label=text, size='xs', variant='quote-light', key="alert"+feature_key)



        with st.expander('Caractéristiques', False):
            cols_caracteristiques = st.columns([2.5, 1])
            # Afficher les modalités de caractéristiques O/N            
            with cols_caracteristiques[-1]:
                label_caracteristiques = sac.switch(label='détail', align='center', size='md', key=constant.FEATURE_CARACTERISTIQUE)

            with cols_caracteristiques[0]:
                st.markdown(lb.PRESENTATION_FORMAT_DONNEES_DETAIL_CARAC, unsafe_allow_html=True)
                
                for feature_key in constant.FEATURES_BY_TYPE[constant.FEATURE_CARACTERISTIQUE] :
                    feature_value = constant.FEATURES[feature_key][constant.FEATURE_MODALITE]
                    # Afficher chaque variable
                    st.markdown(f'- __{feature_key}__')

                    if label_caracteristiques==True and len(feature_value)>0 :
                        text = "<ul><ul>"
                        for modalite_key, modalite_value in feature_value.items() :
                            # Afficher les modalités de chaque variable
                            text +=  f"<li>{modalite_key} - {modalite_value} </li><br />"
                            pass
                        text += "</ul></ul>"
                          
                        sac.alert(label=text, size='xs', variant='quote-light', key="alert"+feature_key)


        with st.expander('Véhicules', False):
            cols_vehicules = st.columns([2.5, 1])
            # Afficher les modalités de véhicules O/N            
            with cols_vehicules[-1]:
                label_vehicules = sac.switch(label='détail', align='center', size='md', key=constant.FEATURE_VEHICULE)

            with cols_vehicules[0]:
                st.markdown(lb.PRESENTATION_FORMAT_DONNEES_DETAIL_VEHI, unsafe_allow_html=True)
                
                for feature_key in constant.FEATURES_BY_TYPE[constant.FEATURE_VEHICULE] :
                    feature_value = constant.FEATURES[feature_key][constant.FEATURE_MODALITE]
                    # Afficher chaque variable
                    st.markdown(f'- __{feature_key}__')

                    if label_vehicules==True and len(feature_value)>0 :
                        text = "<ul><ul>"
                        for modalite_key, modalite_value in feature_value.items() :
                            # Afficher les modalités de chaque variable
                            text +=  f"<li>{modalite_key} - {modalite_value} </li><br />"
                            pass
                        text += "</ul></ul>"
                          
                        sac.alert(label=text, size='xs', variant='quote-light', key="alert"+feature_key)


        with st.expander('Usagers', False):
            cols_usagers = st.columns([2.5, 1])
            # Afficher les modalités de usagers O/N            
            with cols_usagers[-1]:
                label_usagers = sac.switch(label='détail', align='center', size='md', key=constant.FEATURE_USAGER)

            with cols_usagers[0]:
                st.markdown(lb.PRESENTATION_FORMAT_DONNEES_DETAIL_USAGER, unsafe_allow_html=True)
                
                for feature_key in constant.FEATURES_BY_TYPE[constant.FEATURE_USAGER] :
                    feature_value = constant.FEATURES[feature_key][constant.FEATURE_MODALITE]
                    # Afficher chaque variable
                    st.markdown(f'- __{feature_key}__')

                    if label_usagers==True and len(feature_value)>0 :
                        text = "<ul><ul>"
                        for modalite_key, modalite_value in feature_value.items() :
                            # Afficher les modalités de chaque variable
                            text +=  f"<li>{modalite_key} - {modalite_value} </li><br />"
                            pass
                        text += "</ul></ul>"
                          
                        sac.alert(label=text, size='xs', variant='quote-light', key="alert"+feature_key)


    elif tabs == TAB_MERGE:
        st.markdown(lb.PRESENTATION_MERGE_PREMIERE_ETAPE, unsafe_allow_html=True)
        sac.alert(label="Commentaire", description=lb.PRESENTATION_MERGE_ALERT_1, icon=True)
        show_space(1)
        st.markdown(lb.PRESENTATION_MERGE_SECONDE_ETAPE, unsafe_allow_html=True)
        sac.alert(label="Commentaire", description=lb.PRESENTATION_MERGE_ALERT_2, icon=True)        
        show_space(1)

        # AFFICHAGE CODE : MERGE FICHIERS DE DONNEES
        affiche_code = sac.switch(label='afficher le code', align='start', size='sm')
        if affiche_code :
            st.code(lb.PRESENTATION_MERGE_CODE_1, line_numbers=True)
            st.markdown(lb.PRESENTATION_MERGE_RESULTAT_1, unsafe_allow_html=True)
            st.code(lb.PRESENTATION_MERGE_CODE_2, line_numbers=True)
            st.markdown(lb.PRESENTATION_MERGE_RESULTAT_2, unsafe_allow_html=True)
            st.code(lb.PRESENTATION_MERGE_CODE_3, line_numbers=True)
            st.markdown(lb.PRESENTATION_MERGE_RESULTAT_3, unsafe_allow_html=True)

        # Affichage des données 
        affiche_data = sac.switch(label='afficher les données', align='start', size='sm')
        if affiche_data :
            st.write(data.head())
        
        # Afficher la courbe de nombre de d'accidents entre 2005 à 2022
        affiche_courbe_tendance = sac.switch(label="afficher la courbe d'évolution du nombre d'accident par année", align='start', size='sm')
        if affiche_courbe_tendance :
            show_image(image_volume_df)
            sac.alert(label="Nombre d'enregistrements", description="Total de 2636355 enregistrements et 81 variables", variant='outline', icon=True)        

        
        st.markdown(lb.PRESENTATION_MERGE_TROISIEME_ETAPE, unsafe_allow_html=True)
        sac.alert(label=lb.PRESENTATION_MERGE_LISTE_AGG, size='xs', variant='quote-light')

        with st.expander('code', False):
            st.code(lb.PRESENTATION_MERGE_CODE_4, line_numbers=True)


    else:
        show_error_404()
        pass