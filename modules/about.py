import streamlit as st
import streamlit_antd_components as sac
import Librairies.constant as constant
from Librairies.utils import *
import Librairies.texte as lb

# Constantes
TAB_ETAPE_PARTIE1 = 'PARTIE 1 - Le jeu de données'
TAB_ETAPE_PARTIE2 = 'PARTIE 2 - Modélisation'
TAB_ETAPE_CONCLUSION = 'Conclusions et perspectives'

def main():
    st.subheader('Projet Fil Rouge Accident de la route', anchor=False)
    st.write(lb.LE_PROJET_HEADER)

    _,col2, _ = st.columns([0.35, 0.3, 0.35])
    with col2 :
        show_image(constant.CHEMIN_IMAGE+"pageGarde.jpg")

    tabs = sac.tabs([sac.TabsItem('Le projet', icon='easel'), sac.TabsItem("L'équipe", icon='cursor')], size='sm')
    if tabs == 'Le projet':
        show_space(1)
        st.markdown(lb.LE_PROJET_INTRODUCTION, unsafe_allow_html=True)
        show_space(2)

        '''
        tabs_etapes = sac.tabs([sac.TabsItem(TAB_ETAPE_PARTIE1), 
                                 sac.TabsItem(TAB_ETAPE_PARTIE2), 
                                 sac.TabsItem(TAB_ETAPE_CONCLUSION)],
                                 size='sm')
        c = st.columns([1])

        if tabs_etapes == TAB_ETAPE_PARTIE1 :
            # Détailler la phase de data visualisation
            sac.steps(
                items=[
                    sac.StepsItem(title='step 1', description='Découverte des données'),
                    sac.StepsItem(title='step 2', description='Aggrégation des données en un seul DataFrame'),
                    sac.StepsItem(title='step 3', description='Data Visualisation'),
                    sac.StepsItem(title='step 4', description='Tests statistiques'),
                ], 
            )
            # Détail step sélectionné



        elif tabs_etapes == TAB_ETAPE_PARTIE2 :
            # Détailler la phase de modélisation
            sac.steps(
                items=[
                    sac.StepsItem(title='step 1', description='Phase de préprocessing'),
                    sac.StepsItem(title='step 2', description='Phase de Machine Learning'),
                    sac.StepsItem(title='step 3', description='Phase de Deep Learning'),
                    sac.StepsItem(title='step 4', description="Phase d'interprétabilité"),
                ], 
            )

        else :
            # Détailler les conclusions et perspectives
             sac.steps(
                items=[
                    sac.StepsItem(title='Les résultats', description='Comparaison de chaque modèle'),
                    sac.StepsItem(title="Les perspectives", description="d'amélioration de nos modèles"),
                    sac.StepsItem(title='En synthèse', description='Ce qui le projet nous a apporté'),
                ], 
            )
        '''

    else:
        # --------
        # L'EQUIPE
        # --------
        st.markdown(lb.LE_PROJET_EQUIPE, unsafe_allow_html=True)
