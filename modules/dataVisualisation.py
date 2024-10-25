import streamlit as st
import streamlit_antd_components as sac
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
import seaborn as sns
import pydeck as pdk
from Librairies.utils import *
import Librairies.constant as constant
import Librairies.exploration as exploration
import Librairies.texte as lb

# Constantes
TAB_METHODE = "Méthodologie"
TAB_SIMU = "Data'Viz"
TAB_TESTS = "Tests Statistiques"

EXPENDER_CARACTERISTIQUES = "Approche par Caractéristiques"
EXPENDER_LIEUX = "Approche par Lieux"
EXPENDER_VEHICULE = "Approche par Véhicules"
EXPENDER_USAGER = "Approche par Usagers"

# Onglet - Méthode
EXPENDER_PROFONDEUR_HISTORIQUE = "1. Profondeur d'historique"
EXPENDEUR_VARIABLE_CIBLE = "2. Analyse de la variable cible"
EXPENDEUR_VARIABLE_TECHNIQUE = "3. Suppression des données techniques"
EXPENDEUR_ANALYSE_DETAILLEE = "4. Analyse de chaque donnée"
EXPENDEUR_VARIABLES_EXPLICATIVES = "5. Sélection des variables explicatives"

# Onglet - Data'Viz
EXPENDER_PARAMS = "Sélection"
EXPENDER_TEMPORAL = "Evolution dans le temps"
EXPENDER_DISTRIBUTION = "Distribution"
EXPENDER_TEST_STAT = "Test Statistique"


# Palette de couleurs personnalisée pour chaque année
color_map = {
    0:    '#e4f5ff',                                    # Rose pâle
    2018: '#ff9999',                                    # Rose pâle
    1:    '#9ed7ff',                                    # Bleu clair
    2019: '#66b3ff',                                    # Bleu clair
    2:    '#4fa9f9',                                    # Vert pâle
    2020: '#99ff99',                                    # Vert pâle
    3:    '#0b70d0',                                    # Orange pâle
    2021: '#ffcc99',                                    # Orange pâle
    4:    '#004280',                                    # Violet pâle
    2022: '#c2c2f0'                                     # Violet pâle
} 

def filtre_data() :
    st.write("Change")

def main():
    # ------
    # HEADER
    # ------
    st.subheader('Exploration des données', anchor=False)
    st.markdown('''<p style="text-align:justify;">
                La phase d'exploration des données présentée ici l'a été à partir de notre DataFrame consolidé "Usager" (maille la plus fine) en ne conservant que les années 2018-2022 soit les 5 dernières années disponibles.         
                </p>''', unsafe_allow_html=True)

    # ---------------------
    # CHARGEMENT DES DONNES
    # ---------------------
    data = exploration.load_data()

    # ----------------------
    # MENU ONGLETS PRINCIPAL
    # ----------------------
    tabs = sac.tabs([
        sac.TabsItem(TAB_METHODE),
        sac.TabsItem(TAB_SIMU), 
        #sac.TabsItem(TAB_TESTS)
        ], size='sm')
    if tabs == TAB_METHODE :
        # ----------------------------------------
        # MODE OPERATOIRE DE L'ANALYSE DES DONNEES
        # ----------------------------------------
        # INTRODUCTION
        # ------------
        st.markdown(lb.DATAVIZ_METHODE_INTRODUCTION, unsafe_allow_html=True)

        show_space(1)

        # 1. PROFONDEUR D'HISTORIQUE
        # -----------------------
        with st.expander(EXPENDER_PROFONDEUR_HISTORIQUE, False):
            st.markdown(lb.DATAVIZ_METHODE_PERIMETRE_DATA, unsafe_allow_html=True)
            show_space(1)
            sac.alert(label="Profondeur des données", description=lb.DATAVIZ_ALERT_PROFONDEUR_DATA_1, variant='transparent', icon=True, closable=False)
            show_space(1)
            show_image(constant.CHEMIN_IMAGE+"ProfondeurData.png")
            show_space(1)
            sac.alert(label="Profondeur des données", description=lb.DATAVIZ_ALERT_PROFONDEUR_DATA_2, variant='quote-light', icon=True, closable=False)

            show_space(1)

        # 2. VARIABLE CIBLE
        # --------------
        with st.expander(EXPENDEUR_VARIABLE_CIBLE, False) :
            st.markdown(lb.DATAVIZ_METHODE_VARIABLE_CIBLE, unsafe_allow_html=True)
            show_space(1)
            sac.alert(label="Valeurs manquantes sur variable cible", description=lb.DATAVIZ_METHODE_ALERT_CIBLE1, variant='transparent', icon=True, closable=False)
        
            show_space(0)
        
            # GRAPHE VARIABLE CIBLE
            code_to_xlabel = {'0' : 'Indemne', '1' : 'Blessé léger', '2' : 'Blessé hospitalisé', '3' : 'Tué', '-1' : 'Non définie'}
            exploration.moncountplot (
                df=data,
                x='user_gravite', 
                title="Variable cible 'user_gravite'", 
                code_to_xlabel=code_to_xlabel, 
                ascending=False,
                viewPercent=0.0)

            show_space(1)
            sac.alert(label="Déséquilibre de classes", description=lb.DATAVIZ_METHODE_ALERT_CIBLE2, variant='quote-light', icon=True, closable=False)

            show_space(1)

        # 3. SUPPRESSION DES DONNEES TECHNIQUES
        # ----------------------------------
        with st.expander(EXPENDEUR_VARIABLE_TECHNIQUE, False) :
            st.markdown(lb.DATAVIZ_METHODE_VARIABLES_INUTILES, unsafe_allow_html=True)

            show_space(1)

        # 4. ANALYSE DETAILLEE DE CHAQUE VARIABLE
        # ---------------------------------------
        with st.expander(EXPENDEUR_ANALYSE_DETAILLEE, False) :
            st.markdown(lb.DATAVIZ_METHODE_ANALYSE_VARIABLE, unsafe_allow_html=True)

            show_space(1)

            affiche_outliers = sac.switch(label="Exemple d'analyse des valeurs aberrantes réalisée sur la variable 'Vitesse maximale'", align='start', size='sm')
            if affiche_outliers :
                show_image(constant.CHEMIN_IMAGE+"Outliers1.png")
                st.markdown(lb.DATAVIZ_METHODE_ANALYSE_OUTLIERS, unsafe_allow_html=True)
                show_image(constant.CHEMIN_IMAGE+"Outliers2.png")

            show_space(1)

            affiche_pair = sac.switch(label='Diagramme de pairs', align='start', size='sm')
            if affiche_pair :
                show_image("donnees/images/PairPlot.png")
                st.markdown(lb.DATAVIZ_METHODE_DIAGRAMME_DE_PAIRS, unsafe_allow_html=True)
            
                show_space(1)
                sac.alert(label="Diagramme de pairs", description=lb.DATAVIZ_METHODE_ALERT_PAIRS, variant='quote-light', icon=True, closable=True)

            show_space(1)

            affiche_multicolinearite = sac.switch(label='Recherche de multicolinéarité', align='start', size='sm')
            if affiche_multicolinearite :
                st.markdown(lb.DATAVIZ_METHODE_ANALYSE_COLINEARITE, unsafe_allow_html=True)
                show_image(constant.CHEMIN_IMAGE+"HeatMap.png")
                show_space(1)
                sac.alert(label="Variance inflation vector (VIF)", description=lb.DATAVIZ_METHODE_ALERT_MULTICOLINEARITE, variant='quote-light', icon=True, closable=True)

            show_space(1)

        # 5. SELECTION DES VARIABLES EXPLICATIVES
        # -------------------------------------------
        with st.expander(EXPENDEUR_ANALYSE_DETAILLEE, False) :
            st.markdown(lb.DATAVIZ_METHODE_SELECTION_EXPLICATIVE, unsafe_allow_html=True)

            affiche_top10 = sac.switch(label='afficher le top 10', align='center', size='md')
            if affiche_top10 :
                show_image(constant.CHEMIN_IMAGE+"top10_variablesExplicatives.png")

            affiche_flop10 = sac.switch(label='afficher le flop 10', align='center', size='md')
            if affiche_flop10 :
                show_image(constant.CHEMIN_IMAGE+"flop10_variablesExplicatives.png")

            show_image(constant.CHEMIN_IMAGE+"selection_variablesExplicatives.png")
            sac.alert(label="Sélection des variables explicatives", description=lb.DATAVIZ_METHODE_ALERT_VARIABLE_EXPLICATIVE, variant='quote-light', icon=True, closable=True)
        

    elif tabs == TAB_SIMU:
        # --------------------------DATA'VIZ INTERACTIVE -------------------------------
        data_gps = data[(data[constant.CARAC_GPS_LAT].isna()==False) & (data[constant.CARAC_GPS_LONG].isna()==False)]
        with st.expander(EXPENDER_PARAMS, True):            
            # FILTRES - ANNEES
            items = [annee for annee in range(2020, 2023)]
            selected_annee = sac.transfer(
                items=items, 
                label = 'Années',
                titles=['Disponible', 'Sélectionnée'], 
                align='center', 
                oneway=True,
            )
            show_space(1)

            # FILTRES - CATEGORIE DE VEHICULES
            selected_vehi = sac.chip(
                items=[
                    sac.ChipItem(label='vélo'),
                    sac.ChipItem(label='sans permis'),
                    sac.ChipItem(label='moto'),
                    sac.ChipItem(label='vl'),
                    sac.ChipItem(label='pl'),
                    sac.ChipItem(label='autres'),
                ], label='Catégorie de véhicule', index=[], align='left', radius='md', multiple=True, return_index=True
            )
            show_space(1)

            # FILTRES - VARIABLE CIBLE
            selected_gravite = sac.chip(
                items=[
                    sac.ChipItem(label='indemne'),
                    sac.ChipItem(label='blessés légers'),
                    sac.ChipItem(label='blessés hospitalisés'),
                    sac.ChipItem(label='tués'),
                ], label="Gravité de l'accident", index=[], align='left', radius='md', multiple=True, return_index=True
            )
    

            if selected_annee and selected_vehi and selected_gravite  :
                
                # FILTRE DU DF AVEC LES CRITERE SELECTIONNES
                selected_years = [int(item) for item in selected_annee]
                
                filtre_annee = [int(item) for item in selected_annee]
                filtre_vehi = [x + 1 for x in selected_vehi]
                filtered_df = data_gps[
                    (data_gps['carac_an'].isin(filtre_annee)) & (data_gps['user_gravite'].isin(selected_gravite)) & (data_gps['agg_catv_perso'].isin(filtre_vehi))]
                

                nb_selected_records = filtered_df.shape
                sac.result(label=f"{nb_selected_records}", description=f"Nombre d'enregistrement(s) sélectionné(s)", status='success')

            else :
                description =  'une année' if not selected_annee else 'une catégorie de véhicule' if not selected_vehi else 'une gravité'
                sac.result(label='Veuillez sélectionner au moins', description=description)


        with st.expander('Variable cible', False):
            try:
                if filtre_annee :
                    pass
            except:
                filtre_annee=[]

            try:
                if filtre_vehi :
                    pass
            except:
                filtre_vehi=[]      

            filtered_df = data_gps[
                (data_gps['carac_an'].isin(filtre_annee)) & (data_gps['user_gravite'].isin(selected_gravite)) & (data_gps['agg_catv_perso'].isin(filtre_vehi))]

            if filtered_df.shape[0]>0 :    
                code_to_xlabel = {'0' : 'Indemne', '1' : 'Blessé léger', '2' : 'Blessé hospitalisé', '3' : 'Tué', '-1' : 'Non définie'}
                exploration.moncountplot (
                    df=filtered_df,
                    x='user_gravite', 
                    title="Gravité des accidents", 
                    code_to_xlabel=code_to_xlabel, 
                    viewPercent=0.0)

        # -----------------------
        # EVOLUTION DANS LE TEMPS
        # -----------------------
        with st.expander('Evolution dans le temps', False):
            try:
                if filtre_annee :
                    pass
            except:
                filtre_annee=[]

            try:
                if filtre_vehi :
                    pass
            except:
                filtre_vehi=[]      

            filtered_df = exploration.get_filtered_df (data_gps, filtre_annee, selected_gravite, filtre_vehi)
            if filtered_df.shape[0]>0 :
                if 0 in selected_gravite and filtered_df.shape[0]>0:
                    st.write ("LES PERSONNES INDEMNES")
                    
                    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 3))
                    df_gravite0 = filtered_df[filtered_df.user_gravite==0]

                    # Premier graphique sur le premier axe
                    sns.countplot(x=df_gravite0['carac_an'], palette='husl', ax=axes[0])
                    axes[0].set_title("Evolution du nombre d'accidents par année")
                    axes[0].set_xlabel('Année')
                    axes[0].set_ylabel("INDEMNES")

                    # Deuxième graphique sur le deuxième axe (par exemple un autre graphique)
                    sns.lineplot(x='carac_mois', y='count',  hue='carac_an', style='carac_an',  estimator='sum', palette='husl', data=df_gravite0, ax=axes[1])
                    axes[1].set_title("Evolution du nombre d'accidents par mois")
                    axes[1].set_xlabel('Mois')
                    axes[1].set_ylabel("Nombre d'accidents")

                    axes[1].legend(title='Année', bbox_to_anchor=(1, 0), loc='lower right')

                    # Ajuster l'espace entre les graphiques
                    plt.tight_layout()

                    # Afficher la figure dans Streamlit
                    st.pyplot(fig)

                if 1 in selected_gravite and filtered_df.shape[0]>0:
                    st.write ("LES BLESSES LEGERS")

                    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 3))
                    df_gravite1 = filtered_df[filtered_df.user_gravite==1]

                    # Premier graphique sur le premier axe
                    sns.countplot(x=df_gravite1['carac_an'], palette='husl', ax=axes[0])
                    axes[0].set_title("Evolution du nombre d'accidents par année")
                    axes[0].set_xlabel('Année')
                    axes[0].set_ylabel("BLESSES LEGERS")

                    # Deuxième graphique sur le deuxième axe (par exemple un autre graphique)
                    sns.lineplot(x='carac_mois', y='count',  hue='carac_an', style='carac_an',  estimator='sum', palette='husl', data=df_gravite1, ax=axes[1])
                    axes[1].set_title("Evolution du nombre d'accidents par mois")
                    axes[1].set_xlabel('Mois')
                    axes[1].set_ylabel("Nombre d'accidents")

                    axes[1].legend(title='Année', bbox_to_anchor=(1, 0), loc='lower right')

                    # Ajuster l'espace entre les graphiques
                    plt.tight_layout()

                    # Afficher la figure dans Streamlit
                    st.pyplot(fig)

                if 2 in selected_gravite and filtered_df.shape[0]>0:
                    st.write ("LES BLESSES HOSPITALISES")

                    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 3))
                    df_gravite2 = filtered_df[filtered_df.user_gravite==2]

                    # Premier graphique sur le premier axe
                    sns.countplot(x=df_gravite2['carac_an'], palette='husl', ax=axes[0])
                    axes[0].set_title("Evolution du nombre d'accidents par année")
                    axes[0].set_xlabel('Année')
                    axes[0].set_ylabel("BLESSES HOSPITALISES")

                    # Deuxième graphique sur le deuxième axe (par exemple un autre graphique)
                    sns.lineplot(x='carac_mois', y='count',  hue='carac_an', style='carac_an',  estimator='sum', palette='husl', data=df_gravite2, ax=axes[1])
                    axes[1].set_title("Evolution du nombre d'accidents par mois")
                    axes[1].set_xlabel('Mois')
                    axes[1].set_ylabel("Nombre d'accidents")

                    axes[1].legend(title='Année', bbox_to_anchor=(1, 0), loc='lower right')

                    # Ajuster l'espace entre les graphiques
                    plt.tight_layout()

                    # Afficher la figure dans Streamlit
                    st.pyplot(fig)

                if 3 in selected_gravite and filtered_df.shape[0]>0:
                    st.write ("LES TUES")

                    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 3))
                    df_gravite3 = filtered_df[filtered_df.user_gravite==3]

                    # Premier graphique sur le premier axe
                    sns.countplot(x=df_gravite3['carac_an'], palette='husl', ax=axes[0])
                    axes[0].set_title("Evolution du nombre d'accidents par année")
                    axes[0].set_xlabel('Année')
                    axes[0].set_ylabel("PERSONNES TUEES")

                    # Deuxième graphique sur le deuxième axe (par exemple un autre graphique)
                    sns.lineplot(x='carac_mois', y='count',  hue='carac_an', style='carac_an',  estimator='sum', palette='husl', data=df_gravite3, ax=axes[1])
                    axes[1].set_title("Evolution du nombre d'accidents par mois")
                    axes[1].set_xlabel('Mois')
                    axes[1].set_ylabel("Nombre d'accidents")

                    axes[1].legend(title='Année', bbox_to_anchor=(1, 0), loc='lower right')

                    # Ajuster l'espace entre les graphiques
                    plt.tight_layout()

                    # Afficher la figure dans Streamlit
                    st.pyplot(fig)

                # ---------------------------
                # DETAIL PAR TRANCHE HORAIRES
                # ---------------------------
                sac.divider(label='Visualisation par tranches horaires', icon='', align='center', variant='dashed', color='gray')
                hourly_counts = filtered_df.groupby('heure').size().reset_index(name='count')

                # ETAPE 2 - Vérification visuelle --> Distribution des accidents par tranches horaires
                fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
                try:
                    sns.barplot (x='heure', y='count',  data=hourly_counts, palette='husl', ax=axes)
            
                except :
                    pass
                
                # Pourcentages
                #total = filtered_df[filtered_df.user_gravite==3]['count'].sum()
                total = filtered_df['count'].sum()
                for p in axes.patches:
                    height = p.get_height()
                    percent=(height / total * 100)
                    if percent> 5.2 :
                        axes.annotate(f'{percent:.1f}%', 
                            (p.get_x() + p.get_width() / 2., height), 
                            ha='center', va='center', 
                            fontsize=10, color='black', 
                            xytext=(0, 5), 
                            textcoords='offset points')

                axes.set_title("Distribution des accidents par tranches horaires")
                axes.set_ylabel("Nombre d'accidents")
                axes.set_xlabel("Tranches horaires")

                # Afficher la figure dans Streamlit
                st.pyplot(fig)

        with st.expander('Etude par département', False):
            try:
                if filtre_annee :
                    pass
            except:
                filtre_annee=[]

            try:
                if filtre_vehi :
                    pass
            except:
                filtre_vehi=[]      

            filtered_df = exploration.get_filtered_df (data_gps, filtre_annee, selected_gravite, filtre_vehi)

            if filtered_df.shape[0]>0 :    
                fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

                # Définir les couleurs pour chaque année
                # A Gauche : les 10 départements les plus accidentogènes
                nbAccidents = filtered_df.carac_dept.value_counts().head(10)
                try :
                    sns.barplot (x=nbAccidents.index, y=nbAccidents,  order=nbAccidents.index, palette='husl', ax=axes[0])
                
                except :
                    pass

                # Pourcentages
                total = filtered_df['count'].sum()
                for p in axes[0].patches:
                    height = p.get_height()
                    axes[0].annotate(
                        f'{(height / total * 100):.1f}%', 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha='center', va='center', 
                        fontsize=10, color='black', 
                        xytext=(0, 5), 
                        textcoords='offset points'
                    )

                axes[0].set_title("Départements les plus accidentogènes")
                axes[0].set_ylabel("Nombre d'accidents")
                axes[0].set_xlabel("Départements")

                # A droite les 10 les moins accidentogènes
                nbAccidents = filtered_df.carac_dept.value_counts().tail(10)
                try :
                    sns.barplot (x=nbAccidents.index, y=nbAccidents,  order=nbAccidents.index, palette='husl', ax=axes[1])
            
                except :
                    pass

                # Pourcentages
                for p in axes[1].patches:
                    height = p.get_height()
                    axes[1].annotate(
                        f'{(height / total * 100):.1f}%', 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha='center', va='center', 
                        fontsize=10, color='black', 
                        xytext=(0, 5), 
                        textcoords='offset points'
                    )

                axes[1].set_title("Départements les moins accidentogènes")
                axes[1].set_ylabel("Nombre d'accidents")
                axes[1].set_xlabel("Départements")
    
                # Afficher la figure dans Streamlit
                st.pyplot(fig)

        with st.expander('Cartographie des zones accidentogènes', False):
            # AFFICHAGE SUR UNE CARTE DE FRANCE
            # ---------------------------------
            try:
                if filtre_annee :
                    pass
            
            except:
                filtre_annee=[]

            try:
                if filtre_vehi :
                    pass
            
            except:
                filtre_vehi=[]      

            filtered_df = exploration.get_filtered_df (data_gps, filtre_annee, selected_gravite, filtre_vehi)
            if filtered_df.shape[0]>0:
                if filtered_df.shape[0]<30000 :
                    scatter_layer = pdk.Layer(
                        "ScatterplotLayer",
                        filtered_df,
                        get_position=f"[{constant.CARAC_GPS_LONG}, {constant.CARAC_GPS_LAT}]",
                        get_radius=300,  # Ajuste la taille des points
                        get_color='[200, 30, 0, 160]',  # Rouge avec transparence (tu peux personnaliser les couleurs)
                        radius_min_pixels=3,  # Taille minimale des points en pixels (même si on dézoome beaucoup)
                        radius_max_pixels=20,  # Taille maximale des points en pixels (pour éviter qu'ils soient trop gros)
                        pickable=True
                    )

                    # Vue initiale de la carte (centres sur les points)
                    view_state = pdk.ViewState(
                        latitude=filtered_df[constant.CARAC_GPS_LAT].mean(),
                        longitude=filtered_df[constant.CARAC_GPS_LONG].mean(),
                        zoom=5,  # Ajuste le niveau de zoom
                        pitch=50   # Tu peux incliner la carte en modifiant cette valeur (0 signifie pas d'inclinaison)
                    )

                    # Créer la carte pydeck avec les paramètres définis
                    r = pdk.Deck(
                        layers=[scatter_layer],
                        initial_view_state=view_state,
                        tooltip={"text": f"Gravité: {{{selected_gravite}}}"},
                        map_style='mapbox://styles/mapbox/light-v9'
                    )

                    st.pydeck_chart(r)
            
                else :
                    st.map (
                        filtered_df, 
                        latitude=constant.CARAC_GPS_LAT, 
                        longitude=constant.CARAC_GPS_LONG
                    )


                # -----------------
                # CARTE PAR DENSITE
                # -----------------
                if filtered_df.shape[0]<40000 :
                    # Centrer la carte sur une latitude et longitude moyenne
                    midpoint = (filtered_df[constant.CARAC_GPS_LAT].mean(), filtered_df[constant.CARAC_GPS_LONG].mean())

                    # Définition de la couche de densité (HeatmapLayer)
                    heatmap_layer = pdk.Layer(
                        "HeatmapLayer",
                        filtered_df,
                        opacity=0.9,
                        get_position=f'[{constant.CARAC_GPS_LONG}, {constant.CARAC_GPS_LAT}]',
                        get_weight=5,                           # Utilise cette colonne pour intensifier la chaleur
                    )
            
                    # Définition de la carte avec pydeck
                    view_state = pdk.ViewState(
                        latitude=midpoint[0],
                        longitude=midpoint[1],
                        zoom=5,                                 # Niveau de zoom
                        pitch=50                                 # Inclinaison de la vue
                    )

                    # Créer la carte Pydeck
                    r = pdk.Deck(
                        layers=[heatmap_layer],
                        initial_view_state=view_state,
                        map_style="mapbox://styles/mapbox/light-v10",  # Choix du style de la carte
                    )

                    # Affichage de la carte avec Streamlit
                    st.write("Carte de densité des accidents")
                    st.pydeck_chart(r)
            
                elif filtered_df.shape[0]>0 :
                    sac.result(label="Carte de densité", description=f'Veuillez limiter le nombre de données à afficher.', status='error')

        with st.expander('Analyse détaillée par variable', False):
            try:
                if filtre_annee :
                    pass
            
            except:
                filtre_annee=[]

            try:
                if filtre_vehi :
                    pass
            
            except:
                filtre_vehi=[]      

            filtered_df = exploration.get_filtered_df (data_gps, filtre_annee, selected_gravite, filtre_vehi)
            if filtered_df.shape[0]>0:
                # FILTRE - UNE COLONNE EN PARTICULIER
                categorial_columns = [ key for key, value in constant.FEATURES.items()]

                # SELECTION D'UNE COLONNE POUR ANALYSE   
                show_space(1)         
                selected_column = st.selectbox('Sélectionnez une colonne', categorial_columns )

                # Retrouver le nom de la colonne du champs sélectionné
                feature = exploration.get_value_from_label(selected_column)

                # Afficher du texte expliquant la variable
                sac.alert(label=feature[constant.FEATURE_COLUMNNAME], description=feature[constant.FEATURE_DESIGNATION], icon=True, closable=True)

                # REPARTITION PAR VARIABLE PRE SELECTIONNEE
                mapping = feature[constant.FEATURE_MODALITE]
                if len(mapping)>0 :
                    titre = f"Répartition des accidents par {selected_column}"
                    # VARIABLE VATEGORIELLE - AFFICHAGE D'UN CAMEMBERT
                    data['col_label'] = data[feature[constant.FEATURE_COLUMNNAME]].map(mapping)
                    col_count = data['col_label'].value_counts().reset_index()
                    col_count.columns = [feature[constant.FEATURE_COLUMNNAME], 'Nombre d\'accidents']

                    fig_pie = px.pie(                                       # Création du graphique en camembert
                        col_count,
                        names=feature[constant.FEATURE_COLUMNNAME],
                        values='Nombre d\'accidents',
                        title=titre,
                        color=feature[constant.FEATURE_COLUMNNAME],                                          # Utilise une couleur distincte pour chaque catégorie
                        color_discrete_sequence=px.colors.qualitative.Set3  # Choisir une palette de couleurs
                    )
                    fig_pie.update_traces(textinfo='label+percent', pull=[0.1, 0.1, 0.1])  # Exemple d'espacement
                    fig_pie.update_traces(rotation=feature[constant.FEATURE_PIE_ROTATION])  
                    st.plotly_chart(fig_pie)                                # Afficher le graphique dans Streamlit


                # Distribution sous forme d'histogramme
                chart_placeholder = st.empty()

                
                if len(mapping)==0 :
                    titre=f"Distribution des accidents par {selected_column}"
                    #col_count = data.groupby(feature[constant.FEATURE_COLUMNNAME]).size().reset_index(name='Nombre d\'accidents')

                    num_bins = st.slider(
                        'Nombre de bins', 
                        min_value=feature[constant.FEATURE_DISTRI_MIN_BINS], 
                        max_value=feature[constant.FEATURE_DISTRI_MAX_BINS], 
                        value=feature[constant.FEATURE_DISTRI_DEFAULT_BINS]
                    )

                    #fig_distrib = px.bar(col_count, x=feature[constant.FEATURE_COLUMNNAME], y='Nombre d\'accidents', title=f"Distribution des accidents par {selected_column}")
                    fig_distrib = px.histogram(
                       filtered_df,
                        x=feature[constant.FEATURE_COLUMNNAME],
                        nbins=num_bins,  
                        title=titre
                    )

                    # Ajustement de l'espace entre les barres pour les rendre plus lisibles
                    fig_distrib.update_layout(
                        bargap=0.1,  # Réduit l'espace entre les barres
                    )

                    # Calculer la courbe de densité
                    col_values = filtered_df[feature[constant.FEATURE_COLUMNNAME]].dropna()  # Récupère les valeurs de la colonne lieu_vma sans les NaN
                    density = gaussian_kde(col_values)  # Densité de la distribution
                    x_vals = np.linspace(col_values.min(), col_values.max(), 200)  # Créer un ensemble de valeurs pour l'axe x
                    y_vals = density(x_vals)  # Calculer la densité pour ces valeurs x

                    # Ajouter la courbe de densité à l'histogramme
                    fig_distrib.add_trace(
                        go.Scatter(x=x_vals, y=y_vals * len(col_values),  # Multiplie par len(vma_values) pour ajuster à l'échelle
                        mode='lines', 
                        name='Courbe de densité', 
                        line=dict(color='red'))
                    )


                    chart_placeholder.plotly_chart(fig_distrib)                      # Afficher le graphique dans Streamlit
                

    # -------------------------
    # PARTIE TESTS STATISTIQUES
    # -------------------------
    else:
        show_error_404        
        
        

        