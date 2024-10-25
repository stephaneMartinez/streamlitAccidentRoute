import streamlit as st
import streamlit_antd_components as sac
from Librairies.utils import *
import Librairies.texte as lb
import Librairies.constant as constant
import Librairies.exploration as exploration
import Librairies.texte as lb
import pandas as pd

TAB_PIPELINE=" Pipeline"

def main():
    # ------
    # HEADER
    # ------
    st.subheader('Pré-processing des données', anchor=False)
    st.markdown('''<p style="text-align:justify;">
                Opération d'automatisation des actions préparatoires au données.        
                </p>''', unsafe_allow_html=True)

    # ---------------------
    # CHARGEMENT DES DONNES
    # ---------------------
    data = exploration.load_data()

    # ----------------------
    # MENU ONGLETS PRINCIPAL
    # ----------------------
    tabs = sac.tabs([
        sac.TabsItem(TAB_PIPELINE),
        #sac.TabsItem(TAB_SIMU), 
        #sac.TabsItem(TAB_TESTS)
        ], size='sm')
    
    # TRANSFORMER
    if tabs == TAB_PIPELINE :
        # OBJECTIF ET METHODOLOGIE
        st.markdown(lb.PREPROCESSING_PIPELINE_INTRODUCTION, unsafe_allow_html=True)    

        _, col2, _ = st.columns([0.3, 0.3, 0.3])
        with col2 :
            show_image(constant.CHEMIN_IMAGE+"GlobalPipeline.png", use_column_width=False)

        # PREPROCESSING VARIABLES EXPLICATIVES
        st.markdown(lb.PREPROCESSING_PIPELINE_PREPROCESS_VARIABLES, unsafe_allow_html=True)

        affiche_ex_dept = sac.switch(label="Exemple d'utilisation Variable 'Département'", align='start', size='sm')
        if affiche_ex_dept :
            _, col2, _ = st.columns(3)
            with col2 :    
                show_image(constant.CHEMIN_IMAGE+"Transformer_dept.png")
            st.code(lb.PREPROCESSING_PIPELINE_EX_DEPT_CODE, line_numbers=True)
            
        affiche_ex_vma = sac.switch(label="Exemple d'utilisation Variable 'Vitesse maximale'", align='start', size='sm')
        if affiche_ex_vma :
            _, col2, _ = st.columns(3)
            with col2 :     
                show_image(constant.CHEMIN_IMAGE+"Transformer_vma.png")
            st.code(lb.PREPROCESSING_PIPELINE_EX_VMA_CODE, line_numbers=True)

        # FEATURE ENGINEERING
        show_space(1)
        st.markdown(lb.PREPROCESSING_PIPELINE_FEATURE_ENGINEERING, unsafe_allow_html=True)
        affiche_cluster = sac.switch(label="Code transformer 'Cluster Geographique'", align='start', size='sm')
        if affiche_cluster :
            st.code(lb.PREPROCESSING_PIPELINE_FEATURE_ENGINEERING_CLUSTER_CODE, line_numbers=True)

        # PREPROCESSING PIPELINE
        st.markdown(lb.PREPROCESSING_PIPELINE_PREPROCESSING, unsafe_allow_html=True) 

        _, col2, _ = st.columns([0.3, 0.3, 0.3])
        with col2 :
            show_image(constant.CHEMIN_IMAGE+"PreprocessingPipeline.png", use_column_width=False)
        st.markdown(lb.PREPROCESSING_PIPELINE_PREPROCESSING_DETAIL, unsafe_allow_html=True) 

        affiche_pipeline_carac = sac.switch(label="Exemple de déclaration du pipeline 'CaracPipeline'", align='start', size='sm')
        if affiche_pipeline_carac :
            _, col2, _ = st.columns(3)
            with col2 : 
                show_image(constant.CHEMIN_IMAGE+"CaracPipeline.png", use_column_width=False)
            st.code(lb.PREPROCESSING_PIPELINE_TYPE_CARAC, line_numbers=True)
        
        show_space(1)
        st.markdown(lb.PREPROCESSING_PIPELINE_GLOBAL, unsafe_allow_html=True) 
        