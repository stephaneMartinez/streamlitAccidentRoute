import streamlit as st
import streamlit_antd_components as sac
from Librairies.utils import *
import Librairies.texte as lb
import Librairies.constant as constant

TAB_CONCLUSION="Conclusions de l'étude"
TAB_PERSPECTIVE="Perspectives et axes d'amélioration"


def main():
    # ------
    # HEADER
    # ------
    st.subheader('Conclusions et perspectives', anchor=False)
    #st.markdown('''<p style="text-align:justify;">

    # ---------------------
    # CHARGEMENT DES DONNES
    # ---------------------
    #data = exploration.load_data()

    # ----------------------
    # MENU ONGLETS PRINCIPAL
    # ----------------------
    tabs = sac.tabs([
        sac.TabsItem(TAB_CONCLUSION),
        sac.TabsItem(TAB_PERSPECTIVE)
        ], size='sm')
    
    # TRANSFORMER
    if tabs == TAB_CONCLUSION :
        # OBJECTIF ET METHODOLOGIE
        st.markdown(lb.CONCLUSION, unsafe_allow_html=True)    

    elif tabs== TAB_PERSPECTIVE:
        st.markdown(lb.PERSPECTIVE, unsafe_allow_html=True)    