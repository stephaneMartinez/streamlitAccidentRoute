import streamlit as st
import streamlit_antd_components as sac
import Librairies.constant as constant 
import modules.about as about
import modules.dataPresentation as dataPresentation
import modules.dataVisualisation as dataVisualisation
import modules.preprocessing as preprocessing
import modules.machineLearning as ml
import modules.DNN as dnn
import modules.interpretabilite as interpretabilite
import modules.conclusion as conclusion
from Librairies.machineLearning import *

# Créer le menu dans la barre latérale
with st.sidebar:
    menu_lateral = sac.menu( 
    [
        #sac.MenuItem('home', icon='house-fill', tag=[sac.Tag('Tag1', color='green'), sac.Tag('Tag2', 'red')]),
        sac.MenuItem(constant.MENU_PROJET, icon='easel'),
        sac.MenuItem(constant.MENU_JDD, icon=sac.BsIcon(name='bar-chart', size=20), children=[
            sac.MenuItem(constant.MENU_JDD_PRESENTATION),
            sac.MenuItem(constant.MENU_JDD_DATAVIZ),
            
        ]),
        sac.MenuItem(constant.MENU_MODELISATION,icon=sac.BsIcon(name='bezier2', size=20),children=[
            sac.MenuItem(constant.MENU_MODELISATION_PREPROCESS),
            sac.MenuItem(constant.MENU_MODELISATION_ML),
            sac.MenuItem(constant.MENU_MODELISATION_DL),
            sac.MenuItem(constant.MENU_MODELISATION_INTERPRETABILITE),
        ]), 
        sac.MenuItem(type='divider'),
        #sac.MenuItem('link', type='group', children=[
        #    sac.MenuItem('antd-menu', icon='heart-fill', href='https://ant.design/components/menu#menu'),
        #    sac.MenuItem('bootstrap-icon', icon='bootstrap-fill', href='https://icons.getbootstrap.com/'),
        #]),
        sac.MenuItem(constant.MENU_CONCLUSION, icon=sac.BsIcon(name='flag', size=20))
    ], open_all=True)

# Navigation
if menu_lateral == constant.MENU_PROJET:
    about.main()
    
elif menu_lateral == constant.MENU_JDD_PRESENTATION:
    dataPresentation.main()

elif menu_lateral == constant.MENU_JDD_DATAVIZ:
    dataVisualisation.main()

elif menu_lateral == constant.MENU_MODELISATION_PREPROCESS:
    preprocessing.main()

elif menu_lateral == constant.MENU_MODELISATION_ML:
    ml.main()

elif menu_lateral == constant.MENU_MODELISATION_DL:
    dnn.main()

elif menu_lateral == constant.MENU_MODELISATION_INTERPRETABILITE:
    interpretabilite.main()

elif menu_lateral == constant.MENU_CONCLUSION:
    conclusion.main()