import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import seaborn as sns
import statsmodels.api
from scipy.stats import chi2_contingency
from scipy.stats import pearsonr
import streamlit as st
import Librairies.constant as constant

#@st.cache_data(persist=False)
def load_data () :
    data = pd.read_csv("donnees/agg_usagers_2020_2022.csv")
    return data


def get_list_from_dict (modalsDict:dict, keys:bool=False) -> list :
    ''' A partir d'un dictionnaire sous la forme 1 : valeur1, 2 : valeur2, retourne une liste [valeur1, valeur2]
    '''
    liste = [value if keys==False else key for key, value in modalsDict.items()]
    
    return liste

def get_value_from_label (label:str)-> str:
    ''' Retourne le détail d'une feature sous forme de dictionnaire. On retrouvera le nom de la colonne,
        sa désignation, les modalités et autres paramétrages pour les graphiques
    '''
    item = constant.FEATURES.get(label, None)
        
    # Retourne la clé associée au label si elle existe
    return item

def get_filtered_df(data:pd.DataFrame, filtre_annee, filtre_gravite, filtre_vehi) -> pd.DataFrame :
    return data[
        (data['carac_an'].isin(filtre_annee)) & \
        (data['user_gravite'].isin(filtre_gravite)) & \
        (data['agg_catv_perso'].isin(filtre_vehi))
    ]
                
def interquantile (df: pd.DataFrame, x: str, afficheDetail: bool = True, exclure_1: bool = True):
    """ Analyse des Outliers d'une colonne de df """
    # Retire les -1 (valeur NaN) pour certaines variables 
    if exclure_1 :
        df_column = df.loc[(df[x] != -1), x]
        
    else :
        df_column = df.loc[:, x]


    # Analyse des Outliers via IQR :
    Q1 = df_column.quantile(0.25)
    Q3 = df_column.quantile(0.75)
    IQR = Q3 - Q1
    inf = Q1 - 1.5 * IQR
    sup = Q3 + 1.5 * IQR

    print ("-> Q1=", Q1, "- Q3=", Q3, "- IQR=", IQR, "- Borne inf=", inf, "- Borne sup=", sup)
    
    if afficheDetail or df_column[(df_column <= inf)].count() > 0 :
        st.write ("-> Outlier(s) inférieur(s) - min=", df_column[(df_column < inf)].min(),
               "- max=", df_column[(df_column < inf)].max(),
               "- nb=", df_column[(df_column < inf)].count()
        )
        
    if afficheDetail :
        st.write (df_column[(df_column < inf)].sort_values().values)

    if afficheDetail or df_column[(df_column >= sup)].count() > 0 :
        st.write ("-> Outlier(s) supérieur(s) - min=", df_column[(df_column > sup)].min(),
               "- max=", df_column[(df_column > sup)].max(),
               "Outliers nb=", df_column[(df_column > sup)].count()
        )
        
    if afficheDetail :
        st.write (df_column[(df_column > sup)].sort_values().values)
    
def monboxplot (
    df: pd.DataFrame, 
    x: str=None, 
    y: str=None, 
    title: str=None, 
    viewOutliers: bool=True, 
    exclure_1: bool=True,
    code_to_xlabel:dict={}, 
    rotation_xlabel: int=0,
    colors: list=['green', 'yellow', 'orange', 'red'],
    horizontal: bool=False,
    axe:plt.Axes=None) -> list:
    """ Affiche une figure composée de 2 graphes (un boxplot et une ditribution de la variable) de 1 à n boxplots (si x et y définis) 
        Affiche l'analyse des Outliers via les intervales inter Quantile. 

        - df : DataFrame contenant les données à afficher
        - x (str) : variable à analyser
        - y (str) : OPTIONNEL - variable complémentaire (varaible de découpage)
        - title (str) : OPTIONNEL - Titre du graphe
        - viewOutliers (bool) : OPTIONNEL - Le détail des outliers sera affichée sous forme de liste (True).
        - exclure_1 (bool) : OPTIONNEL - Les valeurs -1 seront exclues du graphe (True)
        - code_to_xlabel (dict) : OPTIONNEL - Dictionnaire permettant d'afficher le libellés plutôt que les modalité de 'x'
        - rotation_xlabel (int) : OPTIONNEL - Applique une rotation aux libellés sur l'axe des x
        - horizontal (bool) : OPTIONNEL - Affichage horizontal (True) ou vertical par défaut
        - colors ([]) : OPTIONNEL - Palette de couleurs à utiliser
        - axe (plt.Axes) : OPTIONNEL - Affichage dans cet axe, création d'un nouveau sinon
        
        Returns :
        - figure
        - axes[]
        
    """
    fig = None
    if axe is None :
        fig, axe = plt.subplots(1, 1, figsize=(15, 5))
        
    if horizontal :
        sns.boxplot(x=y, y=x, data = df, ax = axe, palette=colors)

    else :
        sns.boxplot(x=x, y=y, data = df, ax = axe, palette=colors)
        
    axe.set_title (title, fontsize = 12);
    st.write ("Outliers=") 
    if y is not None :
        interquantile(df, y, viewOutliers, exclure_1)
        
    if code_to_xlabel is not None :
        xticks_labels = axe.get_xticklabels()
        labels = [label.get_text() for label in xticks_labels]
        new_xlabels = [code_to_xlabel[label] if label in code_to_xlabel else label for label in labels]
        axe.set_xticks(ticks = range(len(new_xlabels)), labels = new_xlabels, rotation=rotation_xlabel);

    st.pyplot(fig)

    return fig, axe 
    
def moncountplot (df: pd.DataFrame, 
                  x: str, 
                  hue: str=None, 
                  title: str=None, 
                  code_to_xlabel: dict={}, 
                  rotation_xlabel: int=0,
                  code_to_huelabel: dict={},
                  viewPercent: float=0.0,
                  ascending: bool=None,
                  axe:plt.Axes=None ) -> list:
    """ Affiche un Countplot personnalisé :
        - df : DataFrame contenant les données à afficher
        - x (str) : nom de la variable à afficher sur l'axe des abscisses 
        - hue (str) : OPTIONNEL - variable complémentaire (découpage graphe par cette variable) 
        - title (str) : OPTIONNEL - Titre à afficher
        - code_to_xlabel (dict) : OPTIONNEL - Libellés à privilégier pour les modalités de la variable 'x' (les modalités sinon)
        - rotation_xlabel (int) : OPTIONNEL - Angle de rotation des libellés de l'axe des absisses
        - code_to_huelabel (dict) : OPTIONNEL - Libellés à privilégierpour les modalités de la variable 'hue' (les modalités sinon)
        - viewPercent (float) : OPTIONNEL - seuil à partir duquel le % sera affiché sur le graphe pour chaque modalité (>100 pour ne pas afficher les %)
        - ascending (bool) : OPTIONNEL - Affichage par ordre croissant (True), décroissant (False) ou pas de tri (None)
        - axe (plt.Axes) : OPTIONNEL - Affichage dans cet axe, création d'un nouveau sinon

        Return :
        - fig, axe : fig <-- None si utilisation de l'axe transmis en paramètre
    
    """
    fig = None
    if axe is None :
        fig, axe = plt.subplots(1, 1, figsize=(15, 5))
        
    # Passe les colonnes en str pour éviter un plantage sur startwith
    if hue is None :
        df_str = df[[x]].astype(str)  

    else :
        df_str = df[[x, hue]].astype(str)  
    
    # Ordre d'affichage
    ordre = None
    if ascending is not None: 
        ordre = df_str[x].value_counts(ascending=ascending).index

    sns.countplot(x=x, hue=hue, data=df_str, ax=axe, order=ordre)       # plutot que x=x pour forcer le type à str (sinon plantage startwith dans countplot

    # Calcul des % de chaque modalité 
    total = len(df_str[x])
    for p in axe.patches:
        percent = 100 * p.get_height() / total
        if viewPercent < percent :
            percentage = '{:.1f}%'.format(percent)
            X = p.get_x() + p.get_width() / 2 - 0.1
            Y = p.get_height() + 0.1
            axe.text(X, Y, percentage, ha = 'center')

    # Libellé axe des x
    xticks_labels = axe.get_xticklabels()
    labels = [label.get_text() for label in xticks_labels]
    new_xlabels = [code_to_xlabel[label] if label in code_to_xlabel else label for label in labels]
    axe.set_xticks(ticks = range(len(new_xlabels)), labels = new_xlabels, rotation=rotation_xlabel);
    axe.set_title(title, fontsize = 12)

    # Libellé légende (si hue est défini)
    if hue is not None:
        handles, labels = axe.get_legend_handles_labels()
        new_huelabels = [code_to_huelabel[label] if label in code_to_huelabel else label for label in labels]
        axe.legend(handles, new_huelabels)

    st.pyplot(fig)

    return fig, axe
   
def monhistplot (df: pd.DataFrame, 
                 x: str, 
                 y: str=None,
                 title: str=None, 
                 bins: int=50, 
                 viewPercent: float=0.0,
                 axe:axes._axes.Axes=None)-> list:
    """ Affiche un histplot personnalisé :
        - df : DataFrame contenant les données à afficher
        - x (str) : nom de la variable à afficher sur l'axe des abscisses 
        - y (str) : nom de la variable à afficher sur l'axe des ordonnés 
        - title (str) : Optionel - titre à afficher
        - bins (int) : nombre d'intervalles de regroupement par défaut 50
        - viewPercent (float) : seuil à partir duquel le % sera affiché sur le graphe pour chaque modalité (>100 pour ne pas afficher les %)
        - axe (matplotlib.axes._axes.Axes) : Optionel - Affichage dans cet axe, création d'un nouveau sinon

        Return :
        - fig, axe : fig <-- None si utilisation de l'axe transmis en paramètre
    
    """
    fig = None
    if axe is None :
        fig, axe = plt.subplots(1, 1, figsize=(15, 5))
        
    sns.histplot(x=x, y=y, data = df, bins=bins, ax=axe)

    total = len(df[x])
    for p in axe.patches:
        percent = 100 * p.get_height() / total
        if viewPercent < percent:
            percentage = '{:.1f}%'.format(percent)
            X = p.get_x() + p.get_width() / 2 - 0.1
            Y = p.get_height() + 0.1
            axe.text(X, Y, percentage, ha = 'center')
    
    axe.set_title(title, fontsize = 12)

    st.pyplot(fig)

    return fig, axe

# Fonction d'analyse d'une variable catégorielle
def analyse_variable_categorielle (df: pd.DataFrame, 
                                   x: str, 
                                   cible: str='user_gravite', 
                                   modalite_cible: any=3, 
                                   liste_taux: list=np.arange(0.0, 5.0, 0.1),
                                   valna: any=None, 
                                   # attributs pour la gestion du graphe de distribution (countplot)
                                   title: str=None,
                                   code_to_xlabel: dict={'0' : 'Indemne', '1' : 'Blessé léger', '2' : 'Blessé grave', '3' : 'Tué'}, 
                                   code_to_huelabel: dict=None,
                                   rotation_xlabel: int=0,
                                   viewPercent: float=0.25,
                                   axe: plt.Axes=None,
                                   limiteModalite: int=999) :
    """ Fonction permettant de faire une analyse rapide de la répartition de chaque modalité d'une variable ainsi que de ses Missing Values.

        Des tests statistiques permettent de définir le niveau de corrélation de cette variable avec la varible cible et d'identifier
        s'il y  a un intérêt de privilégier à cette variable :
           - un regroupement binaire (le taux de regroupement sera recherché afin de maximiser le V Cramer) : (cf.regroupement_binaire_variable_categorielle())
           - un OneHotEncoding
           
        ARGS :   
        - df (DataFrame) : Encsemble de données à analyser
        - x (str) : nom de la variable catégorielle à analyser
        - cible (str) : variable cible pour affichage Tableau de contingence
        - modalite_cible (any) = OPTIONNEL - modalité de la variable cible  privilégier (par défaut : 3=Tué)
        - liste_taux (float) : OPTIONNEL - % de répartition à atteindre par la modalité pour être regroupé en modalité 1 sinon modalité 0
        - valna (any) : OPTIONNEL - Catégorie à considérer comme valeurs manquantes (en complément des nan)
        - title (str) : OPTIONNEL - Titre du graphe
        - code_to_xlabel ({}) : OPTIONNEL - Dictionnaire modalité --> libellé (si fourni, les countplot sont affichés)
        - code_to_huelabel ({}) : OPTIONNEL - Dictionnaire modalité --> libellé pour la variable à analyser (x)
        - rotation_xlabel (int) : OPTIONNEL - Angle de rotation des libellés de l'axe des absisses
        - viewPercent (float) : OPTIONNEL - seuil à partir duquel le % sera affiché sur le graphe pour chaque modalité (>100 pour ne pas afficher les %)
        - axe (plt.Axes) : OPTIONNEL - Affichage dans cet axe, création d'un nouveau sinon
        - limiteModalite (int) : OPTIONNEL - Affichage des 'limiteModalite' premières Modalités les plus utilisées

        RETURN : best_regroupement si calculé, df[x] sinon
    """
    print()
    nb_na = df[x].isna().sum()
    if valna is not None :
        nb_na += df[(df[x] == valna)][x].count()

    # Répartition des différentes modalités
    pourcentage = round(nb_na / len(df), 2) * 100
    print(f"Il y a {pourcentage}% ({nb_na}) de données manquantes (ou assimilées).", end="\n")
    nombre = df[x].nunique()
    print(f"Il y a {nombre} modalités.", end="\n\n")
    print(df[x].value_counts(normalize=True, dropna=False).head(limiteModalite), end="\n\n")

    if cible is not None :
        # Test Statistique du chi2
        ctnorm = pd.crosstab(df[x], df[cible], normalize='index')
        ct = pd.crosstab(df[x], df[cible])
        chi2, p_value_noRgmt, dof, expected = chi2_contingency(ct)
        V_Cramer_noRgmt = np.sqrt(chi2/pd.crosstab(df[x], df[cible]).values.sum())
        
        print("Test Statistique du chi2 (sans regroupement) : p-value=%.3f" % p_value_noRgmt, "; V de Cramer=%.5f" % V_Cramer_noRgmt, end="\n\n")
        print ("Tableau de contingence entre la variable et la variable cible (pour chi2): \n\n")
        print (ct.head(limiteModalite), end="\n\n")
        
        # Ajout de 2 graphes : distribution d ela variable analysée à gauche, =f(cible) à droite
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        axe = axs[0]
        if title is None :
            title = f"Analyse de la variable {x}"
            
        if  (code_to_huelabel is not None and len(code_to_huelabel)>0) :
            # Affichage d'un countPlot de la variable
            moncountplot(df, 
                         x=x, 
                         title=title, 
                         code_to_xlabel=code_to_huelabel, 
                         rotation_xlabel=rotation_xlabel,
                         viewPercent=viewPercent,
                         axe=axe
                    )

            axe = axs[1]
            moncountplot(df, 
                         x=cible, 
                         hue=x, 
                         title="en fonction de la variable cible", 
                         code_to_xlabel=code_to_xlabel, 
                         code_to_huelabel=code_to_huelabel,
                         rotation_xlabel=rotation_xlabel,
                         viewPercent=viewPercent,
                         axe=axe
                    )
            plt.show();

        # 2 autres graphe type Boxplot
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        axe = axs[0]
        monboxplot(df,
                   x=x,
                   horizontal=True,
                   axe=axe
                  )
            
        axe = axs[1]
        monboxplot(df,
                   x=cible,
                   y=x,
                   axe=axe
                  )                       

        plt.show();
            
        # Recherche d'un regroupement binaire maximisant la corrélation avec la variable cible
        best_optimisation = 'BINAIRE'
        _modalite_1 = "Liste des catégories regroupée sur la modalité 1 : "
        _modalite_0 = f"Liste des modalités regroupées sur la modalité 0 :"
        print ("TENTATIVE OPTIMISATION PAR REGROUPEMENT BINAIRE")

        # Tableau de contingence normalisé
        print ("Tableau de contingence normalisé entre la variable et la variable cible : \n\n")
        print (ctnorm.head(limiteModalite), end="\n\n")
        best_regroupement, best_taux, best_p_value, best_V_Cramer, best_modalite_1, best_modalite_0, _ = regroupement_variable_categorielle (
                                                                      df, 
                                                                      x=x, 
                                                                      cible=cible, 
                                                                      modalite_cible=modalite_cible, 
                                                                      binary = True,                  # regroupement en catégorie binaire
                                                                      liste_taux=liste_taux, 
                                                                      code_to_xlabel=code_to_xlabel)
        
        # Recherche d'un regroupement non binaire maximisant la corrélation avec la variable cible
        print ("TENTATIVE OPTIMISATION PAR REGROUPEMENT NON BINAIRE")
        best_regroupement_Rgmt, best_taux_Rgmt, best_p_value_Rgmt, best_V_Cramer_Rgmt, best_modalite_1_Rgmt, best_modalite_0_Rgmt, new_modalite = regroupement_variable_categorielle (
                                                                    df, 
                                                                    x=x, 
                                                                    cible=cible, 
                                                                    modalite_cible=modalite_cible, 
                                                                    binary=False,
                                                                    liste_taux=liste_taux, 
                                                                    code_to_xlabel=code_to_xlabel)
        
        # Conservation du meilleur regroupement identifié
        if best_V_Cramer_Rgmt > best_V_Cramer :
            best_regroupement = best_regroupement_Rgmt
            best_taux = best_taux_Rgmt
            best_p_value = best_p_value_Rgmt
            best_V_Cramer = best_V_Cramer_Rgmt
            best_modalite_1 = best_modalite_1_Rgmt
            best_modalite_0 = best_modalite_0_Rgmt
            best_optimisation = 'NON BINAIRE'
            _modalite_1 = "Liste des catégories conservées : "
            _modalite_0 = f"Liste des modalités regroupées pour former la nouvelle modalité '{new_modalite}' :"

        print (f"\nOptimisation Corrélation variable '{x}':")
        if best_V_Cramer > V_Cramer_noRgmt :
            print (f"OPTIMISATION REGROUPEMENT {best_optimisation} : taux de regroupement optimal={best_taux} " \
                   f", best_p_value=%.3f" % best_p_value, " - best_V_Cramer=%.5f" % best_V_Cramer)
            print (_modalite_1, best_modalite_1)
            print (_modalite_0, best_modalite_0)

            return best_regroupement

    # La variable sans regroupement est plus performante...
    print (" ==>> AUCUNE OPTIMISATION IDENTIFIEE PAR TRANSFORMATION DU REGROUPEMENT DE LA VARIBLE.")
    
    return df[x]

# Permet de regouper en 2 modalités une variable catégorielle selon le taux de la modalité tué qui nous intéresse
def regroupement_variable_categorielle(df: pd.DataFrame, 
                                       x: str, 
                                       cible: str='user_gravite', 
                                       modalite_cible: any=3, 
                                       binary: bool=True,
                                       liste_taux: list=np.arange(2, 5, 0.1), 
                                       code_to_xlabel: dict={'0' : 'Indemne', '1' : 'Blessé léger', '2' : 'Blessé grave', '3' : 'Tué'}) :
    """ Fonction permettant de regrouper les catégories afin de ne conserver que certaines modalités. 2 Modes de regroupement possible :
        - df (DataFrame) : Encsemble de données à analyser
        - x (str) : nom de la variable catégorielle à analyser
        - cible (str) : OPTIONNAL - variable cible pour affichage Tableau de contingence
        - modalite_cible (any) = OPTIONNAL - modalité de la variable cible  privilégier (par défaut : 3=Tué)
        - binary (bool) = OPTIONNAL - True  --> catégorisation binaire : 2 modalités de regroupement seront réalisée : les modalités qui atteingnent le seuil et les autres
                                    - False --> Regroupement sous un nouveau code de toutes les modalités dont la proba d'atteindre la modalité cible de référence n'est pas atteinte
        - liste_taux (float) : OPTIONAL - % de répartition à atteindre par la modalité pour être regroupé en modalité 1 sinon modalité 0
        - code_to_xlabel ({}) : OPTIONNAL - Dictionnaire modalité --> libellé (si fourni, les countplot sont affichés)

        RETURN : np contenant le nouveau regroupement
    """
    print()
    ctnorm = pd.crosstab(df[x], df[cible], normalize='index')

    if binary == False :
        # Recherche de la modalité la plus grande pour la variable en cours d'analyse
        new_modalite = df[x].unique().max().astype(int) + 1
        print ("REGROUPEMENT NON BINAIRE : new_modalite=", new_modalite)
        
    else :
        new_modalite = None
        
    p_values = []
    V_Cramers = []
    best_V_Cramer = -1 
    best_p_value = -1
    best_taux = 0
    best_modalite_0 = []
    best_modalite_1 = []
    best_regroupement = []
    for taux in liste_taux :
        if binary :
            regroupement = np.where(df[x].isin(ctnorm[ctnorm[modalite_cible] >= taux/100].index.tolist()), 1, 0)

        else :
            regroupement = np.where(df[x].isin(ctnorm[ctnorm[modalite_cible] >= taux/100].index.tolist()), df[x], new_modalite)
            
        # Test statistique à partir de ce regroupement
        stat, p_value, _, _ = chi2_contingency(pd.crosstab(regroupement, df[cible]))
        V_Cramer = np.sqrt(stat/pd.crosstab(regroupement, df[cible]).values.sum())
        
        #print("TAUX=", taux, " - Test Statistique du chi2 : p-value=%.3f" % p_value, " ; V de Cramer=%.5f" %V_Cramer)
        
        if p_value > 0.05 :
            p_values.append(0.05)
            V_Cramers.append(0)
        
        else :
            p_values.append(p_value)
            V_Cramers.append(V_Cramer)

        if (p_value < 0.05) & (best_V_Cramer <= V_Cramer) :
            # Si égalité entre best_model et current_modele, on ne garde le best_modele que si binary == False afin de garder le plus grand taux 
            if (best_V_Cramer == V_Cramer) & binary == True :
                continue
            else :
                best_taux = taux
                best_p_value = p_value
                best_V_Cramer = V_Cramer
                best_modalite_0 = ctnorm[ctnorm[modalite_cible] < taux/100].index.tolist()
                best_modalite_1 = ctnorm[ctnorm[modalite_cible] >= taux/100].index.tolist()
                best_regroupement = regroupement
        
    print(f"Meilleurs résultats obtenu avec taux=", best_taux)
    if binary == False :
        print(f"Liste des modalités regroupées dans la nouvelle catégorie {new_modalite} :", best_modalite_0)

    else :
        print("Liste des modalités retenues en modalité de regroupement 1 :", best_modalite_1)
    
    print("Test Statistique du chi2 : p-value=%.3f" % best_p_value, " ; V de Cramer=%.5f" % best_V_Cramer, end="\n\n")

    # Résultat de la maximisation de la corrélation entre la variable et la variable cible
    plt.plot(np.array(liste_taux), V_Cramers, color='blue', label='V Cramer')
    plt.plot(np.array(liste_taux), p_values, color='orange', marker='o', label='p_value')

    plt.xlabel('Taux')
    plt.ylabel('p-value')
    plt.title('Comparaison des valeurs de p-value et de V Cramer pour différents taux de regroupement')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show();

    if best_V_Cramer == -1 :
        best_regroupement = df[x]
        
    return best_regroupement, best_taux, best_p_value, best_V_Cramer, best_modalite_1, best_modalite_0, new_modalite
                                       
# Fonction d'analyse d'une variable catégorielle binaire
def analyse_variable_binaire(df: pd.DataFrame, x: str, cible: str=None, valna: any=None, code_to_xlabel: dict=None) :
    """ Fonction permettant de faire une analyse rapide de la répartition de chaque modalité ainsi que des Missing Values
        - df (DataFrame) : Encsemble de données à analyser
        - x (str) : nom de la variable catégorielle à analyser
        - cible (str) : OPTIONAL - variable cible pour affichage Tableau de contingence
        - valna (any) : OPTIONAL - Catégorie à considérer comme valeurs manquantes (en complément des nan)
        - code_to_xlabel ({}) : OPTIONAL - Dictionnaire modalité --> libellé (si fourni, les countplot sont affichés)
    """
    print()
    nb_na = df[x].isna().sum()
    if valna is not None :
        nb_na += df[(df[x] == valna)][x].count()
    
    # Répartition des différentes modalités
    pourcentage = round(nb_na / len(df), 3) * 100
    print(f"Il y a {pourcentage}% de données manquantes ou assimilées ({nb_na}).", end="\n\n")
    nombre = df[x].nunique()
    print(f"Répartition de la variable :", end="\n\n")
    print(df[x].value_counts(normalize=True, dropna=False), end="\n\n")

    # Graphe de répartition de la variable
    if code_to_xlabel :
        moncountplot(df, 
                     x, 
                     title=f"Répartition de la variable '{x}'", 
                     code_to_xlabel=code_to_xlabel, 
                    )
    
    # Tableau de contingence avec la variable cible 
    if cible is not None :
        # Test Statistique du chi2
        ctnorm = pd.crosstab(df[x], df[cible], normalize='index')
        ct = pd.crosstab(df[x], df[cible])
        chi2, p_value, dof, expected = chi2_contingency(ct)
        V_Cramer = np.sqrt(chi2/pd.crosstab(df[cible], df[x]).values.sum())
        
        print("Test Statistique du chi2 : p-value=%.3F" % p_value, " ; V de Cramer=", V_Cramer, end="\n\n")

        print ("Tableau de contingence entre la variable et la variable cible (pour chi2): \n\n")
        print (ct, end="\n\n")
        
        print ("Tableau de contingence normalisé entre la variable et la variable cible : \n\n")
        print (ctnorm, end="\n\n")
        
        
        if code_to_xlabel :
            code_to_huelabel = {'0' : 'Indemne', '1' : 'Blessé léger', '2' : 'Blessé grave', '3' : 'Tué'}
            moncountplot(df, 
                     x, 
                     cible, 
                     f"Analyse variable '{x}'", 
                     code_to_xlabel=code_to_xlabel, 
                     code_to_huelabel=code_to_huelabel
                    )
    
# Fonction d'analyse d'une variable quantitative
def analyse_variable_quantitative (df: pd.DataFrame, 
                                   x: str, 
                                   cible: str=None, 
                                   valna: any=None, 
                                   code_to_xlabel: dict=None, 
                                   rotation_xlabel: int=0,
                                   viewPercent: float=0.25,
                                   viewOutliers: bool=True,
                                   bins:int=10) :
    """ Fonction permettant de faire une analyse rapide de la distribution de chaque modalité ainsi que des Missing Values
        
        Affichage possible de boxplot
        
        - df (DataFrame) : Encsemble de données à analyser
        - x (str) : nom de la variable catégorielle à analyser
        - cible (str) : OPTIONNEL - variable cible pour affichage Tableau de contingence
        - valna (any) : OPTIONNEL - Catégorie à considérer comme valeurs manquantes (en complément des nan)
        - code_to_xlabel ({}) : OPTIONNEL - Dictionnaire modalités cible --> libellé (si fourni, les boxplots sont affichés)
        - rotation_xlabel (int) : OPTIONNEL - Orientation label axe des x
        - viewPercent (float) : OPTIONNEL - Ajout du % sur les graphes type Histplot si valeur Histo > viewPercent 
        - viewOutliers (bool) : OPTIONNEL - Détail des outliers affiché O/N sur BoxPlot
        - bins (int) : OPTIONNEL - Nombre d'histo
        
    """
    print()
    nb_na = df[x].isna().sum()
    nb_valna = 0
    if valna is not None :
        nb_valna = df[(df[x] == valna)][x].count()
        
    nb_na += nb_valna
    
    # Répartition des différentes modalités
    pourcentage = round(nb_na / len(df), 3) * 100
    print(f"Il y a {pourcentage}% ({nb_na}) de données manquantes (ou assimilées).", end="\n\n")
    if nb_valna > 0 :
        print("Statistiques de la variable (hors valeur nan): ", end="\n\n")
        print(df[(df[x] != valna)][x].describe(), end="\n\n")
        df_without_valna = df[df[x] != valna]
        
    else :
        print("Statistiques de la variable : ", end="\n\n")
        print(df[x].describe(), end="\n\n")

    if code_to_xlabel is not None :
        # Affichage de la distribution de la variable sous forme d'histogramme    
        fig, axs = monhistplot(df, x=x, title=f"Distribution variable {x}", bins = bins, viewPercent=viewPercent)
        
        if nb_valna > 0 :
            fig, axs = monhistplot(df_without_valna, x=x, title=f"Distribution variable {x} (sans valna={valna})", bins = bins, viewPercent=viewPercent)

        plt.show();
        
    if cible is not None :
        # Analyse statistique - test Anova (entre année de naissance (quantitative) de l'usager et la variable cible ='user_gravite' (catégorielle)
        result = statsmodels.formula.api.ols(f'{cible} ~ {x}', data=df).fit()
        table = statsmodels.api.stats.anova_lm(result)
        if nb_valna > 0 :
            print ("\nTest statistique ANOVA (avec valna)- PR(>F)=%.5f" %table.iloc[0, 4], end="\n\n")
            result = statsmodels.formula.api.ols(f'{cible} ~ {x}', data=df_without_valna).fit()
            table = statsmodels.api.stats.anova_lm(result)
            print ("\nTest statistique ANOVA (sans valna)- PR(>F)=%.5f" %table.iloc[0, 4], end="\n\n")
            
        else :
            print ("\nTest statistique ANOVA - PR(>F)=%.5f" %table.iloc[0, 4], end="\n\n")
            
        st.write(table)

        # Matrice de corrélation entre variable et variable cible
        matrice = df[[x, cible]].corr()
        st.write ("Matrice de corrélation : ", "(coef=%.3f)" %matrice[x][cible])
        st.write(matrice)

        # Affichage de la distribution de la variable sous forme de BoxPloten fonction de la variable cible
        if nb_valna > 0 :
            monboxplot(df_without_valna, x= cible, y=x, title=f"Distribution de la variable (sans les NaN)=f({cible})", code_to_xlabel=code_to_xlabel, rotation_xlabel=rotation_xlabel, viewOutliers=viewOutliers)

        else :
            monboxplot(df, x= cible, y=x, title=f"Distribution de la variable =f({cible})", code_to_xlabel=code_to_xlabel, rotation_xlabel=rotation_xlabel, viewOutliers=viewOutliers)
            
        plt.show();
        st.pyplot(fig)

# Fonction d'analyse des missing value pour une variable définie
def analyse_valeurs_manquantes (df: pd.DataFrame, x: str, variables_continues: list=[], variables_categorielles : list=[]) :
    """ Fonction qui permet d'analyser la tendance d'une variables par rapport à un périmètre de variables continues 
        d'une part et catégorielles d'autre part afin d'identifier des tendances

        L'objectif étant d'identifier des règles de remplacement des missing value peut être plus pertinente qu'un remplacement par défaut

        - df (DataFrame) : Ensemble des données à analyser
        - x (str) : nom de la variable à analyser
        - variables_continues ([]) : Liste des variables continues à analyser
        - variables_categorielles ([]) : Liste des variables catégorielles à analyser
        
    """

    # Variables continues
    liste_variables = variables_continues
    liste_variables.append(x)
    print("Liste=", liste_variables) 
    print (f"La valeur médiane selon {x} : ", end="\n\n")
    print (df[liste_variables].groupby(x).median(), end="\n\n")

    # Variables catégorielles
    liste_variables = variables_categorielles
    liste_variables.append(x)
    print (f"Le mode selon {x} : ", end="\n\n")
    print(df[liste_variables].groupby(x).apply(pd.DataFrame.mode).set_index(x), end="\n\n")

    # Vérification des proportions
    for variable in variables_categorielles :
        print("Proportion : ", end="\n\n")
        print(df.groupby(variable)[x].value_counts(normalize=True), end="\n\n")
                        
