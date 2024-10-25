import streamlit as st
import streamlit_antd_components as sac


# ---------
# 1. PROJET
# ---------
LE_PROJET_HEADER = '''
    Ce projet a été réalisé dans le cadre de la formation de Data Science via l'organisme Datascientest.
    '''
LE_PROJET_INTRODUCTION = '''<p style="text-align:justify;">                    
            L'objectif est de prédire les risques d'accidents de la route à partir de la 
            base de donnée nationale mise à disposition par le gouvernement Français.
            <ul>
                <li> Identification de zone à risque</li>
                <li> Identification de caractéristiques agravantes</li>
                <li> Prédiction des risques d'accidents grave</li>
            </ul><br>
            Ce streamlit présente notre démarche pour mener à bien ce projet, de la découverte des données
            (exploration des données, consolidation, data cleaning) à la phase de modélisation (modèles 
            simples de machine learning, modèles de réseaux de neurones denses, interprétabilité).<br>
            <br>
            Il reprend les 2 parties qui ont composées ce projet pour terminer sur une synthèse des résultats 
            obtenus et des axes d'amélioration.<br>
        '''
LE_PROJET_EQUIPE = '''<p style="text-align:justify;">
<b><a href=https://www.linkedin.com/in/gabriel-del-vecchio-5b9ba5155/>Gabriel Del Vecchio</a></b>
<br>
Commercial dans le domaine de la data, je travaille avec des outils analytics au quotidien. Bien que la 
problématique des accidents routiers et la méthodologie d’un projet de Machine Learning soient nouvelles 
pour moi, je suis enthousiaste à l'idée de relever ce défi. Mon expérience avec l'analyse des données me 
donne une bonne base pour aborder ce projet de Data Science, en particulier sur un jeu de données de grande 
dimension. Je suis convaincu que cette opportunité me permettra d'acquérir de nouvelles compétences précieuses 
et d'apporter une valeur ajoutée à notre entreprise.<br>
<br>
<b><a href=https://www.linkedin.com/in/sophie-doublier/>Sophie Doublier</a></b>
<br> 
En tant que chercheuse en biologie, la problématique des accidents routiers ne m’est pas familière. 
Bien qu’habituée à analyser des données, j’apprends pratiquement depuis zéro à mener à bien un projet de 
Data Science et, plus particulièrement, la méthodologie d’un projet de Machine Learning concernant un jeu 
de données de grande dimension. Mon objectif est de monter en compétence par l’acquisition de nouvelles 
méthodes de travail et ainsi d’adapter mon bagage professionnel au monde du travail d’aujourd’hui et de demain. 
Je n’ai pas connaissance de projet similaire au sein de mon entreprise.<br>
<br>
<b><a href=https://www.linkedin.com/in/st%C3%A9phane--martinez/>Stéphane Martinez</a></b>
<br>
Chef de projet en informatique, mon expérience professionnelle et mon appétence pour les sujets techniques 
devraient être un plus pour ce projet. Il n’en demeure pas moins que d’une part, les spécificités autour 
de la data Science restent nouvelles pour moi et d’autre part, je suis habitué dans mon domaine à répondre 
à des problématiques clairement définies, ce qui n’est pas le cas ici.<br>
<br>
<b><a href=](https://www.linkedin.com/in/marcel-njapo-tchakounte-93926265/>Marcel Njapo</a></b>
<br>
Consultant technique en système d'information plus particulièrement dans le développement d'applications web 
métier dans le secteur de la finance, Je suis amené à manipuler des chiffres et vérifier la cohérence de ceux-ci. 
La problématique de la data science sur les accidents de la route bien que nouvelle pour moi, je pense pouvoir 
apporter mes compétences en développement d’application tout en mettant en pratique les bonnes méthodes de data 
science que nous voyions actuellement dans notre formation.<br>
'''

# --------------------------
# 2. EXPLORATION DES DONNEES
# --------------------------
# PAGE 1 - PRESENTATION
# ---------------------
PRESENTATION_HEADER = '''<p style="text-align:justify;">
            L'étude s'est basée sur le fichier national des accidents corporels de la circulation dit «<strong>Fichier BAAC</strong> » administré par l’Observatoire national interministériel de la sécurité routière <strong>"ONISR"</strong>.
            Elles sont accessibles à l'adresse <a href=https://www.data.gouv.fr/fr/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2022/>data.gouv.fr</a>.
            </p>
            '''
# ONGLET 1 - Règles de gestion
PRESENTATION_REGLE_GESTION = '''<p style="text-align:justify;">                    
            Pour chaque accident corporel, des saisies d’information décrivant l’accident sont effectuées par 
            l’unité des forces de l’ordre (police, gendarmerie, etc.) qui est intervenue sur le lieu de 
            l’accident.
            <br>
            <br>
            Ces saisies sont rassemblées dans une fiche intitulée bulletin d’analyse des accidents corporels. 
            L’ensemble de ces fiches constitue le fichier national des accidents corporels de la circulation 
            dit « <strong>Fichier BAAC</strong> » administré par l’Observatoire national interministériel de 
            la sécurité routière "<strong>ONISR</strong>".
            <br>
            <br>
            <u><strong>Est considéré comme accident corporel :</strong></u>
            <ul>
                    <li>Un accident survenu sur une voie ouverte à la circulation publique,</li>
                    <li>Impliquant au moins un véhicule,</li>
                    <li>Ayant fait au moins une victime ayant nécessité des soins,</li>
            </ul>
            <br>
            <u><strong>Profondeur d'historique :</u></strong> 
            <br>       
            Sont disponibles les données de 2005 à 2022. 
            <br>
            <br>
            <u><strong>Evolution de la méthodologie de collecte des données au fil du temps :</u></strong>  
            Le site est très bien documenté et référence l'exhaustivité des modifications apportées aux règles de collecte
            des données qui impose une <u><b>grande prudence</b></u> lors de leur utilisation.
            <br>
            <br>
            Nous avons bien évidemment tenu compte de ces évolutions afin de définir le périmètre de données qui nous 
            servirait de base de travail. L'ensemble des informations utilisées est disponible dans le document fourni
            par l'ONISR <a href=https://www.data.gouv.fr/fr/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2022/#/resources/8ef4c2a3-91a0-4d98-ae3a-989bde87b62a>ici</a>.    
            <br>       
            <br>               
            '''

PRESENTATION_REGLE_GESTION_ALERT_1 = '''
A partir de 2018, les règles de classification des personnes hospitalisée a changé ce qui nous interdit de 
réaliser des comparatifs de cette variable entre les périodes antérieures et postérieures à 2018'''

# ONGLET 2 - FORMAT DES DONNEES
PRESENTATION_FORMAT_DONNEES_1 = '''<p style="text-align:justify;">
        Les données à disposition sont organisées sous 4 fichiers distincts pour chaque année d'historique :<br>
        <br>
        <u><strong>Modèle conceptuel des données :</strong></u>
        '''

PRESENTATION_FORMAT_DONNEES_2 ='''<p style="text-align:justify;">
        <u><strong>Détail des variables disponibles :</strong></u>
        '''

PRESENTATION_FORMAT_DONNEES_DETAIL_LIEU = '''<p style="text-align:justify;">Toutes les variables en lien avec le lieu de l'accident'''
PRESENTATION_FORMAT_DONNEES_DETAIL_CARAC = '''<p style="text-align:justify;">Toutes les variables en lien avec les caractéristiques de l'accident'''
PRESENTATION_FORMAT_DONNEES_DETAIL_VEHI = '''<p style="text-align:justify;">Toutes les variables en lien avec le véhivule de l'accident'''
PRESENTATION_FORMAT_DONNEES_DETAIL_USAGER = '''<p style="text-align:justify;">Toutes les variables en lien avec l'usager de l'accident'''

# ONGLET 3 - CONSOLIDATION DES DONNEES
PRESENTATION_MERGE_PREMIERE_ETAPE = '''<p style="text-align:justify;">
        Afin de simplifier l'accès aux données, nous décidons de consolider les données des 4 
        fichiers par année (soit un total de 72 fichiers de données brutes) en un seul dataFrame. 
        <br><br>
        <strong><u>Objectifs :</strong></u>
        <ul>
            <li>Simplifier l'accès aux données par la suite</li>
            <li>S'assurer de la cohérence des données (aucun enregistrement orphelin)</li>
            <li>Estimation de la qualité globale des données </li>
        </ul
        <br> <br>
        <strong><u>Première étape :</u></strong> Concaténation de tous les fichiers annuels de chaque type de caractéristique en un 
        seul.
        <ul> 
            <li> Nous avons dû harmoniser le type de chaque variable entre ces différents fichiers </li>
            <li> Intégrer les évolutions de règles de gestion au fil des ans afin d'avoir des données homogènes (ex. département multiplié par 10 certaines années)</li>
            <li> Contrôler l'absence de doublon une fois les données annuelles consolidées en un seul fichier</li>
        </ul>
        <br>
        '''
PRESENTATION_MERGE_SECONDE_ETAPE = '''<p style="text-align:justify;">
        <strong><u>Seconde étape :</u></strong> Merge des 4 fichiers ainsi obtenus en un seul. Pour ce faire,
        <ul> 
            <li> Merge des fichiers "lieu" et "caractéristique"  </li>
            <li> Merge avec le fichier véhicule</li>
            <li> Merge avec le fichier "Usager" </li>
        </ul> <br>
        '''

PRESENTATION_MERGE_ALERT_1 = "A ce stade nous nous interdisons de transformer les données hormis pour tenir compte de changement de règles sur la période d'historique."
PRESENTATION_MERGE_ALERT_2 = "Un contrôle de cohérence du volume de données a été réalisé afin de s'assurer de ne pas perdre de données."
PRESENTATION_MERGE_CODE_1 = '''
        # I.2. Merge des caractéristiques avec les lieux en un seul df
        df_merged = pd.merge(carac, lieux, on=['id_accident'])

        # Contrôle
        print ("nb d'enreg carac=", carac.shape) 
        print ("nb d'enreg lieux=", lieux.shape) 
        print ("nb d'enregistrement carac+lieux=", df_merged.shape) 
        '''
PRESENTATION_MERGE_RESULTAT_1 = '''
        nb d'enreg carac= (1176873, 16)<br>
        nb d'enreg lieux= (1176873, 19)<br>
        nb d'enregistrement carac+lieux= (1176873, 34) <br>
        '''
PRESENTATION_MERGE_CODE_2 = '''
        # I.3. Ajout des véhicules
        # Merge des caractéristiques avec les lieux en un seul df
        df_merged = pd.merge(df_merged, vehi, on=['id_accident'])

        # Contrôle
        print ("nb d'enreg vehicules=", vehi.shape) 
        print ("nb d'enregistrement carac+lieux+vehi=", df_merged.shape)
        '''
PRESENTATION_MERGE_RESULTAT_2 = '''
        nb d'enreg vehicules= (2009395, 11)<br>
        nb d'enregistrement carac+lieux+vehi= (2009395, 44)<br>
        '''
PRESENTATION_MERGE_CODE_3 = '''
        # I.4. Ajout des usagers
        accident = pd.merge(df_merged, usager, on=['id_accident', 'id_vehicule', 'num_veh'])

        # Contrôle
        print ("nb d'enregistrement carac+lieux+vehi=", df_merged.shape)
        print ("nb d'enreg usager=", usager.shape) 
        print ("nb d'enregistrement carac+lieux+vehi+usager=", accident.shape)
        '''
PRESENTATION_MERGE_CODE_4 = '''
        # III.4. Sauvegarde fichier usagers.csv et accidents.csv (aggréation de usagers)
        usager.to_csv(repDataConsolidees + "agg_usagers" + suffixe, index=False, encoding='utf-8')

        # Fichier accident aggrégé
        agg_accident.to_csv(repDataConsolidees + "agg_accidents" + suffixe, index=False, encoding='utf-8')
        '''

PRESENTATION_MERGE_RESULTAT_3 = '''
        nb d'enregistrement carac+lieux+vehi= (2009395, 44)<br>
        nb d'enreg usager= (2636377, 16) <br>
        nb d'enregistrement carac+lieux+vehi+usager= (2636355, 57)<br>
        '''
PRESENTATION_MERGE_TROISIEME_ETAPE = '''<p style="text-align:justify;">
        <strong><u>Troisième étape : </u></strong> Agrégation des données<br>   
        <Très rapidement nous nous sommes intérogés sur l'intérêt d'enrichir les données :
        - En ajoutant des données extérieures telles que les périodes de vacances scolaires
        - En aggrégeant des données déjà disponibles afin d'avoir des compléments d'information 
        au niveau de chaque enregistrement.
        <br>Si nous décidons à ce stade de ne pas enrichir les données avec les informations de vacances 
        scolaire (pour des raisons de temps), nous restons persuadé que cela pourrait apporter une amélioration 
        à nos résultats (axe d'amélioration noté pour la suite).
        <br> <br>
        Pour l'agrégation des données, nous décidons à cette étape de ne pas générer un mais deux dataFrames :
        <br>
        <ul>
            <li>un premier à la maille la plus fine à savoir l'utilisateur</li>
            <li>un second à la maille "Accident" avec en complément des données agrégées tel que :</li>
        </ul>
        '''
PRESENTATION_MERGE_LISTE_AGG = '''<p style="text-align:justify;"><ul>
        <li> Nb de vélo(s) concerné(s) par l'accident</li>
        <li> Nb de voiture(s) concerné(s) par l'accident</li>
        <li> Nb de véhicule(s) sans permis concerné(s) par l'accident</li>
        <li> Nb de moto(s) concerné(s) par l'accident</li>
        <li> Nb de poids lourd(s) concerné(s) par l'accident</li>
        <li> Nb de véhicule(s) autre(s) concerné(s) par l'accident</li>
        <li> Nb total de véhicule(s) concerné(s) par l'accident</li>
        <li> Nb de piéton(s) concerné(s) par l'accident</li>
        <li> Nb de passager(s) concerné(s) par l'accident</li>
        <li> Nb de conductrice(s) concernée(s) par l'accident</li>
        <li> Nb de conducteur(s) concerné(s) par l'accident</li>
        <li> Nb de passager(s) dans le véhicule</li>
        <li> Le genre du conducteur du véhicule</li>
        </ul>'''

# PAGE 2 - DATA VISUALISATION
# ---------------------------

# ONGLET 1 - METHODE
DATAVIZ_METHODE_INTRODUCTION = '''<p style="text-align:justify;">
            L'objectif de cette partie du projet est de s'approprier les données, d'estimer leur qualité 
            notamment par l'analyse des valeurs manquantes (missing value) et des valeurs aberrantes (outliers) 
            mais surtout d'estimer l'importance que chaque variable aura sur la résolution de notre 
            problématique.<br>
            <br>
            A la fin de cette étape nous devrions :<br> 
            <ul>
                <li> avoir identifié les variables à conserver pour la suite</li>
                <li> avoir une idée des techniques de gestion des valeurs manquantes et aberrantes à mettre en oeuvre
                afin de maximiser nos résultats</li>
            </ul>
            <br>
            Pour ce faire, nous avons : <br>
            <ol>
                <li> Défini un scope de données à conserver pour la suite (profondeur d'historique)
                <li> Analysé la variable cible</li>
                <li> identifié les variables techniques (identifiants) qu'il faudra exclure</li>
                <li> Analysé dans le détail chaque donnée :<br>
                    <ul>
                        <li> Variables temporelles</li>
                        <li> Catégorielles</li>
                        <li> Quantitatives</li>
                    </ul>
                <li> Défini les variables explicatives à conserver pour la suite</li>
            </ol>
            '''
DATAVIZ_METHODE_VARIABLE_CIBLE = '''<p style="text-align:justify;">
            La variable est une donnée catégorielle composée de 4 modalités ordonnées de 0 à 3 et d’une 
            modalité -1 qui correspond aux valeurs manquantes (215 enregistrements) qui correspondent 
            aux personnes en délit de fuite lors d’accident, personnes pour lesquels aucun suivi n’est possible.
            '''
DATAVIZ_METHODE_ALERT_CIBLE1 = "Ces enregistrements devront être écartés"
DATAVIZ_METHODE_ALERT_CIBLE2 = "On constate un déséquilibre de classes très important que nous devrons prendre en compte lors de la phase de modélisation"

DATAVIZ_METHODE_PERIMETRE_DATA = '''<p style="text-align:justify;">
            Si le volume a disposition peut faire sens pour des travaux analytiques des données, il ne 
            semble pas opportun de garder une telle volumétrie notamment dans l'utilisation de modèle 
            de machine learning.<br><br>
            Pour définir la profondeur d'historique à conserver pour la suite nous avons tenu compte : <br>
            <ul>
                <li> Des évolutions de règles de saisie pouvant rendre incompatibles certaines années</li>
                <li> Les ressources à notre disposition pour traiter ces données par la suite</li>
            </ul>    
            '''
DATAVIZ_ALERT_PROFONDEUR_DATA_1 = "A partir de 2019, nous avons une complétude des données qui est parfaite"
DATAVIZ_ALERT_PROFONDEUR_DATA_2 = "Pour la suite, nous privilégierons la période 2020-2022 soit 3 ans d'historique"

DATAVIZ_METHODE_VARIABLES_INUTILES = '''<p style="text-align:justify;">
            Les variables d'index risqueraient de biaiser le modèle, elles devront être exclues des variables 
            explicatives (id_accident, id_vehicule, num_veh).
            '''

DATAVIZ_METHODE_DIAGRAMME_DE_PAIRS = '''<p style="text-align:justify;">
            Permet d'explorer les relations entre plusieurs variables. Il peut être intéressant pour identifier des 
            regroupements ou des formes non linéaires, détecter des outlier, il permet également de visualiser 
            rapidement des distributions. <br>
            '''
DATAVIZ_METHODE_ALERT_PAIRS = "Dans notre cas, le nombre important, à la fois des variables explicatives et du nombre d'enregistrements ne nous à pas permis d'utiliser ce diagramme de manière optimale."

DATAVIZ_METHODE_ANALYSE_VARIABLE = '''<p style="text-align:justify;">
            Cette étape doit nous permettre d’identifier l'importance de chaque variable par rapport à notre 
            problématique. Nous en profiterons pour estimer les transformations qu'il serait 
            nécessaires de réaliser pour être exploitées dans un modèle de ML.<br>
            <br>
            Nous baserons nos décisions sur des tests statistiques afin d’estimer le niveau d’adhérence de chaque 
            variable avec notre variable cible.<br>
            <ul>
                <li>Un test de chi2 + Cramer pour les données catégorielles</li>
                <li>Un test Anova + coefficient de corrélation pour les données quantitative</li>
            </ul>
            <br>
            Ces tests seront exécutés :<br>
            <ul>
                <li> Sur les variables après remplacement des valeurs manquantes par la catégorie ‘-1’</li>
                <li> Sur les variables après remplacement des valeurs manquantes par la modalité la plus 
                fréquente.</li>
            </ul><br>
            L’analyse de ces résultats statistiques nous permettra de décider pour chaque variable :<br>
            <ul>
                <li> De l’exclure du scope des features (p-value > 0.05 ou indice de Cramer/coef de 
                corrélation trop faible</li>
                <li> De définir la règle de remplacement des valeurs manquantes : sera privilégiée la méthode 
                qui apporte une adhérence maximisée</li>
                <li> D'identifier les règles de remplacement/conservation des valeurs aberrantes</li>
            <ul><br>
            '''

DATAVIZ_METHODE_ANALYSE_OUTLIERS = '''<p style="text-align:justify;">
            Après analyse détaillée, nous pouvons largement supposer que les 7128 Outliers positionnés 
            au-dessus de l’interquantile Q3 et 125 en dessous de Q1 correspondent à des erreurs de saisies 
            que nous nous proposons de corriger comme suit :<br>
            <ul>
                <li> Les valeurs inférieures à 10 seront multipliées par 10</li>
                <li> Les valeurs supérieures à 130 seront divisées par 10</li>
            </ul><br>
            Ces règles permettent de corriger 99% des outliers. Il subsiste 15 enregistrements associés à une 
            vitesse de 130km/h et non associées à la catégorie autoroute.<br>
            <ul>
                <li> Si l’accident est hors agglo : on positionne la vitesse à 110 (on suppose une erreur 
                de saisie de 1 caractère 130 -> 110)</li>
                <li> Si l’accident est en agglo (c’est le cas de 4 accidents sur les 15) et après un 
                contrôle par géolocalisation qui a confirmé le bien fondée de la règle, la vitesse est forcée à 50>/li>
            <ul><br>
            Après correction de ces outliers, nous obtenons le graphe suivant qui contient toujours des Outliers 
            mais cohérents :
            '''

DATAVIZ_METHODE_ANALYSE_COLINEARITE = '''<p style="text-align:justify;">
            Représentation d'une matrice de corrélation, visuellement proposée sous forme d'un heatmap afin :<br>
            <ul>
                <li> De confirmer visuellement la corrélation entre variables explicatives et variables cible</li>
                <li> D'identfier de possible corrélation entre 2 variables explicatives qui n'auraient pas déjà été rejetées.</li>
            </ul>
            Nous avons privilégié dans ce cas celle ayant une meilleure corrélation avec la variable cible. L'autre sera écartée 
            afin d'éviter le phénomène de multi colinéarité.<br>
            '''
DATAVIZ_METHODE_ALERT_MULTICOLINEARITE = "Avant d'exclure une variable, nous aurions pu calculer le VIF afin de quantifier la multicolinéarité"
DATAVIZ_METHODE_SELECTION_EXPLICATIVE = '''<p style="text-align:justify;">
            Les analyses précédemment présentées ont été regroupées dans une fonction. Nous avons conservé les 
            résultats dans un dataframe contenant :
            <ul>
                <li> Le nom de la variable analysée</li>
                <li> Un indicateur "catégorielle" : O/N</li>
                <li> p_value (Chi2/Anova) selon la valeur de "catégorielle"</li>
                <li> Coef (indice de Cramer/coefficient de corrélation) selon l'indicateur "catégorielle"</li>
                <li> Un indicateur "MV_class" O/N (afin d'identifier la méthode de remplacement des missing value donnant les meilleurs résultats de tests statisitique</li>
            </ul><br>
            Ce dataframe a été travaillé de manière à ne conserver qu'une seule entrée par nom de variable 
            en privilégiant la valeur absolue du coeeficient de test statistique le plus élevé.
            '''
DATAVIZ_METHODE_ALERT_VARIABLE_EXPLICATIVE = '''<p style="text-align:justify;">
            Pour la suite, nous exclurons les variables ayant un indice de Cramer/coefficient de corrélation 
            compris dans l’intervalle [-0.1 :0.1]. Ce seuil est positionné « à priori ». Il nous permet de 
            conserver approximativement 50% des variables à disposition.
            '''
# ONGLET 2 - DATA'VIZ

# ONGLET 3 - TESTS STATISTIQUES

# ---------------
# 3. MODELISATION
# ---------------
# PREPROCESSING
# -------------

PREPROCESSING_PIPELINE_INTRODUCTION ='''<p style="text-align:justify;">
        Nous avons pris le parti de réaliser la phase de pré-processing dans un pipeline de manière :<br>
        <ul>
            <li> A automatiser cette opération sur l'ensemble de nos variables explicatives</li>
            <li> A pouvoir facilement modifier une règle de préprocessing sur tout ou partie de nos variables afin d'infirmer/confirmer nos choix initiaux</li>
        </ul><br>
        L'objectif étant de pouvoir utiliser un GridsearchCV afin :<br>
        <ul>
            <li> De pouvoir tuner nos hyperparamètres de pré-processing</li>
            <li> De pouvoir tuner l'algo de rééquilibrage (pour palier au problème de déséquilibrage de nos données)</li>
            <li> De pouvoir tuner l'algo de SelectKBest (ou de le désactiver)</li>
            <li> De pouvoir tester plusieurs modèles de Machine Learning Simple</li>
        </ul>       
        '''

PREPROCESSING_PIPELINE_PREPROCESS_VARIABLES  = '''<p style="text-align:justify;">
        <b><u> ETAPE 1 - Pré-processing de nos variables explicatives :</u></b>
        Nous avons écrit un transformer générique (sur la base des BaseEstimator, TransformerMixin afin de 
        pouvoir les utiliser dans un pipeline, capable de mettre en forme n'importe quelle variable. Il permet de :<br>
        <ul>
            <li> Gérer les valeurs manquantes </li>
            <li> Gérer les outliers</li>
            <li> D'activer ou non un encoder (dichotomisation pandas, OneHotEncoder)</li>
            <li> D'activer ou non un scaler (StandardScaler, MinMaxScaler)</li>
            <li> De conserver ou supprimer la variable</li>
        </ul>
        <br>
        Cette classe générique sera utilisée (via héritage) pour créer autant de classes de transformation 
        que de variables explicatives. Des valeurs par défaut seront définies pour chaque argument de manière 
        à accéder par défaut au comportement attendu pour chacune des variables.
        '''

PREPROCESSING_PIPELINE_EX_DEPT_CODE = '''
        # TRANSFORMER VARIABLE 'carac_dept'
        class Wcarac_dept (Transformer) :
            def __init__ (self, *, nan_strategy=None, fill_value=None, encoder:str='OneHotEncoder', scaler:str=None, dropC:bool=False) :
                super().__init__(columnName=self.__class__.__name__[1:], nan_strategy=nan_strategy, encoder=encoder, scaler=scaler, dropC=dropC, outliers=False)
        '''

PREPROCESSING_PIPELINE_EX_VMA_CODE = '''
    # TRANSFORMER VARIABLE 'lieu_vma'
    class Wlieu_vma(Transformer) :
        def __init__ (self, nan_strategy=None, fill_value:any=None, encoder:str=None, scaler:str='StandardScaler', outliers:bool=True, dropC:bool=True) :
            super().__init__(columnName=self.__class__.__name__[1:], nan_strategy=nan_strategy, fill_value=fill_value, encoder=encoder, scaler=scaler, dropC=dropC, outliers=outliers)

        def _outliers_transform(self, X:pd.DataFrame) -> pd.DataFrame:
            if self.outliers :
                # Gestion des Ouliers pour cette variable
                # REPRISE 1 
                ##   - Les valeurs < 10 seront multipliées par 10
                ##   - Les valeurs > 130 seront divisées par 10
                X[self.columnName] = X[self.columnName].apply(lambda valeur : valeur*10 if (valeur > 0) and (valeur < 10) else valeur/10 if valeur > 130 else valeur)

                # REPRISE 2 -  VMA = 130km/h en dehors d'une autoroute
                ## Force à 110 hors agglomération (supposition d'une erreur de saisie de 1 caractère
                ## Force à 50 en agglomération
                X[self.columnName] = X.apply(lambda row : 50 if (row[self.columnName]==130) and (row['lieu_catr'] != 1) and (row['carac_agg'] == 2) else row[self.columnName], axis=1)
            
            return X

        def _missing_values_fit (self, X:pd.DataFrame, y) :
            """ Surcharge Fonction missing values exécutée dans le fit
            Cette version permet de calculer les modalités les plus fréquentes selon la clé 'catr' + 'agg' pour les lignes <> -1.
            Ces valeurs seront utilisées dans _missing_value_transform() afin de remplacer les NaN.
            """
            self.mode_lieu_vma = X[X[self.columnName] != -1].groupby(['carac_agg', 'lieu_catr'])[self.columnName].agg(lambda x: x.value_counts().idxmax())

            return self
        
        def _missing_values_transform (self, X:pd.DataFrame) -> pd.DataFrame:
            """ Surcharge Fonction missing Values exécutée dans le transform """
            def replace_lieu_vma(row, mode_lieu_vma):
                if row[self.columnName] == -1:
                    return mode_lieu_vma.get((row['carac_agg'], row['lieu_catr']), row[self.columnName])
    
                return row[self.columnName]
        
            X[self.columnName] = X.apply(lambda row: replace_lieu_vma(row, self.mode_lieu_vma), axis=1)

            return X
        '''

PREPROCESSING_PIPELINE_FEATURE_ENGINEERING = '''<p style="text-align:justify;">
        <b><u> ETAPE 2 - Feature Engineering :</u></b><br>
        Nous allons enrichir nos données de deux informations :<br>
        <ul>
            <li> La date de l'accident construite à partir des jour, mois et année à notre disposition</li>
            <li> Une notion de Cluster construite à partir des coordonnées gps du lieu de l'accident</li>
        <ul>
        '''

PREPROCESSING_PIPELINE_FEATURE_ENGINEERING_CLUSTER_CODE = '''
    class WKMeans(BaseEstimator, TransformerMixin):
        def __init__(self, gpslat_name = 'carac_gps_lat', gpslong_name='carac_gps_long', n_clusters=20, inputsdropC=True, normalize=False, encoder=True, outputColumnName='zone_geographique'):
            self.n_clusters = n_clusters
            self.outputColumnName = outputColumnName
            self.columnName = outputColumnName                        # Compatibilité entre transformers pour utiliation dans pipeline  
            self.dropC = False                                        # Compatibilité entre transformers pour utiliation dans pipeline
            self.inputsdropC = inputsdropC                            # Permet de définir si les colonnes en entrée doivent être conservées ou supprimées (pas le même sens que dropC)
            self.normalize = normalize                                # Normalise les clusters obtenus O/N
            self.encoder = encoder                                    # OneHotEncode les clusters obtenus O/N
            self._initialized_encoder = None                          # instance de l'encoder
            self.gpslat_name = gpslat_name
            self.gpslong_name = gpslong_name
            self.kmeans = KMeans(n_clusters=self.n_clusters)

        def fit(self, X, y=None):
            self.kmeans.fit(X[[self.gpslat_name, self.gpslong_name]])

            # Initialiation de l'encoder si besoin
            if self.encoder == True:
                self._initialized_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore', drop='first')
                self._initialized_encoder.fit(np.arange(0, self.n_clusters).reshape(-1, 1) )    # Force les données en array 2D
                #self._initialized_encoder.fit(X[[self.columnName]])

            return self

        def transform(self, X):
            clusters = self.kmeans.predict(X[[self.gpslat_name, self.gpslong_name]])
            if self.inputsdropC == True :
                X = X.drop([self.gpslat_name, self.gpslong_name], axis = 1)

            if self.normalize :
                return np.c_[X, clusters/self.n_clusters]  # Concatène les clusters comme une nouvelle caractéristique (avec noramlisation des clusters

            elif self._initialized_encoder is not None : 
                # OneHotEncoder
                encoded_array = self._initialized_encoder.transform(clusters.reshape(-1, 1))
                encoded_df = pd.DataFrame(encoded_array, columns=self._initialized_encoder.get_feature_names_out([self.outputColumnName]))
                #X = X.drop(self.outputColumnName, axis=1)
                X = pd.concat([X.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

                return X

            return np.c_[X, clusters]  # Concatène les clusters comme une nouvelle caractéristique (avec noramlisation des clusters

        def get_feature_names_out(self, input_features=None):
            """ Fonction permettant de retourner le nom des colonnes après transformation
                (si utilisation de make_column_transformer les noms seront perdus)
            """
            nb_clusters = self.n_clusters
            cluster_column_names = [f"{self.outputColumnName}_{i}" for i in range(nb_clusters-1)]
            return cluster_column_names
'''

PREPROCESSING_PIPELINE_PREPROCESSING = '''<p style="text-align:justify;">
        <b><u> ETAPE 3 - Pipeline Pré-processing :</u></b>
        L'ensemble des transformers seront regroupés au sein d'un pipeline 'Pré-processing de la manière suivante :<br> 
        '''

PREPROCESSING_PIPELINE_PREPROCESSING_DETAIL = '''<p style="text-align:justify;">
        Chaque sous-pipeline ('Caractéristique', 'lieu', 'véhicule' et 'usager') regroupera l'ensemble des variables qui lui 
        sont associés.<br> 
        '''

PREPROCESSING_PIPELINE_TYPE_CARAC = '''
        CaracPipeline_mct = make_column_transformer (
            (Wcarac_agg(encoder='OneHotEncoder'), ['carac_agg']),
            (Wcarac_atm(), ['carac_atm']),
            (Wcarac_col(), ['carac_col']),
            (Wcarac_com(dropC=True), ['carac_com']),
            (Wcarac_dept(dropC=True), ['carac_dept']),
            (Wcarac_int(), ['carac_int']),
            (Wcarac_lum(), ['carac_lum']),
            (Wcarac_an(), ['carac_an']),
            (Wcarac_mois(), ['carac_mois']),
            (Wcarac_jour(), ['carac_jour'])
        ) '''

PREPROCESSING_PIPELINE_GLOBAL  = '''<p style="text-align:justify;">
    <b><u> ETAPE 4 - Finalisation du pipeline Global :</u></b><br>
    Il nous reste à enchainer à cette opération de pré-processing, l'ensemble des opérations nécessaires à 
    l'apprentissage d'un modèle.<br>
    <ul>
        <li> Rééquilibrage de l'échantillon d'entrainement</li>
        <li> Sélection des KBest meilleurs features</li>
        <li> L'apprentissage de notre premier modèle </li>
    <ul>

    '''

# MACHINE LEARNING
# ----------------
PREPROCESSING_METHODOLOGIE_TYPE_PROBLEME = '''<p style="text-align:justify;">
        Nous allons traiter un problème de classification déséquilibrée. Notre pipeline nous a permis de tester différentes
        méthodes de rééquilibrage tel que SMOTE, random under Sampling et random over Sampling.<br>
        <br>
        L'algorithme de SelectKBest n'ayant pas permis d'améliorer les résultats, nous avons abandonné son 
        utilisation assez rapidement.<br>
        <br>
        <u><b>Métriques de performance : </b></u><br>
        <ul>
            <li> Nous avons privilégié le score F1 qui correspond à la moyenne harminique de la précision et du 
            rappel et prend en compte à la fois la capacité du modèle à identifier correctement les positifs 
            (précision) et à ne pas manquer de vrais positifs (rappel). Ainsi, il assure un équilibre à notre 
            modèle</li>
            <li> Nous resterons vigilant au rappel des classes minoritaires. Dans notre problème, il est préférable de 
            prédire à tort un accident grave que de manquer un vrai positif</li>
            <li> Nous surveillerons l'évolution des métriques entre échantillons train et test afin de détecter du surapprentissage
            (overfitting)</li>
            <li> Analyse de la matrice de confusion afin d'identifier où se trompe le modèle</li>
        </ul>
        <br>
        <u><b> Découpage de notre échantillon de données : </b></u><br>
        <ul>
            <li> Train (75%) : 270 932 enregistrements</li>
            <li> Test (25%) : 90273 enregistrements</li>
        <ul><br>
        <u><b>Modèles de rééquilibrage utilisés : </b></u><br>
        <ul>
            <li> under sampling</li>
            <li> SMOTE</li>
        <ul>
        <br>
        <u><b>Modèles essayés : </b></u><br>
        <ul>
            <li> Régression logistique</li>
            <li> SVC</li>
            <li> Random Forest</li>
            <li> XGBoostClassifier</li>
            <li> AdaBoostClassifier</li>
            <li> Bagging (Décision Tree)</li>
        <ul><br>
        <br>
        '''
PREPROCESSING_METHODOLOGIE_PREMIERE_CONSTATATION = '''<p style="text-align:justify;">
        <ul>
            <li> Trois modèles de machine learning dit simples se dégagent légèrement avec des résultats assez similaires :
                <ul>
                    <li> SVC</li>
                    <li> XGBoost Classifier</li>
                    <li> Bagging (Decision Tree)</li>
                </ul>
            </li>
            <li> Nous pouvons constater l'absence de surapprentissage sur chacun des modèles.</li>
            <li> Légères différences entre les modèles de rééquilibrage : 
                <ul>
                    <li> L'utilisation de l'undersampling permet d'optenir de meilleurs rappels avec des scores F1 moins performants</li>
                    <li> Le SMOTE propose un équilibre meilleur entre rappel et score F1</li>
                </ul>
            </li>
        </ul>
         <b>Malgré cela, nos premiers résultats ne sont pas concluants. Les modèles peinent à dépasser des 
         scores de 50%.</b>
        '''

PREPROCESSING_METHODOLOGIE_AJUSTEMENT_PROBLEME = '''<p style="text-align:justify;">
        <u><b>Bascule en classification binaire : </u></b><br>
        Pour la suite, nous décidons de simplifier le problème en basculant sur une <u><b>classification binaire</u></b>. 
        <br>
        La variable cible sera modifiée comme suit :
        <ul>
            <li> Indemne / blessé léger</li>
            <li> blessé hospitalisé / tué </li>
        </ul>
        '''
ML_RESULTATS_BINAIRE = '''<p style="text-align:justify;">
        <u><b>Voici un résumé des performances : Précision du modèle :</u></b><br>
        <ul>
            <li> Données d'apprentissage : entre 68% et 75% </li>
            <li> Données de test : entre 67% et 73%</li>
        </ul>
        <br>
        <u><b> Nos analyses ont révélé deux modèles particulièrement performants :</u></b>
        <ul> 
            <li> XGBoostClassifier :
            <ul>
                <li> Score F1 : 0.5</li>
                <li> Précision sur la classe "Blessé grave/tué" : 82,81%</li>
            </ul>    
            </li>
            <li> Bagging :
            <ul>
                <li> Score F1 : 0.5</li>
                <li> Précision sur la classe "Blessé grave/tué" : 85,81%</li>
            </ul>
            </li>
        </ul>
        <br>
        <u><b>Analyse des performances :</u></b>
        <ul>
            <li> Points positifs :
            <ul>
                <li> Aucun modèle ne présente de surapprentissage, ce qui est encourageant.</li>
                <li> Le score F1 d'environ 50% est acceptable, bien qu'il laisse place à des améliorations.</li>
            </ul>
            </li>
            <li> Points d'attention :
            <ul>
                <li> Environ 20% des cas réels ne sont pas détectés (faux négatifs), ce qui est significatif dans notre contexte critique.</li>
            </ul>
            </li>
            <li> Prochaines étapes que nous envisageons : 
            <ul>
                <li> La technique SMOTE n'a pas apporté d'amélioration significative par rapport aux résultats précédents.</li>
                <li> Pour tenter d'améliorer les performances de notre modèle, nous allons donc explorer une approche basée sur <u>les réseaux de neurones.</u></li>
            </ul>
            </li>
        </ul>

        '''

# SIMULATEUR
# ----------
SIMULATEUR_CRITERE_CARACTERISTIQUE = '''<p style="text-align:justify;">
        <u><b>Critères 'Caractéristiques' :</u></b>
        '''
SIMULATEUR_CRITERE_LIEU = '''<p style="text-align:justify;">
        <u><b>Critères 'Lieux' :</u></b>
        '''
SIMULATEUR_CRITERE_USAGER = '''<p style="text-align:justify;">
        <u><b>Critères 'Usagers' :</u></b>
        '''
SIMULATEUR_CRITERE_VEHICULE = '''<p style="text-align:justify;">
        <u><b>Critères 'Caractéristiques' :</u></b>
        '''

# DEEP LEARNING
# -------------
DNN_INTRODUCTION = '''<p style="text-align:justify;">
        Nous allons essayer de comparer deux tailles de réseaux de neurones afin de répondre à notre 
        problématique et classification multiple et binaire. Seront essayés :<br>
        <ul> 
            <li> Les fonctions d'activation reLu, tanh et polynomial (cubic)</li>
            <li> Les métriques f1-score et f1-score pondéré</li>
        <ul>
        <br>
        <u><b>Voici la structure de nos réseaux de neurones (optimiseur='adam') :</u></b>
        <br>
        '''

DNN_EXEMPLE_RESEAU_MULI = '''<p style="text-align:justify;">
        Exemple de réseau DNN pour une classification multiple (Loss=’sparse_categorial_cross_entropy’)
        '''

DNN_EXEMPLE_RESEAU_BINARY = '''<p style="text-align:justify;">
        Exemple de réseau DNN pour une classification binaire (Loss=’binary_cross_entropy’)
        '''

DNN_EXEMPLE_STACKED_BINARY = '''<p style="text-align:justify;">
        Réseau Stacked (DNN+XGBoost) pour une classification binaire (Loss=’binary_cross_entropy’)
        '''

DNN_EXEMPLE_MLPCLASSIFIER_BINARY = '''<p style="text-align:justify;">
        Exemple de réseau DNN modèle MLPClassifier pour une classification binaire (Loss=’binary_cross_entropy’)
        '''

DNN_DATAFRAME = '''<p style="text-align:justify;">
        <u><b>Echantillons de données utilisé pour entrainer nos modèles DNN : </u></b><br>
        Afin d'optimiser les temps de calcul, nous partirons du dataframe récupéré en sortie du pipeline de rééquilibrage.
        Nous n'utiliserons que des données rééquilibrées avec SMOTE dont voici la répartition :<br>
        <br>
        <u><b>Callback implémentés : </u></b>
        <br>
        <ul>
            <li> EarlyStopping : monitor=’val_loss’, patience=3, restore_best_weights=True</li>
            <li> ModelCheckpoint : ‘model_best.h5, monitor=’val_loss’, save_best_only=True</li>
            <li> LearningRateScheduler : epoch < 10 maintien du lr initial, sinon lr  lr *e−0.1 (réduction de 9,52% à chaque époque)</li>
        </ul>    
        '''

DNN_COMPARATIF_TAILLE_DNN = '''<p style="text-align:justify;">
        <u><b> Premières constatations sur l'utilisation de ces modèles :</u></b><br>
        <ol>
            <li> La fonction d’activation reLu ne semble pas améliorer les scores et même augmente le phénomène 
        de surapprentissage du modèle. Il ne semble pas y avoir de relation non linéaire faible sur 
        ce problème de classification multiclasse.<br>
            <li> Les résultats sont similaires lors de l’utilisation d’une fonction d’activation polynomiale 
        cubique ce qui laisse à penser qu’il n’y a pas de relation non linéaire importante.</li>
            <li> Les réseaux de moins de 30 neurones donnent des scores plus faibles que les réseaux avec plus 
        de neurones.</li>
            <li> A contrario, nous pouvons constater un phénomène de surapprentissage plus important sur les réseaux
        de plus de 300 neurones et ce malgré l’utilisation de couche dropout et d’un callback 
        EarlyStopping (phénomène moins présent en classification binaire).</li>
        <ol>
        <br>
        <u><b>Pour diminuer ce surapprentissage, nous pourrions :</u></b><br>
        <ul> 
            <li> Rechercher des modèles de DNN ayant fait leurs preuves sur d'autres problématiques de classification</li> 
            <li> Trouver un compromis sur le nombre de neurones</li>
            <li> Revoir le % de drop out</li>
            <li> Privilégier un modèle plus complexe (mixer DNN et modèle simple)</li>
        </ul>
        <br>
        '''
DNN_XGBOOST = '''<p style="text-align:justify;">
        Les scores sont légèrement améliorés mais le sur-apprentissage est encore plus important.<br>
        Nous pourrions essayer de diminuer ce sur-apprentissage en :<br>
        <ul> 
            <li> Ajoutant des techniques L1/L2</li>
            <li> En optimisant le modèle de régression logistique</li>
            <li> En remplaçant ce modèle par un autre modèle tel qu’un arbre de décision par exemple</li>
        <ul>
        '''
DNN_MLPCLASSIFIER = '''<p style="text-align:justify;">
        Si les scores d’apprentissage semblent très bons sur les deux classes qui nous intéressent, 
        il ressort un surapprentissage très (trop) important pour permettre l’utilisation de ce modèle 
        en l’état.<br>
        Il reste intéressant et mériterait d’être tuné afin de corriger ce phénomène de sur-apprentissage. 
        Malheureusement, les temps de calcul (8 fois plus long qu’un modèle Kéras à plus de 300 neurones)
        et le faible nombre de paramètres disponible pour le configurer nous contraignent à ne pas 
        pousser son analyse plus loin.

        '''

# INTERPRETABILITE
# ----------------
INTERP_INTRODUCTION = '''<p style="text-align:justify;">
        Nous avons comparé les résultats proposés par la librairie Shap (méthode agnostique) sur plusieurs 2 modèles simples et 1 DNN. Nous
        avons étudié :<br>
        <ul>
            <li> Graphique de Beeswarm</li>
            <li> Graphique de dépendance</li>
            <li> Graphique de force</li>
        </ul><br>
        Cette analyse a été complétée par l'utilisation du modèle Lime (méthode spécifique) afin d'approximer localement le comportement 
        d'un modèle complexe.<br>

        '''
INTERP_TOP10 = "<u><b>Top 10</u></b>"
INTERP_FLOP10 = "<u><b>Flop 10</u></b>"
INTERP_FORCE = "<u><b>Graphe de force (DNN)</u></b>"

ML_RESULTATS_INTERPRETABILITE = '''<p style="text-align:justify;">
        Nous devrions avoir des résultats assez similaires entre les deux méthodes SHAP, graphe de force et LIME. 
        Si l’on retrouve des similitudes : <br>
        <ul>
            <li> importante influence de ‘user_secu1’
            <li> idem pour ‘vehi_obsm_1’
        </ul> 
        Ce n’est pas le cas pour d’autres champs qui selon la méthode d’interprétabilité privilégiée n’aura 
        pas la même influence :<br>  
        <ul>
            <li> ‘zone_geographique’ qui influence le modèle vision SHAP et qui n’apparait pas en méthode LIME.
        </ul>
        Ces écarts montrent les limites de ces modèles d’interprétabilité qui, s’ils permettent d’apporter 
        des éléments d’explication quant au comportement d’un modèle de ML, ne permet pas d’expliquer 
        précisément les raisons exactes : <u>les résultats restent des estimations</u>. 
        '''

# -----------------------------
# 4. CONCLUSIONS & PERSPECTIVES
# -----------------------------
CONCLUSION = '''<p style="text-align:justify;">
        Dès la phase d'exploration des données, nous avions constaté une très faible corrélation avec la variable cible. 
        Cette faible corrélation s'est confirmée au fil du projet avec des résultats de prédiction pas à la hauteur de nos 
        espérances.<br>
        <br>
        Les faibles résultats obtenus par nos modèles de prédictions sont finalement contre balancés par les informations
        intéressantes que nous ont permis de faire ressortir d'une part la phase de visualisation des données et 
        d'autres part l'interprétabilité.<br>
        <br>
        <b><u>Visualisation des données :</b></u><br>
        <ul>
            <li> Les zones accidentogènes sont plus fréquentes en agglomération qu'en dehors</li>
            <li> S'il y a plus d'accidents en zones denses démographiquement, ce ne sont pas les zones les plus meurtrières</li>
            <li> Il y a plus de chance d'avoir un accident grave, seul, en pleine ligne droite par beau temps.</li>
        </ul><br>
        <br>
        <b><u>Interprétabilité des modèles : </b></u><br>
        <ul>
            <li> Il ressort que le port des équipements de sécurité influence les prédictions de tous les modèles</li>
            <li> Beaucoup de caractéristiques en lien avec l'usager ressortent avec une forte influence</li>
        </ul>
        '''
PERSPECTIVE = '''<p style="text-align:justify;">
        Pour aller plus loin, nous pourrions :<br>
        <ol>
            <li> Rechallenger la phase de préprocessing : une passe pour rechallenger l’ensemble du travail 
            préparatoire et notamment une réflexion autour de la méthode de sélection des meilleures variables
            explicatives</li>
            <li> Enrichir les données disponibles : Difficile dans notre cas de figure de récupérer des informations 
            complémentaires mais nous pouvons constater qu’il nous manque des données certainement très importantes 
            pour notre analyse :<br>
            <ul> 
                <li> Conduite sous l’emprise de substances interdites (alcool, drogue)</li>
                <li> Conditions physiques / vigilance dégradée du conducteur (baisse des réflexes, distraction due aux écrans, utilisation de téléphone portable, etc.)</li>
                <li> Possession d’un permis de conduire valide</li>
                <li> Possession d’une assurance à jour</li>
                <li> Conducteur propriétaire du véhicule (mettant en jeu la responsabilité du conducteur)
                <li> Vitesse réelle au moment de l’impact>/li>
                <li> Respect des distances de sécurité</li>
            </ul>
            <li> Aborder le problème par une approche différente : <br>
            <ul>
                <li> Privilégier des données à la maille 'accident' plutôt qu'à la maille 'usager'</li>
                <li> Basculer sur un problème de régression linéaire plutôt que de classification en se 
                basant par exemple sur les abaques de notation des accidents proposés par le ministère.</li>
            </ul>
        </ol>

        '''

