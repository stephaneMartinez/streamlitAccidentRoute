import pickle

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer 
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as Pipeline_imb
from sklearn.cluster import KMeans
 
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



def init_df (carac_an,
             carac_mois,
             carac_jour,
             carac_agg,
             carac_atm,
             carac_col,
             carac_lum,
             carac_lat,
             carac_long,
             lieu_catr,
             lieu_circ,
             lieu_plan,
             lieu_situ,
             agg_catv_perso,
             vehi_choc,
             vehi_manv,
             vehi_motor,
             vehi_obsm,
             vehi_obs,
             user_catu,
             user_secu1,
             user_secu2,
             user_secu3,
             user_trajet,
             ):
    colonnes = ['id_accident','id_vehicule','num_veh','carac_an','carac_mois','carac_jour', 'carac_hrmn',
                'carac_agg', 'carac_atm', 'carac_col', 'carac_com', 'carac_dept', 'carac_gps_lat', 
                'carac_gps_long', 'carac_int', 'carac_lum', 'lieu_catr', 'lieu_circ', 'lieu_infra', 'lieu_larrout',
                'lieu_lartpc', 'lieu_nbv', 'lieu_plan', 'lieu_prof', 'lieu_situ', 'lieu_surf', 'lieu_vma', 
                'lieu_vosp', 'vehi_catv', 'vehi_choc', 'vehi_manv', 'vehi_motor', 'vehi_obs', 'vehi_obsm',
                'vehi_occutc', 'vehi_senc', 'user_an_nais', 'user_catu', 'user_actp', 'user_etatp', 'user_locp',
                'user_place', 'user_secu1', 'user_secu2', 'user_secu3', 'user_sexe', 'user_trajet', 
                'agg_catv_perso', 'agg_is_conducteur_vehicule', 'agg_is_conductrice_vehicule', 
                'agg_nb_pieton_vehicule', 'agg_nb_passager_vehicule' , 'agg_nb_indemne_vehicule', 
                'agg_nb_blesse_leger_vehicule', 'agg_nb_blesse_grave_vehicule', 'agg_nb_tue_vehicule',
                'agg_nb_total_vehicule', 'agg_nb_total_velo', 'agg_nb_total_vsp', 'agg_nb_total_moto' ,
                'agg_nb_total_vl', 'agg_nb_total_pl', 'agg_nb_total_va', 'agg_nb_total_conducteur', 
                'agg_nb_total_conductrice', 'agg_nb_total_pieton', 'agg_nb_total_passager', 'count', 'date', 'heure']

    df = pd.DataFrame (0, index=range(1), columns=colonnes)
    df.carac_an = carac_an
    df.carac_mois = carac_mois
    df.carac_jour = carac_jour
    df.carac_agg = carac_agg
    df.carac_atm = carac_atm
    df.carac_col = carac_col
    df.carac_lum,
    df.lieu_catr = lieu_catr,
    df.lieu_circ = lieu_circ,
    df.lieu_plan = lieu_plan,
    df.lieu_situ = lieu_situ,
    df.agg_catv_perso = agg_catv_perso,
    df.vehi_choc = vehi_choc,
    df.vehi_manv = vehi_manv,
    df.vehi_motor = vehi_motor,
    df.vehi_obsm = vehi_obsm,
    df.vehi_obs = vehi_obs,
    df.user_catu = user_catu,
    df.user_secu1 = user_secu1,
    df.user_secu2 = user_secu2,
    df.user_secu3 = user_secu3,
    df.user_trajet = user_trajet,
    df.carac_gps_lat = carac_lat,
    df.carac_gps_long = carac_long

    return df


# Fonction de sélection des variables à conserver/éliminer des features
def keep_as_feature(df:pd.DataFrame, variable:str, threshold:float) -> bool :
    """ Retourne True/False selon que threshold >= df.coef pour df.variable = variable
        Cette fonction sera utilisée pour définir si une variable doit être conservée ou éliminée des features en fonction d'un seuil
        d'acceptation.

        ARGS :
        - df : dataframe contenant les colonnes 'variable' et 'coef'
        - variable : (str) nom de la variable à tester
        - threshold (float) : seuil devant être atteint : le test se fait en valeur absolue du coef obtenu lors du test

        RETURN :
        - si le seuil a été atteint en valeur absolue : True
        - sinon : False
    """
    
    if not (df[(df['variable'] == variable) & (np.abs(df['coef'] >= threshold))].empty):
        return True

    return False

class WdropColumns (BaseEstimator, TransformerMixin) :
    """ FONCTION DE TRANSFORMATION PERMETTANT DE SUPPRIMER TOUTES LES COLONNES DEFINIES DANS UNE LISTE
Parameters
----------
- cols_to_drop (list) : Liste de noms de colonnes à supprimer de df

Functions 
---------
fit, fit_transform, transform

    """
    def __init__ (self, cols_to_drop:list=[]) :
        self.cols_to_drop = cols_to_drop

    def fit (self, X:pd.DataFrame, y:list=None) :
        return self

    def transform (self, X:pd.DataFrame):
        return X.drop(self.cols_to_drop, axis=1, errors='ignore')

# Finalement inutile du fait de l'utilisation de make_column_transformer qui filtre les colonnes non explicitement identifiées à traiter

# TRANSFORMER EN CHARGE DE GENERER LA DATE_ACCIDENT à partir de 'carac_an', 'carac_mois' et 'carac_jour'
class WdateAccident (BaseEstimator, TransformerMixin) :
    """ FONCTION DE TRANSFORMATION UTILISABLE DANS UN PIPELINE :
        - Génère une variable 'date_accident' à partir de 'carac_an', 'carac_mois' et 'carac_jour'
        - Supprime les variables utilisées pour générer cette date

Parameters
----------
- dropC (bool) : indicateur de suppression de X des variables utilisées pour générer 'date_accident'

Functions 
---------
fit, fit_transform, transform

"""
    def __init__ (self, columnName='date_accident') :
        self.columnName = columnName
        
    def fit (self, X:pd.DataFrame, y:list=None) :
        return self
    
    def transform (self, X:pd.DataFrame):
        cols = ['carac_an', 'carac_mois', 'carac_jour']

        # Contrôle existance des variable an, mois, jour dans X
        for col in cols :
            if col not in X.columns :
                return None
                
        # Création de la date                     
        X[self.columnName] =  X.apply (self._composerDate, axis=1)

            
        return X[['date_accident']]
    
    def _composerDate(self, row) :
        jour_str = str(int(row['carac_jour'])).zfill(2)
        mois_str = str(int(row['carac_mois'])).zfill(2)
        année_str = str(int(row['carac_an']))
    
        return int(f"{année_str}{mois_str}{jour_str}")

    def get_feature_names_out(self, input_features=None):
        ''' Fonction permettant de retourner le nom de la colonne après transformation
            (si utilisation de make_column_transformer les noms seront perdus)
        '''
        # Sinon retourne le nom d'origine
        return ['data_Accident']

# TRANSFORMER EN CHARGE DE GERER LES DIFFERENTS HYPERPARAMETRE  POUR CHAQUE VARIABLE
class Transformer (BaseEstimator, TransformerMixin) :
    """ FONCTION DE TRANSFORMATION UTILISABLE DANS UN PIPELINE : VARIABLES CATEGORIELLE    
Parameters
----------
- columnName (str)    : Nom de la colonne à transformer
        
- nan_stratey : 
  -> None               : aucune opération
  -> 'drop' (str)       : dropna() 
  -> 'specific' (str)   : Les fonctions _missing_values_fit() et _missing_values_transform() seront utilisées : elles pourrront être 
                          surchargées dans la classe mère
  -> strategy of SimpleImputer : application de la stratégie prise en charge par SimpleImputer ('constant', 'mean', 'median', 'most_frequent')
- fill_value :
  -> valeur à privilégier si nan_strategy='constant'

- encoder : 
  ->  None              : aucun encodage
  -> 'OneHotEncoder'    : encodage avec OneHotEncoder
  -> 'dummies'          : encodage avec get_dummies

- scaler :
  -> 'StandardScaler'   : Normalisation
  -> 'MinMaxScaler'     : Sérialisation (pour distrib. loi normale)

- dropC  (bool)         : Suppression de la colonne   

- outliers (bool)       : Reprise des Outliers    
        
Functions 
---------
- fit, fit_transform, transform
- _outliers_fit(X, y)           : Fonction de reprise des outliers lancée dans fit, cette fonction pourra être surchargée dans la classe fille
- _outliers_transform(X) -> X   : Fonction de reprise des outliers lancée dans transform, cette fonction sera surchargée dans la classe fille
- _missing_values_fit (X, y)    : Fonction de reprise des missing values lancée dans fit, cette fonction devra être surchargée et contiendra les règles spécifique.
- _missing_values_transfor(X) -> X : Fonction qui pourra être surchargée afin d'appliquer une règle spécifique de gestion des NaN                                
            
    """
    # Initialisation des constantes 
    OneHotEncoder = 'OneHotEncoder'             # Encoder
    dummies = 'dummies'                         # Encoder
    MinMaxScaler = 'MinMaxScaler'               # Scaler
    StandardScaler = 'StandardScaler'           # Scaler
    drop = 'drop'                               # NaN
    specific = 'specific'                       # NaN
    
    def __init__ (self, *, columnName:str, nan_strategy:str, encoder:str, scaler:str, dropC:bool, outliers:any, fill_value:any=None) :
        # Initialisation de la classe : les paramètres doivent être conservés en l'état
        self.columnName = columnName
        self.nan_strategy = nan_strategy              # missing value
        self.fill_value = fill_value    
        self.encoder = encoder                        # encoder
        self.scaler = scaler                          # scaler
        self.dropC = dropC                            # Conservarion de la colonne O/N
        self.outliers = outliers                      # Outlier

        self._initialized_encoder = None              # encoder
        self._dummies = False
        self._initialized_scaler = None               # scaler
        self._initialized_imputer = None              # imputer
        self._dropna = False                          # missing values
        self._specific_nan = False
    

    def _initialize_params(self):
        # Suppression de la colonne demandée, pas besoin d'initialiser les autres paramètres
        if self.dropC:
            return

        # Paramétrage missing_values
        if self.nan_strategy == Transformer.drop:
            self._dropna = True
            
        elif self.nan_strategy == Transformer.specific:
            self._specific_nan = True
            
        elif self.nan_strategy:
            self._initialized_imputer = SimpleImputer(strategy=self.nan_strategy, fill_value=self.fill_value)

        # Paramétrage encoder
        if self.encoder == Transformer.OneHotEncoder:
            self._initialized_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore', drop='first')
            
        elif self.encoder == Transformer.dummies:
            self._dummies = True

        # Paramétrage de Scaler
        if self.scaler == Transformer.StandardScaler:
            self._initialized_scaler = StandardScaler()
            
        elif self.scaler == Transformer.MinMaxScaler:
            self._initialized_scaler = MinMaxScaler()

 
    def fit (self, X:pd.DataFrame, y:list=None) :
        
        # Initialisation objets du trasnformer
        self._initialize_params()
        
        if self.dropC :
            return self                                     # demande de suppression de la colonne : rien à faire dans fit

        # outliers
        self._outliers_fit(X, y)

        # missing_values
        if self._initialized_imputer is not None :
            self._initialized_imputer.fit(X[[self.columnName]])
        
        elif self._specific_nan :
            # lancement de la fonction spécifique
            self._missing_values_fit(X, y)

        # One Hot Encoder
        if self._initialized_encoder is not None :
            self._initialized_encoder.fit(X[[self.columnName]])

        # scaler
        if self._initialized_scaler is not None :
            self._initialized_scaler.fit(X[[self.columnName]])

        return self
    
    def transform (self, X:pd.DataFrame):
        # Suppression de la colonne si demandé
        if self.dropC :
            return X.drop(self.columnName, axis=1)
            
        # Ouliers
        X = self._outliers_transform(X)

        # missing Values
        if self._initialized_imputer :
            # Traitement des NaN via Simple Imputer
            X[self.columnName] = self.simpleimputer.transform(X[[self.columnName]])
                    
        elif self._specific_nan :
            X = self._missing_values_transform(X)

        # OneHotEncoder
        if self._initialized_encoder is not None :
            encoded_array = self._initialized_encoder.transform(X[[self.columnName]])
            encoded_df = pd.DataFrame(encoded_array, columns=self._initialized_encoder.get_feature_names_out([self.columnName]))
            X = X.drop(self.columnName, axis=1)
            X = pd.concat([X.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

        elif self._dummies :
            X = pd.get_dummies(X, columns=[self.columnName], drop_first=True )
 
        # Scaler
        if self._initialized_scaler is not None :
            # Exécution du scaler sur la variable
            X[self.columnName] = self._initialized_scaler.transform(X[[self.columnName]])

        return X

    def get_feature_names_out(self, input_features=None):
        ''' Fonction permettant de retourner le nom des colonnes après transformation
            (si utilisation de make_column_transformer les noms seront perdus)
        '''
        if self._initialized_encoder is not None :
            # Si OneHotEncoder
            #return [f"{feature}_{i}" for feature in input_features for i in range(self.encoder.n_values_[i])] 
            return self._initialized_encoder.get_feature_names_out([self.columnName])

        # Sinon retourne le nom d'origine
        return input_features  
    
    def _outliers_fit(self, X:pd.DataFrame, y=None) :
        """ Fonction de reprise des outliers qui sera exécutée dans le fit afin de calculer des données nécessaire à 
        _outliers_transform().

        Cette fonction ne fait rien mais si nécessaire, elle pourra être surchargée dans la classe fille.
        """
        
        return self

    
    def _outliers_transform(self, X:pd.DataFrame) -> pd.DataFrame :
        """ Fonction de reprise des outliers : cette fonction sera surchargée dans la classe fille. """
        if self.outliers is not None and self.outliers != False : 
            print (">>> La fonction _outliers doit être surchargée. Traitement des Outliers ignoré !!!")
        
        return X

        
    def _missing_values_fit(self, X:pd.DataFrame, y=None) :
        """ Fonction exécutée dans fit et permettant de réaliser des calculs préparatoire à une opération de transform. 
        Cette fonction devra être surchargée dans la classe fille afin de répondre au besoin spécifique.
        Par exemple : Réaliser un remplacement de la classe la plus représentée par clé de répartition

        Il sera préférable de rechercher ces valeurs de remplacement dans la fonction fit pour éviter la fuite de données.        
        """

        return self

        
    def _missing_values_transform(self, X:pd.DataFrame) -> pd.DataFrame :
        """ Fonction exécutée dans le trasnform et réalisant la trasnformation de X attendue.
        Elle utilisera les valeurs de remplacment qui auront été préalablement calculées dans _missing_values_fit (si nécessaire).

        Cette fonction devra être surchargée dans la classe fille.
        """
        # fillna()
        if self._specific_nan :
            print (">>> La fonction _missing_values_transform (et si besoin fit()) doit être surchargée. Traitement spécifique des missing values ignoré !!!")
        
        return X

# Transformer de clusterisation des coordonnées gps (carac_gps_lat et carac_gps_long) : au lieu de les supprimer
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
        ''' Fonction permettant de retourner le nom des colonnes après transformation
            (si utilisation de make_column_transformer les noms seront perdus)
        '''
        nb_clusters = self.n_clusters
        cluster_column_names = [f"{self.outputColumnName}_{i}" for i in range(nb_clusters-1)]
        return cluster_column_names
    
    # TRANSFORMER VARIABLE 'carac_an' - 'carac_mois' - 'carac_jour'
class Wcarac_an (Transformer) :
    def __init__ (self, nan_strategy=None, fill_value=None, encoder:str=None, dropC:bool=True) :
        super().__init__(columnName=self.__class__.__name__[1:], nan_strategy=nan_strategy, encoder=encoder, scaler=None, dropC=dropC, outliers=False)

class Wcarac_mois (Transformer) :
    def __init__ (self, nan_strategy=None, fill_value=None, encoder:str=None, dropC:bool=True) :
        super().__init__(columnName=self.__class__.__name__[1:], nan_strategy=nan_strategy, encoder=encoder, scaler=None, dropC=dropC, outliers=False)

class Wcarac_jour (Transformer) :
    def __init__ (self, nan_strategy=None, fill_value=None, encoder:str=None, dropC:bool=True) :
        super().__init__(columnName=self.__class__.__name__[1:], nan_strategy=nan_strategy, encoder=encoder, scaler=None, dropC=dropC, outliers=False)

# TRANSFORMER VARIABLE 'carac_agg'
class Wcarac_agg (Transformer) :
    def __init__ (self, nan_strategy=None, fill_value=None, encoder:str=None, dropC:bool=False) :
        super().__init__(columnName=self.__class__.__name__[1:], nan_strategy=nan_strategy, encoder=encoder, scaler=None, dropC=dropC, outliers=False)

# TRANSFORMER VARIABLE 'carac_atm'
class Wcarac_atm (Transformer) :
    def __init__ (self, nan_strategy=None, fill_value=None, encoder:str='OneHotEncoder', dropC:bool=False) :
        super().__init__(columnName=self.__class__.__name__[1:], nan_strategy=nan_strategy, encoder=encoder, scaler=None, dropC=dropC, outliers=False)

# TRANSFORMER VARIABLE 'carac_col'
class Wcarac_col (Transformer) :
    def __init__ (self, nan_strategy=None, fill_value=None, encoder:str='OneHotEncoder', dropC:bool=False) :
        super().__init__(columnName=self.__class__.__name__[1:], nan_strategy=nan_strategy, encoder=encoder, scaler=None, dropC=dropC, outliers=False)

# TRANSFORMER VARIABLE 'carac_com'
class Wcarac_com (Transformer) :
    def __init__ (self, *, nan_strategy=None, fill_value=None, encoder:str='OneHotEncoder', scaler:str=None, dropC:bool=False) :
        super().__init__(columnName=self.__class__.__name__[1:], nan_strategy=nan_strategy, encoder=encoder, scaler=scaler, dropC=dropC, outliers=False)

# TRANSFORMER VARIABLE 'carac_dept'
class Wcarac_dept (Transformer) :
    def __init__ (self, *, nan_strategy=None, fill_value=None, encoder:str='OneHotEncoder', scaler:str=None, dropC:bool=False) :
        super().__init__(columnName=self.__class__.__name__[1:], nan_strategy=nan_strategy, encoder=encoder, scaler=scaler, dropC=dropC, outliers=False)

# TRANSFORMER VARIABLE 'carac_int'
class Wcarac_int (Transformer) :
    def __init__ (self, nan_strategy=None, fill_value=None, encoder:str='OneHotEncoder', dropC:bool=True) :
        super().__init__(columnName=self.__class__.__name__[1:], nan_strategy=nan_strategy, encoder=encoder, scaler=None, dropC=dropC, outliers=False)

# TRANSFORMER VARIABLE 'carac_lum'
class Wcarac_lum (Transformer) :
    def __init__ (self, nan_strategy=None, fill_value=None, encoder:str='OneHotEncoder', dropC:bool=False) :
        super().__init__(columnName=self.__class__.__name__[1:], nan_strategy=nan_strategy, encoder=encoder, scaler=None, dropC=dropC, outliers=False)

# TRANSFORMER VARIABLE 'lieu_catr'
class Wlieu_catr (Transformer) :
    def __init__ (self, nan_strategy=None, fill_value=None, encoder:str='OneHotEncoder', dropC:bool=False) :
        super().__init__(columnName=self.__class__.__name__[1:], nan_strategy=nan_strategy, encoder=encoder, scaler=None, dropC=dropC, outliers=False)

# TRANSFORMER VARIABLE 'lieu_circ'
class Wlieu_circ (Transformer) :
    def __init__ (self, nan_strategy=None, fill_value=None, encoder:str='OneHotEncoder', dropC:bool=False) :
        super().__init__(columnName=self.__class__.__name__[1:], nan_strategy=nan_strategy, encoder=encoder, scaler=None, dropC=dropC, outliers=False)

# TRANSFORMER VARIABLE 'lieu_infra'
class Wlieu_infra (Transformer) :
    def __init__ (self, nan_strategy=None, fill_value=None, encoder:str='OneHotEncoder', dropC:bool=True) :
        super().__init__(columnName=self.__class__.__name__[1:], nan_strategy=nan_strategy, encoder=encoder, scaler=None, dropC=dropC, outliers=False)

# TRANSFORMER VARIABLE 'lieu_plan'
class Wlieu_plan (Transformer) :
    def __init__ (self, nan_strategy=None, fill_value=None, encoder:str='OneHotEncoder', dropC:bool=False) :
        super().__init__(columnName=self.__class__.__name__[1:], nan_strategy=nan_strategy, encoder=encoder, scaler=None, dropC=dropC, outliers=False)

# TRANSFORMER VARIABLE 'lieu_situ'
class Wlieu_situ (Transformer) :
    def __init__ (self, nan_strategy=None, fill_value=None, encoder:str='OneHotEncoder', dropC:bool=False) :
        super().__init__(columnName=self.__class__.__name__[1:], nan_strategy=nan_strategy, encoder=encoder, scaler=None, dropC=dropC, outliers=False)

# TRANSFORMER VARIABLE 'lieu_surf'
class Wlieu_surf (Transformer) :
    def __init__ (self, nan_strategy=None, fill_value=None, encoder:str='OneHotEncoder', dropC:bool=True) :
        super().__init__(columnName=self.__class__.__name__[1:], nan_strategy=nan_strategy, encoder=encoder, scaler=None, dropC=dropC, outliers=False)

# TRANSFORMER VARIABLE 'lieu_vosp'
class Wlieu_vosp (Transformer) :
    def __init__ (self, nan_strategy=None, fill_value=None, encoder:str='OneHotEncoder', dropC:bool=True) :
        super().__init__(columnName=self.__class__.__name__[1:], nan_strategy=nan_strategy, encoder=encoder, scaler=None, dropC=dropC, outliers=False)

# TRANSFORMER VARIABLE 'vehi_catv'
class Wvehi_catv (Transformer) :
    def __init__ (self, nan_strategy=None, fill_value=None, encoder:str='OneHotEncoder', dropC:bool=True) :
        super().__init__(columnName=self.__class__.__name__[1:], nan_strategy=nan_strategy, encoder=encoder, scaler=None, dropC=dropC, outliers=False)

# TRANSFORMER VARIABLE 'agg_catv_perso'
class Wagg_catv_perso (Transformer) :
    def __init__ (self, nan_strategy=None, fill_value=None, encoder:str='OneHotEncoder', dropC:bool=False) :
        super().__init__(columnName=self.__class__.__name__[1:], nan_strategy=nan_strategy, encoder=encoder, scaler=None, dropC=dropC, outliers=False)


# TRANSFORMER VARIABLE 'vehi_choc'
class Wvehi_choc (Transformer) :
    def __init__ (self, nan_strategy=None, fill_value=None, encoder:str='OneHotEncoder', dropC:bool=False) :
        super().__init__(columnName=self.__class__.__name__[1:], nan_strategy=nan_strategy, encoder=encoder, scaler=None, dropC=dropC, outliers=False)

# TRANSFORMER VARIABLE 'vehi_manv'
class Wvehi_manv (Transformer) :
    def __init__ (self, nan_strategy=None, fill_value=None, encoder:str='OneHotEncoder', dropC:bool=False) :
        super().__init__(columnName=self.__class__.__name__[1:], nan_strategy=nan_strategy, encoder=encoder, scaler=None, dropC=dropC, outliers=False)

# TRANSFORMER VARIABLE 'vehi_motor'
class Wvehi_motor (Transformer) :
    def __init__ (self, nan_strategy=None, fill_value=None, encoder:str='OneHotEncoder', dropC:bool=False) :
        super().__init__(columnName=self.__class__.__name__[1:], nan_strategy=nan_strategy, encoder=encoder, scaler=None, dropC=dropC, outliers=False)

# TRANSFORMER VARIABLE 'vehi_obs'
class Wvehi_obs (Transformer) :
    def __init__ (self, nan_strategy=None, fill_value=None, encoder:str='OneHotEncoder', dropC:bool=False) :
        super().__init__(columnName=self.__class__.__name__[1:], nan_strategy=nan_strategy, encoder=encoder, scaler=None, dropC=dropC, outliers=False)

# TRANSFORMER VARIABLE 'vehi_obsm'
class Wvehi_obsm (Transformer) :
    def __init__ (self, nan_strategy=None, fill_value=None, encoder:str='OneHotEncoder', dropC:bool=False) :
        super().__init__(columnName=self.__class__.__name__[1:], nan_strategy=nan_strategy, encoder=encoder, scaler=None, dropC=dropC, outliers=False)

# TRANSFORMER VARIABLE 'vehi_senc'
class Wvehi_senc(Transformer) :
    def __init__ (self, nan_strategy=None, fill_value=None, encoder:str='OneHotEncoder', dropC:bool=True) :
        super().__init__(columnName=self.__class__.__name__[1:], nan_strategy=nan_strategy, encoder=encoder, scaler=None, dropC=dropC, outliers=False)

# TRANSFORMER VARIABLE 'user_catu'
class Wuser_catu(Transformer) :
    def __init__ (self, nan_strategy=None, fill_value=None, encoder:str='OneHotEncoder', dropC:bool=False) :
        super().__init__(columnName=self.__class__.__name__[1:], nan_strategy=nan_strategy, encoder=encoder, scaler=None, dropC=dropC, outliers=False)

# TRANSFORMER VARIABLE 'user_secu1'
class Wuser_secu1(Transformer) :
    def __init__ (self, nan_strategy=None, fill_value=None, encoder:str='OneHotEncoder', dropC:bool=False) :
        super().__init__(columnName=self.__class__.__name__[1:], nan_strategy=nan_strategy, encoder=encoder, scaler=None, dropC=dropC, outliers=False)

# TRANSFORMER VARIABLE 'user_secu2'
class Wuser_secu2(Transformer) :
    def __init__ (self, nan_strategy=None, fill_value=None, encoder:str='OneHotEncoder', dropC:bool=False) :
        super().__init__(columnName=self.__class__.__name__[1:], nan_strategy=nan_strategy, encoder=encoder, scaler=None, dropC=dropC, outliers=False)

# TRANSFORMER VARIABLE 'user_secu3'
class Wuser_secu3(Transformer) :
    def __init__ (self, nan_strategy=None, fill_value=None, encoder:str='OneHotEncoder', dropC:bool=False) :
        super().__init__(columnName=self.__class__.__name__[1:], nan_strategy=nan_strategy, encoder=encoder, scaler=None, dropC=dropC, outliers=False)

# TRANSFORMER VARIABLE 'user_trajet'
class Wuser_trajet(Transformer) :
    def __init__ (self, nan_strategy=None, fill_value=None, encoder:str='OneHotEncoder', dropC:bool=False) :
        super().__init__(columnName=self.__class__.__name__[1:], nan_strategy=nan_strategy, encoder=encoder, scaler=None, dropC=dropC, outliers=False)


# TRANSFORMER VARIABLE 'lieu_nbv'
class Wlieu_nbv(Transformer) :
    def __init__ (self, nan_strategy=None, fill_value:any=None, encoder:str=None, scaler:str='StandardScaler', outliers:bool=False, dropC:bool=True) :
        super().__init__(columnName=self.__class__.__name__[1:], nan_strategy=nan_strategy, fill_value=fill_value, encoder=encoder, scaler=scaler, dropC=dropC, outliers=outliers)

# TRANSFORMER VARIABLE 'user_an_nais'
class Wuser_an_nais(Transformer) :
    def __init__ (self, nan_strategy=None, fill_value:any=None, encoder:str=None, scaler:str='StandardScaler', outliers:bool=False, dropC:bool=True) :
        super().__init__(columnName=self.__class__.__name__[1:], nan_strategy=nan_strategy, fill_value=fill_value, encoder=encoder, scaler=scaler, dropC=dropC, outliers=outliers)

# TRANSFORMER VARIABLE 'agg_nb_total_conductrice'
class Wagg_nb_total_conductrice(Transformer) :
    def __init__ (self, nan_strategy=None, fill_value:any=None, encoder:str=None, scaler:str='StandardScaler', outliers:bool=False, dropC:bool=True) :
        super().__init__(columnName=self.__class__.__name__[1:], nan_strategy=nan_strategy, fill_value=fill_value, encoder=encoder, scaler=scaler, dropC=dropC, outliers=outliers)

# TRANSFORMER VARIABLE 'agg_nb_total_vehicule'
class Wagg_nb_total_vehicule(Transformer) :
    def __init__ (self, nan_strategy=None, fill_value:any=None, encoder:str=None, scaler:str='StandardScaler', outliers:bool=False, dropC:bool=False) :
        super().__init__(columnName=self.__class__.__name__[1:], nan_strategy=nan_strategy, fill_value=fill_value, encoder=encoder, scaler=scaler, dropC=dropC, outliers=outliers)

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


# Déclaration du pipeline preprocess

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
)

LieuPipeline_mct = make_column_transformer (
    (Wlieu_catr(), ['lieu_catr']),
    (Wlieu_circ(), ['lieu_circ']),
    (Wlieu_infra(), ['lieu_infra']),
    (Wlieu_plan(), ['lieu_plan']),
    (Wlieu_situ(), ['lieu_situ']),
    (Wlieu_surf(), ['lieu_surf']),
    (Wlieu_vma(), ['lieu_vma']),
    (Wlieu_vosp(), ['lieu_vosp'])
)

VehiPipeline_mct = make_column_transformer (
    (Wvehi_catv(), ['vehi_catv']),
    (Wagg_catv_perso(), ['agg_catv_perso']),
    (Wvehi_choc(), ['vehi_choc']),
    (Wvehi_manv(), ['vehi_manv']),
    (Wvehi_motor(), ['vehi_motor']),
    (Wvehi_obs(), ['vehi_obs']),
    (Wvehi_obsm(), ['vehi_obsm']),
    (Wvehi_senc(), ['vehi_senc'])
)

UserPipeline_mct = make_column_transformer (
    (Wuser_catu(), ['user_catu']),
    (Wuser_secu1(), ['user_secu1']),
    (Wuser_secu2(), ['user_secu2']),
    (Wuser_secu3(), ['user_secu3']),
    (Wuser_trajet(), ['user_trajet'])
)

# Variables quantitatives
QuantitativePipeline_mct = make_column_transformer (
    (Wagg_nb_total_conductrice(), ['agg_nb_total_conductrice']),
    (Wagg_nb_total_vehicule(scaler='MinMaxScaler'), ['agg_nb_total_vehicule']),
    (Wlieu_vma(), ['lieu_vma']),
    (Wlieu_nbv(), ['lieu_nbv']),
    (Wuser_an_nais(nan_strategy='drop'), ['user_an_nais'])                      # test drop et median (4200 NaN)
)

# Zone géographique en remplacement des latitude et longitude (et commune et département)
GeographicalClusterPipeline_mct = make_column_transformer (
    (WKMeans(n_clusters=20, encoder=True), ['carac_gps_lat', 'carac_gps_long'])
)

# Pipeline de création de la date au format integer : YYYYMMDD : la variable sera normalisée.
DateAccident = Pipeline(steps=[('date_accident', WdateAccident()),
                               ('Normalisation_date', MinMaxScaler())
                              ])

# Pipeline regroupant les 4 typologies
preprocess = ColumnTransformer(
    transformers=[
        ('DateAccident', DateAccident, ['carac_an', 'carac_mois', 'carac_jour']),
        ('QuantitativeVars', QuantitativePipeline_mct, ['agg_nb_total_conductrice', 'agg_nb_total_vehicule', 'lieu_vma', 'lieu_nbv', 'user_an_nais']),
        ('Caractéristique', CaracPipeline_mct, ['carac_agg', 'carac_atm', 'carac_col', 'carac_com', 'carac_dept', 'carac_int', 'carac_lum', 'carac_an', 'carac_mois', 'carac_jour']),
        ('Lieu', LieuPipeline_mct, ['lieu_catr', 'lieu_circ', 'lieu_infra', 'lieu_plan', 'lieu_situ', 'lieu_surf', 'lieu_vma', 'lieu_vosp']),
        ('Vehicule', VehiPipeline_mct, ['vehi_catv', 'agg_catv_perso', 'vehi_choc', 'vehi_manv', 'vehi_motor', 'vehi_obs', 'vehi_obsm', 'vehi_senc']),
        ('Usager', UserPipeline_mct, ['user_catu', 'user_secu1', 'user_secu2', 'user_secu3', 'user_trajet']),
        ('ZoneGeo', GeographicalClusterPipeline_mct, ['carac_gps_lat', 'carac_gps_long']),
    ], verbose=False
)


# Regénérer le nom des colonnes du df en sortie
# en fonction des transformations activées pour chaque colonne du df en entrée
# Retourne une liste de noms de colonnes qui pourra être utilisée comme nom pour le df sortant
def get_feature_names(column_transformer):
    feature_names = []
    for name, transformer, columns in column_transformer.transformers_:
        if name in ('DateAccident') :
            feature_names.append(name)
            
        elif name not in ('remainder') : 
            for name_, transformer_, columns_  in transformer.transformers_:   
                if transformer_.dropC == False :
                    if hasattr(transformer_, 'encoder') and transformer_.encoder is not None :
                        if hasattr(transformer_, 'get_feature_names_out'):
                            # Si le transformateur a une méthode get_feature_names_out
                            feature_names.extend(transformer_.get_feature_names_out(columns))
                
                    else:
                        # Pas d'encoder, on retourne le nom de la colonne initial
                        feature_names.extend([transformer_.columnName])
    return feature_names


def load_model (file) :
    return pickle.load(file)

