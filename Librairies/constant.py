# Liste des constantes utilisées dans l'application
CHEMIN_IMAGE = "donnees/images/"
CHEMIN_MODELE = "donnees/models/"
MENU_PROJET = "Le Projet"
MENU_JDD = "Le jeu de données"
MENU_JDD_PRESENTATION = "Présentation"
MENU_JDD_DATAVIZ = "Data Visualisation"
MENU_MODELISATION = "Modélisation"
MENU_MODELISATION_PREPROCESS = "Pré-processing"
MENU_MODELISATION_ML = "Machine Learning"
MENU_MODELISATION_DL = "Deep Learning"
MENU_MODELISATION_INTERPRETABILITE = "Interprétabilité"
MENU_CONCLUSION = "Conclusions et Perspectives"

FEATURE_LIEU = 'lieu'
FEATURE_CARACTERISTIQUE = 'Caractéristique'
FEATURE_USAGER = 'Usager'
FEATURE_VEHICULE = 'Véhicule'

CARAC_AN = 'année'
CARAC_MOIS = 'mois'
CARAC_JOUR = 'jour'
CARAC_HEURE = 'heure'
CARAC_AGG = 'CARACTERISTIQUE - Localisation'
CARAC_ATM = "CARACTERISTIQUE - Conditions atmosphériques"
CARAC_COL = "CARACTERISTIQUE - Type de collision"
CARAC_COM = "CARACTERISTIQUE - Commune"
CARAC_DEPT = "CARACTERISTIQUE - Département"
CARAC_INT = "CARACTERISTIQUE - Intersection"
CARAC_LUM = "CARACTERISTIQUE - Lumière : conditions d’éclairage"
CARAC_GPS_LAT = 'carac_gps_lat'
CARAC_GPS_LONG = 'carac_gps_long'


LIEU_CATR = 'LIEU - Catégorie de route'
LIEU_CIRC = 'LIEU - Régime de circulation'
#LIEU_ENV1 = "LIEU Proximité d'une école"
LIEU_INFRA = "LIEU - Aménagement - Infrastructure"
LIEU_LARROUT = "LIEU - Largeur de la chaussée affectée à la circulation (en m)"
LIEU_LARTPC = "LIEU - Largeur du terre plein central s'il existe (0 sinon) (en m)"
LIEU_NBV = "LIEU - Nombre de voie(s)"
LIEU_PLAN = "LIEU - Tracé en plan"
LIEU_PR = "LIEU - Numéro du PR de rattachement (numéro de borne amont)"
LIEU_PR1 = "LIEU - Distance en mètres au PR (par rapport à la borne amont)"
LIEU_PROF = "LIEU - Profil en log"
LIEU_SITU = "LIEU - Situation de l’accident"
LIEU_SURF = "LIEU - Etat de la surface"
LIEU_V1 = "LIEU - Indice numérique du numéro de route (exemple : 2 bis, 3 ter etc)"
LIEU_V2 = "LIEU - Indice alphanumérique de la route"
LIEU_VMA = "LIEU - Vitesse maximale autorisée au lieu de l'accident (km/h)"
LIEU_VOIE = "LIEU - no de voie"
LIEU_VOSP = "LIEU - Voie réservée"

VEHI_CATV = "VEHICULE - Catégorie"
VEHI_CHOC = 'VEHICULE - Point de choc initial'
VEHI_MANV = "VEHICULE - Manœuvre principale avant l’accident"
VEHI_MOTOR = "VEHICULE - Type de motorisation du véhicule"
VEHI_OBS = "VEHICULE - Obstacle fixe heurté"
VEHI_OBSM = "VEHICULE - Obstacle mobile heurté"
VEHI_SENC = "VEHICULE - Sens de circulation"

AGG_CATV_PERSO = "VEHICULE - Catégorie de véhicule aggrégée"
AGG_CLUSTER = "Cluster coordonnées GPS"

USER_AN_NAIS = "USAGER - Année de naissance"
USER_CATU = "USAGER - Catégorie d'usager"
USER_ACTP = "USAGER - Action du piéton"
USER_ETATP = "USAGER - Permet d'identifier si le piéton accidenté était seul ou non"
USER_LOCP = "USAGER - Localisation du piéton"
USER_PLACE = "USAGER - Emplacement de la personne dans le véhicule"
USER_SECU1 = "USAGER - Equipement de Sécurité 1"
USER_SECU2 = "USAGER - Equipement de Sécurité 2"
USER_SECU3 = "USAGER - Equipement de Sécurité3"
USER_SEXE = "USAGER - Genre de l'usager"
USER_TRAJET = "USAGER - Type de trajet"
USER_GRAVITE = "USAGER - Gravité de blessure"
ZONE_GEO = "Zone géographique"

FEATURE_COLUMNNAME = 'columnName'
FEATURE_MODALITE = 'modalite'
FEATURE_PIE_ROTATION = 'pie:rotation'
FEATURE_DESIGNATION = 'designation'
FEATURE_DEFAULT = 'default'

# DISTRIBUTION
FEATURE_DISTRI_MIN_BINS = "distri:minBins"
FEATURE_DISTRI_DEFAULT_BINS = "distri:maxBins"
FEATURE_DISTRI_MAX_BINS = "distri:defaultBins"
FEATURE_DISTRI_MIN_VAL = "distri:minVisu"
FEATURE_DISTRI_MAX_VAL = "distri:maxVisu"
FEATURE_DISTRI_DISABLE = "distri:disable"

FEATURE_ZONE_GEO = {
    FEATURE_COLUMNNAME : 'zone_geographique',
    FEATURE_DESIGNATION : "Cluster basé sur les coordonnées GPS du lieu de l'accident",
    FEATURE_MODALITE : { 
        'Auvergne-Rhône-Alpes' : {
            'Latitude' : 45.5646,
            'Longitude' : 4.3849
        },
        'Bourgogne-Franche-Comté' : {
            'Latitude' : 47.2807,
            'Longitude' : 4.9994
        },
        'Bretagne' : {
            'Latitude' : 48.2020,
            'Longitude' : -2.9326
        },
        'Centre-Val de Loire' : {
            'Latitude' : 47.7516,
            'Longitude' : 1.6751
        },
        'Corse' : {
            'Latitude' : 42.0396,
            'Longitude' : 9.0129
        },
        'Grand Est' : {
            'Latitude' : 48.6921,
            'Longitude' : 6.1844
        },
        'Hauts-de-France' : {
            'Latitude' : 50.4802, 
            'Longitude' : 2.7766
        },
        'Île-de-France' : {
            'Latitude' : 48.8499,
            'Longitude' : 2.6370
        },
        'Normandie' : {
            'Latitude' : 49.1829,
            'Longitude' : 0.3707
        },
        'Nouvelle-Aquitaine' : {
            'Latitude' : 45.7080,
            'Longitude' : 0.9642
        },
        'Occitanie' : {
            'Latitude' : 43.7044,
            'Longitude' : 2.1987
        },
        'Pays de la Loire' : {
            'Latitude' : 47.7534,
            'Longitude' : -0.3256
        },
        "Provence-Alpes-Côte d'Azur" : {
            'Latitude' : 43.9352,
            'Longitude' : 6.0679
        } 
    },
    FEATURE_DEFAULT : 7
}

FEATURES = {    
    CARAC_AGG : {
        FEATURE_COLUMNNAME : 'carac_agg',
        FEATURE_PIE_ROTATION : 0,
        FEATURE_DESIGNATION : 'Localisation en ou hors agglomération',
        FEATURE_MODALITE : { 
            1 : 'Hors agglomération',
            2 : 'En agglomération',
        },
        FEATURE_DEFAULT : 1
    },
    CARAC_ATM : {
        FEATURE_COLUMNNAME : 'carac_atm',
        FEATURE_PIE_ROTATION : -30,
        FEATURE_DESIGNATION : 'Conditions atmosphériques (Normale, pluie légère, brouillard, etc.)',
        FEATURE_MODALITE : {
            -1 : ' Non renseigné',
            1 : 'Normale',
            2 : 'Pluie légère',
            3 : 'Pluie forte',
            4 : 'Neige - grêle',
            5 : 'Brouillard - fumée',
            6 : 'Vent fort - tempête',
            7 : 'Temps éblouissant',
            8 : 'Temps couvert',
            9 : 'Autre',
        },
        FEATURE_DEFAULT : 1
    },
    CARAC_COL : {
        FEATURE_COLUMNNAME : 'carac_col',
        FEATURE_PIE_ROTATION : 0,
        FEATURE_DESIGNATION : 'Type de collision (2 véhicules frontale, collisions multiples, etc.)',
        FEATURE_MODALITE : {
            -1 : 'Non renseigné',
            1 : 'Deux véhicules - frontale',
            2 : 'Deux véhicules – par l’arrière',
            3 : 'Deux véhicules – par le coté',
            4 : 'Trois véhicules et plus – en chaîne',
            5 : 'Trois véhicules et plus - collisions multiples',
            6 : 'Autre collision',
            7 : 'Sans collision',
        },
        FEATURE_DEFAULT : 3
    },
    CARAC_COM : {
        FEATURE_COLUMNNAME : 'carac_com',
        FEATURE_PIE_ROTATION : 0,
        FEATURE_DISTRI_MIN_BINS : 5,
        FEATURE_DISTRI_DEFAULT_BINS : 5,
        FEATURE_DISTRI_MAX_BINS : 16,
        FEATURE_DISTRI_MIN_VAL : 0,
        FEATURE_DISTRI_MAX_VAL : 12,
        FEATURE_DISTRI_DISABLE : True,
        FEATURE_DESIGNATION : "Commune du lieu de l'accident",
        FEATURE_MODALITE : {
        }
    },
    CARAC_DEPT : {
        FEATURE_COLUMNNAME : 'carac_dept',
        FEATURE_PIE_ROTATION : 0,
        FEATURE_DISTRI_MIN_BINS : 5,
        FEATURE_DISTRI_DEFAULT_BINS : 5,
        FEATURE_DISTRI_MAX_BINS : 16,
        FEATURE_DISTRI_MIN_VAL : 0,
        FEATURE_DISTRI_MAX_VAL : 12,
        FEATURE_DISTRI_DISABLE : True,
        FEATURE_DESIGNATION : "Département du lieu de l'accident",
        FEATURE_MODALITE : {
        }
    },
    #AGG_CLUSTER : {
    #    FEATURE_COLUMNNAME : 'carac_cluster',
    #    FEATURE_PIE_ROTATION : 0,
    #    FEATURE_DESIGNATION : 'Clusterisation des données GPS (20 clusters au total)',
    #    FEATURE_MODALITE : {
    #    }
    #},
    CARAC_INT : {
        FEATURE_COLUMNNAME : 'carac_int',
        FEATURE_PIE_ROTATION : 0,
        FEATURE_DESIGNATION : 'Intersection (hors intersection, en Y en T, en X, giratoire, etc.)',
        FEATURE_MODALITE : {
            -1 : 'Non renseigné',
            1 : 'Hors intersection',
            2 : 'Intersection en X',
            3 : 'Intersection en T',
            4 : 'Intersection en Y',
            5 : 'Intersection à plus de 4 branches',
            6 : 'Giratoire',
            7 : 'Place',
            8 : 'Passage à niveau',
            9 : 'Autre intersection',
        }
    },
    CARAC_LUM : {
        FEATURE_COLUMNNAME : 'carac_lum',
        FEATURE_PIE_ROTATION : 0,
        FEATURE_DESIGNATION : "Luminosité au moment de l'accident (plein jour, crépuscule, nuit sans éclairage, etc.)",
        FEATURE_MODALITE : {
            -1 : ' Non renseigné',
            1 : 'Plein jour',
            2 : 'Crépuscule ou aube',
            3 : 'Nuit sans éclairage public',
            4 : 'Nuit avec éclairage public non allumé',
            5 : 'Nuit avec éclairage public allumé',
        },
        FEATURE_DEFAULT : 1
    },
    LIEU_CATR: {
        FEATURE_COLUMNNAME : 'lieu_catr',
        FEATURE_PIE_ROTATION : 30,
        FEATURE_DESIGNATION : 'Catégorie de route (Autoroute, nationale, hors réseau, etc.)',
        FEATURE_MODALITE : {
            1 : 'Autoroute',
            2 : 'Route nationale',
            3 : 'Route Départementale',
            4 : 'Voie Communales',
            5 : 'Hors réseau public',
            6 : 'Parc de stationnement ouvert à la circulation publique',
            7 : 'Routes de métropole urbaine',
            9 : 'autre',
        },
        FEATURE_DEFAULT : 3
    },
    LIEU_CIRC: {
        FEATURE_COLUMNNAME : 'lieu_circ',
        FEATURE_PIE_ROTATION : 0,
        FEATURE_DESIGNATION : 'Régime de circulation (sens unique, bidirectionnelle, à chaussées séparées, etc.)',
        FEATURE_MODALITE : {
            -1: 'Non renseigné',
            1 : 'A sens unique',
            2 : 'Bidirectionnelle',
            3 : 'A chaussées séparées',
            4 : 'Avec voies d’affectation variable',
        },
        FEATURE_DEFAULT : 2
    },
    LIEU_INFRA: {
        FEATURE_COLUMNNAME : 'lieu_infra',
        FEATURE_PIE_ROTATION : -90,
        FEATURE_DESIGNATION : "Aménagement infrastructure (souterrain, bretelle d'échangeur, voie ferrée, etc.)",
        FEATURE_MODALITE : {
            -1 : 'Non renseigné',
            0 : 'Non',
            1 : 'Oui',
            -1 : 'Non renseigné',
            0 : 'Aucun',
            1 : 'Souterrain - tunnel',
            2 : 'Pont - autopont',
            3 : 'Bretelle d’échangeur ou de raccordement',
            4 : 'Voie ferrée',
            5 : 'Carrefour aménagé',
            6 : 'Zone piétonne',
            7 : 'Zone de péage',
            8 : 'Chantier',
        }
    },
    LIEU_LARROUT: {
        FEATURE_COLUMNNAME : 'lieu_larrout',
        FEATURE_PIE_ROTATION : 0,
        FEATURE_DESIGNATION : 'Largeur de la chaussée affectée à la circulation (en m)',
        FEATURE_MODALITE : {
        }
    },
    LIEU_LARTPC: {
        FEATURE_COLUMNNAME : 'lieu_lartpc',
        FEATURE_PIE_ROTATION : 0,
        FEATURE_DESIGNATION : "Largeur du terre plein central s'il existe (en m)",
        FEATURE_MODALITE : {
        }
    },
    LIEU_NBV: {
        FEATURE_COLUMNNAME : 'lieu_nbv',
        FEATURE_PIE_ROTATION : 0,
        FEATURE_DISTRI_MIN_BINS : 4,
        FEATURE_DISTRI_DEFAULT_BINS : 6,
        FEATURE_DISTRI_MAX_BINS : 8,
        FEATURE_DISTRI_MIN_VAL : 0,
        FEATURE_DISTRI_MAX_VAL : 12,
        FEATURE_DESIGNATION : 'nombre de voie',
        FEATURE_MODALITE : {
        }
    },
    LIEU_PLAN: {
        FEATURE_COLUMNNAME : 'lieu_plan',
        FEATURE_PIE_ROTATION : 0,
        FEATURE_DESIGNATION : 'Tracé en plan (partie rectiligne en courbe à gauche, à droite, etc.)',
        FEATURE_MODALITE : {
            -1 : 'Non renseigné',
            1 : 'Partie rectiligne',
            2 : 'En courbe à gauche',
            3 : 'En courbe à droite',
            4 : 'En « S »',
        },
        FEATURE_DEFAULT : 1
    },
    LIEU_PROF: {
        FEATURE_COLUMNNAME : "lieu_prof",
        FEATURE_PIE_ROTATION : 0,
        FEATURE_DESIGNATION : "Décrit la décivité de la route à l'endroit de l'accident",
        FEATURE_MODALITE : {
            -1 : 'Non renseigné',
            1 : 'Plat',
            2 : 'Pente',
            3 : 'Sommet de côte',
            4 : 'Bas de côte',
        }
    },
    LIEU_SITU: {
        FEATURE_COLUMNNAME : 'lieu_situ',
        FEATURE_PIE_ROTATION : -30,
        FEATURE_DESIGNATION : "Situation de l'accident (sur la chaussée, sur bande d'arrêt d'irgence, etc.)",
        FEATURE_MODALITE : {
            -1 : 'Non renseigné',
            0 : 'Aucun',
            1 : 'Sur chaussée',
            2 : 'Sur bande d’arrêt d’urgence',
            3 : 'Sur accotement',
            4 : 'Sur trottoir',
            5 : 'Sur piste cyclable',
            6 : 'Sur autre voie spéciale',
            8 : 'Autres',
        },
        FEATURE_DEFAULT : 2
    },
    LIEU_SURF: {
        FEATURE_COLUMNNAME : 'lieu_surf',
        FEATURE_PIE_ROTATION : 0,
        FEATURE_DESIGNATION : 'Etat de la surface (normale, mouillée, flaques, inondée, etc.)',
        FEATURE_MODALITE : {
            -1 : 'Non renseigné',
            1 : 'Normale',
            2 : 'Mouillée',
            3 : 'Flaques',
            4 : 'Inondée',
            5 : 'Enneigée',
            6 : 'Boue',
            7 : 'Verglacée',
            8 : 'Corps gras',
        }
    },
    LIEU_VMA: {
        FEATURE_COLUMNNAME : 'lieu_vma',
        FEATURE_PIE_ROTATION : 0,
        FEATURE_DISTRI_MIN_BINS : 5,
        FEATURE_DISTRI_DEFAULT_BINS : 8,
        FEATURE_DISTRI_MAX_BINS : 16,
        FEATURE_DISTRI_MIN_VAL : 0,
        FEATURE_DISTRI_MAX_VAL : 150,
        FEATURE_DISTRI_DISABLE : False,
        FEATURE_DESIGNATION : "Vitesse maximale autorisée au lieu de l'accident (km/h)",
        FEATURE_MODALITE : {
        }
    },
    LIEU_VOIE: {
        FEATURE_COLUMNNAME : 'lieu_voie',
        FEATURE_PIE_ROTATION : 0,
        FEATURE_DISTRI_MIN_BINS : 3,
        FEATURE_DISTRI_DEFAULT_BINS : 4,
        FEATURE_DISTRI_MAX_BINS : 8,
        FEATURE_DISTRI_MIN_VAL : 0,
        FEATURE_DISTRI_MAX_VAL : 12,
        FEATURE_DISTRI_DISABLE : False,
        FEATURE_DESIGNATION : 'Numéro de la voie',
        FEATURE_MODALITE : {
        }
    },
    LIEU_VOSP: {
        FEATURE_COLUMNNAME : 'lieu_vosp',
        FEATURE_PIE_ROTATION : -30,
        FEATURE_DESIGNATION : 'Voix réservée (sans objet, piste cyclable, voie réservée)',
        FEATURE_MODALITE : {
            -1 : 'Non renseigné',
            0 : 'Sans objet',
            1 : 'Piste cyclable',
            2 : 'Bande cyclable',
            3 : 'Voie réservée',
        }
    },
    AGG_CATV_PERSO: {
        FEATURE_COLUMNNAME : 'agg_catv_perso',
        FEATURE_PIE_ROTATION : 0,
        FEATURE_DESIGNATION : 'Catégorie (regroupement en 6 modalités)',
        FEATURE_MODALITE : {
            1 : 'vélo',
            2 : 'véhicule sans permis ou < 125cm3',
            3 : 'moto > 125cm3',
            4 : 'vl (véhicule léger)',
            5 : 'pl (poids lourds y compris transports en commun)',
            6 : 'Autres (y compris trotinettes/roler à partir de 2018)',
        },
        FEATURE_DEFAULT : 3
    },
    VEHI_CATV: {
        FEATURE_COLUMNNAME : 'vehi_catv',
        FEATURE_PIE_ROTATION : 0,
        FEATURE_DESIGNATION : 'Catégorie de véhicule (très détaillée avec plus de 20 modalités)',
        FEATURE_MODALITE : {
            '-1' : 'non renseigné',
            '00' : 'Indeterminable',
            '01' : 'Bicyclette',
            '02' : 'Cyclomoteur <50cm3',
            '03' : 'Voiturette (Quadricycle à moteur carrossé) (anciennement "voiturette ou tricycle à moteur")',
            '07' : 'VL seul',
            '10' : 'VU seul 1,5T <= PTAC <= 3,5T avec ou sans remorque (anciennement VU seul 1,5T <= PTAC <=3,5T)', 
            '13' : 'PL seul 3,5T <= 7,5T', 
            '14' : 'PL seul > 7,5T', 
            '15' : 'PL > 3,5T + remorque', 
            '16' : 'Tracteur routier seul', 
            '17' : 'Tracteur routier + semi-remorque', 
            '20' : 'Engin spécial', 
            '21' : 'Tracteur agricole', 
            '30' : 'Scooter < 50 cm3', 
            '31' : 'Motocyclette > 50 cm3 et <= 125 cm3', 
            '32' : 'Scooter > 50 cm3 et <= 125 cm3', 
            '33' : 'Motocyclette > 125 cm3', 
            '34' : 'Scooter > 125 cm3', 
            '35' : 'Quad léger <= 50 cm3 (Quadricycle à moteur non carrossé)', 
            '36' : 'Quad lourd > 50 cm3 (Quadricycle à moteur non carrossé)', 
            '37' : 'Autobus', 
            '38' : 'Autocar', 
            '39' : 'Train',
            '40' : 'Tramway',
            '41' : '3RM <= 50 cm3',
            '42' : '3RM > 50 cm3 <= 125 cm3',
            '43' : '3RM > 125 cm3',
            '50' : 'EDP à moteur',
            '60' : 'EDP sans moteur',
            '80' : 'VAE',
            '99' : 'Autre véhicule (dont piéton en roller ou en trottinette à partir de l’année 2018 requalifié en "engin de déplacement personnel")'
        }
    },
    VEHI_CHOC: {
        FEATURE_COLUMNNAME : 'vehi_choc',
        FEATURE_PIE_ROTATION : 0,
        FEATURE_DESIGNATION : 'Point de choc initial (aucun, avant, arrière, etc.)',
        FEATURE_MODALITE : {
            -1 : 'non renseigné',
            0 : 'Aucun',
            1 : 'Avant',
            2 : 'Avant droit',
            3 : 'Avant gauche',
            4 : 'Arrière',
            5 : 'Arrière droit',
            6 : 'Arrière gauche',
            7 : 'Côté droit',
            8 : 'Côté gauche',
            9 : 'Chocs multiples (tonneaux)',    
        },
        FEATURE_DEFAULT : 2
    },
    VEHI_MANV: {
        FEATURE_COLUMNNAME : "vehi_manv",
        FEATURE_PIE_ROTATION : 0,
        FEATURE_DESIGNATION : 'Manoeuvre principale avant accident (sans chgt direction, même sens, même file, dans couloir bus, etc.) ',
        FEATURE_MODALITE : {
            -1 : 'non renseigné',
            0 : 'Inconnue',
            1 : 'Sans changement de direction',
            2 : 'Même sens, même file',
            3 : 'Entre 2 files',
            4 : 'En marche arrière',
            5 : 'A contresens',
            6 : 'En franchissant le terre-plein central',
            7 : 'Dans le couloir bus, dans le même sens',
            8 : 'Dans le couloir bus, dans le sens inverse',
            9 : 'En s’insérant',
            10 : 'En faisant demi-tour sur la chaussée',
            11 : 'Changeant de file à gauche',
            12 : 'Changeant de file à droite',
            13 : 'Déporté à gauche',
            14 : 'Déporté à droite',
            15 : 'Tournant à gauche',
            16 : 'Tournant à droite',
            17 : 'Dépassant à gauche',
            18 : 'Dépassant à droite',
            19 : 'Traversant la chaussée',
            20 : 'Manœuvre de stationnement',
            21 : 'Manœuvre d’évitement',
            22 : 'Ouverture de porte',
            23 : 'Arrêté (hors stationnement)',
            24 : 'En stationnement (avec occupants)',
            25 : 'Circulant sur trottoir',
            26 : 'Autres manœuvres',
        },
        FEATURE_DEFAULT : 2
    },
    VEHI_MOTOR: {
        FEATURE_COLUMNNAME : 'vehi_motor',
        FEATURE_PIE_ROTATION : -30,
        FEATURE_DESIGNATION : 'Type de motorisation (inconnue, hydrocarbure, hybride électrique, humaine, etc.)',
        FEATURE_MODALITE : {
            -1 : 'Non renseigné',
            0 : 'Inconnue',
            1 : 'Hydrocarbures',
            2 : 'Hybride électrique',
            3 : 'Electrique',
            4 : 'Hydrogène',
            5 : 'Humaine',
            6 : 'Autre',
        },
        FEATURE_DEFAULT : 2
    },
    VEHI_OBS: {
        FEATURE_COLUMNNAME : 'vehi_obs',
        FEATURE_PIE_ROTATION : -135,
        FEATURE_DESIGNATION : 'Obstacle fixe heurté (véhicule en stationnement, glissière, batiment, etc.)',
        FEATURE_MODALITE : {
            -1 : 'Non renseigné',
            0 : 'aucun',
            1 : 'Véhicule en stationnement',
            2 : 'Arbre',
            3 : 'Glissière métallique',
            4 : 'Glissière béton',
            5 : 'Autre glissière',
            6 : 'Bâtiment, mur, pile de pont',
            7 : 'Support de signalisation verticale ou poste d’appel d’urgence',
            8 : 'Poteau',
            9 : 'Mobilier urbain',
            10 : 'Parapet',
            11 : 'Ilot, refuge, borne haute',
            12 : 'Bordure de trottoir',
            13 : 'Fossé, talus, paroi rocheuse',
            14 : 'Autre obstacle fixe sur chaussée',
            15 : 'Autre obstacle fixe sur trottoir ou accotement',
            16 : 'Sortie de chaussée sans obstacle', 
            17 : "Buse - tête d'aqueduc",
        },
        FEATURE_DEFAULT : 0
    },
    VEHI_OBSM: {
        FEATURE_COLUMNNAME : 'vehi_obsm',
        FEATURE_PIE_ROTATION : 0,
        FEATURE_DESIGNATION : 'Obstacle mobile heurté (piéton, véhicule, animal, etc.)',
        FEATURE_MODALITE : {
            -1 : 'non renseigné',
            0 : 'Aucun',
            1 : 'Piéton',
            2 : 'Véhicule',
            4 : 'Véhicule sur rail',
            5 : 'Animal domestique',
            6 : 'Animal sauvage',
            9 : 'Autre',
        },
        FEATURE_DEFAULT : 3
    },
    VEHI_SENC: {
        FEATURE_COLUMNNAME : 'vehi_senc',
        FEATURE_PIE_ROTATION : 0,
        FEATURE_DESIGNATION : 'Sens de circulation (inconnu, numéro adresse postale croissant, etc.)',
        FEATURE_MODALITE : {
            -1 : 'Non renseigné',
            0 : 'inconnu',
            2 : 'PK ou PR ou numéro d’adresse postale décroissant',
            1 : 'PK ou PR ou numéro d’adresse postale croissant',
        }
    },
    USER_AN_NAIS: {
        FEATURE_COLUMNNAME : 'user_an_nais',
        FEATURE_PIE_ROTATION : 0,
        FEATURE_DISTRI_MIN_BINS : 4,
        FEATURE_DISTRI_DEFAULT_BINS : 6,
        FEATURE_DISTRI_MAX_BINS : 12,
        FEATURE_DISTRI_MIN_VAL : 1910,
        FEATURE_DISTRI_MAX_VAL : 2022,
        FEATURE_DISTRI_DISABLE : False,
        FEATURE_DESIGNATION : "Année de naissance de l'usager",
        FEATURE_MODALITE : {
        }
    },
    USER_CATU: {
        FEATURE_COLUMNNAME : 'user_catu',
        FEATURE_PIE_ROTATION : 0,
        FEATURE_DESIGNATION : "Catégorie de l'usager (Conducteur, passager, piéton, etc.)",
        FEATURE_MODALITE : {
            1 : 'Conducteur',
            2 : 'Passager',
            3 : 'Piéton',
            4 : "Piéton en roller ou trotinette (utilisée jusqu'à 2017) : à partir de 2018 info portée par vehicule.catv=99",
        },
        FEATURE_DEFAULT : 0
    },
    USER_ACTP: {
        FEATURE_COLUMNNAME : 'user_actp',
        FEATURE_PIE_ROTATION : 0,
        FEATURE_DESIGNATION : 'Action du piéton (se déplaçant (sens du véhicule, etc.), masqué, traversant, jouant, avec animal, etc.)',
        FEATURE_MODALITE : {
            '-1' : 'Non renseigné',
            '0' : 'Se déplaçant - Non renseigné ou sans objet',
            '1' : 'Se déplaçant - Sens véhicule heurtant',
            '2' : 'Se déplaçant - Sens inverse du véhicule',
            '3' : 'Traversant',
            '4' : 'Masqué',
            '5' : 'Jouant – courant',
            '6' : 'Avec animal',
            '9' : 'Autre',
            'A' : 'Monte/descend du véhicule',
            'B' : 'Inconnue',
        }
    },
    USER_ETATP: {
        FEATURE_COLUMNNAME : 'user_etatp',
        FEATURE_PIE_ROTATION : -45,
        FEATURE_DESIGNATION : 'Le piéton était-il seul, accompagné, en groupe',
        FEATURE_MODALITE : {
            -1: 'Non renseigné',
            0 : 'Sans Objet',
            1 : 'Seul',
            2 : 'Accompagné',
            3 : 'En groupe',
        }
    },
    USER_LOCP: {
        FEATURE_COLUMNNAME : 'user_locp',
        FEATURE_PIE_ROTATION : -180,
        FEATURE_DESIGNATION : "Localisation du piéton au moment de l'impact (sur trottoir, sur accotement, avec/sans signalisation lumineuse, etc.)",
        FEATURE_MODALITE : {
            -1: 'Non renseigné',
            0 : 'Sans objet',
            1 : 'Sur chaussée : à + 50 m du passage piéton',
            2 : 'Sur chaussée : à – 50 m du passage piéton',
            3 : 'Sur passage piéton sans signalisation lumineuse',
            4 : 'Sur passage piéton avec signalisation lumineuse',
            5 : 'Sur trottoir',
            6 : 'Sur accotement',
            7 : 'Sur refuge ou BAU',
            8 : 'Sur contre allée',
            9 : 'Inconnue',
        }
    },
    USER_PLACE: {
        FEATURE_COLUMNNAME : 'user_place',
        FEATURE_PIE_ROTATION : -120,
        FEATURE_DESIGNATION : 'Emplacement du conducteur dans le véhicule',
        FEATURE_MODALITE : {
            -1 : 'Non renseigné',
            0 : 'Piéton',
            1 : 'Conducteur',
            2 : 'place avant droit véhicule + 3 roue - place arrière véhicule à moins de 4 roues',
            3 : 'Place arrière droit (y compris les side-car)',
            4 : 'Place arrière gauche',
            5 : 'Place arrière centrale',
            6 : 'Place avant centre',
            7 : 'Place latérale gauche (au moins un siège devant et un siège derrière)',
            8 : 'Centre du véhicule (au moins un siège de chaque coté)',
            9 : 'Latéral droit (au moins un siège devant et au moins un derrière)',
            10: 'Piéton',
        }
    },
    USER_SECU1: {
        FEATURE_COLUMNNAME : 'user_secu1',
        FEATURE_PIE_ROTATION : -10,
        FEATURE_DESIGNATION : "Utilisation de l'équipement de sécurité no1 (sur 3)",
        FEATURE_MODALITE : {
            -1: 'Non renseigné',
            0 : 'Aucun équipement',
            1 : 'Ceinture',
            2 : 'Casque',
            3 : 'Dispositif enfants',
            4 : 'Gilet réfléchissant',
            5 : 'Airbag (2RM/3RM)',
            6 : 'Gants (2RM/3RM)',
            7 : 'Gants + Airbag (2RM/3RM)',
            8 : 'Non déterminable',
            9 : 'Autre',
        },
        FEATURE_DEFAULT : 2
    },
    USER_SECU2: {
        FEATURE_COLUMNNAME : 'user_secu2',
        FEATURE_PIE_ROTATION : 0,
        FEATURE_DESIGNATION : "Utilisation de l'équipement de sécurité no2 (sur 3)",
        FEATURE_MODALITE : {
            -1:'Non renseigné',
            0: 'Aucun équipement',
            1: 'Ceinture',
            2: 'Casque',
            3: 'Dispositif enfants',
            4: 'Gilet réfléchissant',
            5: 'Airbag (2RM/3RM)',
            6: 'Gants (2RM/3RM)',
            7: 'Gants + Airbag (2RM/3RM)',
            8: 'Non déterminable',
            9: 'Autre ',
        },
        FEATURE_DEFAULT : 0
    },
    USER_SECU3: {
        FEATURE_COLUMNNAME : 'user_secu3',
        FEATURE_PIE_ROTATION : -87,
        FEATURE_DESIGNATION : "Utilisation de l'équipement de sécurité no3 (sur 3)",
        FEATURE_MODALITE : {
            -1: 'Non renseigné',
            0 : 'Aucun équipement',
            1 : 'Ceinture',
            2 : 'Casque',
            3 : 'Dispositif enfants',
            4 : 'Gilet réfléchissant',
            5 : 'Airbag (2RM/3RM)',
            6 : 'Gants (2RM/3RM)',
            7 : 'Gants + Airbag (2RM/3RM)',
            8 : 'Non déterminable',
            9 : 'Autre ',
        },
        FEATURE_DEFAULT : 0
    },
    USER_SEXE: {
        FEATURE_COLUMNNAME : 'user_sexe',
        FEATURE_PIE_ROTATION : 0,
        FEATURE_DESIGNATION : "Genre de l'usager",
        FEATURE_MODALITE : {
            1 : 'Masculin',
            2 : 'Féminin',
        }
    },
    USER_TRAJET: {
        FEATURE_COLUMNNAME : 'user_trajet',
        FEATURE_PIE_ROTATION : 0,
        FEATURE_DESIGNATION : 'Type de trajet (domicile-travail, courses-achats, loisirs, etc...)',
        FEATURE_MODALITE : {
            -1: 'Non renseigné',
            0 : 'Non renseigné',
            1 : 'Domicile – travail',
            2 : 'Domicile – école',
            3 : 'Courses – achats',
            4 : 'Utilisation professionnelle',
            5 : 'Promenade – loisirs',
            9 : 'Autre',
        },
        FEATURE_DEFAULT : 6
    },
    USER_GRAVITE: {
        FEATURE_COLUMNNAME : 'user_gravite',
        FEATURE_PIE_ROTATION : 0,
        FEATURE_DESIGNATION : "Variable cible, gravité de l'accident. 4 modalités (indemne, blessé léger, blessé hospitalisé ou tué) ",
        FEATURE_MODALITE : {
            -1 : 'non renseigné',
            0 : 'Indemne',
            1 : 'Blessé léger',
            2 : 'Blessé hospitalisé',
            3 : 'Tué',
        }
    }
}

FEATURES_BY_TYPE = {
    FEATURE_CARACTERISTIQUE :[
        CARAC_AGG,
        CARAC_ATM,
        CARAC_COL,
        CARAC_COM,
        CARAC_DEPT,
        CARAC_INT, 
        CARAC_LUM
    ],
    FEATURE_LIEU : [
        LIEU_CATR,
        LIEU_CIRC,
        LIEU_INFRA,
        LIEU_LARROUT,
        LIEU_LARTPC,
        LIEU_NBV,
        LIEU_PLAN,
        LIEU_PROF,
        LIEU_SITU,
        LIEU_SURF,
        LIEU_VMA,
        LIEU_VOIE,
        LIEU_VOSP
    ],
    FEATURE_VEHICULE : [
        VEHI_CATV,
        VEHI_CHOC,
        VEHI_MANV,
        VEHI_MOTOR,
        VEHI_OBS,
        VEHI_OBSM,
        VEHI_SENC
    ],
    FEATURE_USAGER : [
        USER_AN_NAIS,
        USER_CATU,
        USER_ACTP,
        USER_ETATP,
        USER_LOCP,
        USER_PLACE,
        USER_SECU1,
        USER_SECU2,
        USER_SECU3,
        USER_SEXE,
        USER_TRAJET,
        USER_GRAVITE
    ]
}
 
LISTE_COLUMNS = ['carac_agg', 'carac_atm', 'carac_col', 'carac_lum', 'carac_gps_lat', 'carac_gps_long',
                 'lieu_catr', 'lieu_circ', 'lieu_plan', 'lieu_situ',
                 'agg_catv_perso', 'vehi_choc', 'vehi_manv', 'vehi_motor', 'vehi_obsm', 'vehi_obs', 
                 'user_catu', 'user_secu1', 'user_secu2', 'user_secu3', 'user_trajet']
LISTE_GRAVITE = ['Indemne', 'Blessé léger', 'Blessé hospitalisé', 'Tué']
LISTE_GRAVITE_BINAIRE = ['Indemne/Blessé léger', 'Blessé hospitalisé/Tué']

