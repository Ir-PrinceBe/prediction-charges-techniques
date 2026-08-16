# =====================================================
# IMPORTATION DES BIBLIOTHÈQUES
# =====================================================

# Streamlit sert à créer l'interface web
import streamlit as st

# Joblib permet de charger le modèle Machine Learning sauvegardé
import joblib

# Pandas permet de créer et manipuler les tableaux de données
import pandas as pd

# Matplotlib permet de créer les graphiques
import matplotlib.pyplot as plt


# =====================================================
# CONFIGURATION GÉNÉRALE DE LA PAGE
# =====================================================

# Cette instruction définit le titre de la page,
# l'icône affichée dans l'onglet du navigateur
# et l'utilisation d'une largeur étendue.
st.set_page_config(
    page_title="Prévision des charges techniques",
    page_icon="📊",
    layout="wide"
)


# =====================================================
# CHARGEMENT DU MODÈLE MACHINE LEARNING
# =====================================================

# @st.cache_resource permet à Streamlit de charger le modèle
# une seule fois au lieu de le recharger à chaque interaction.
@st.cache_resource
def charger_modele():

    # Chargement du modèle Random Forest sauvegardé avec joblib
    return joblib.load("modele_random_forest.pkl")


# Appel de la fonction afin de récupérer le modèle
model = charger_modele()


# =====================================================
# VARIABLES UTILISÉES PAR LE MODÈLE
# =====================================================

# Cette liste contient exactement les variables utilisées
# pendant l'entraînement du Random Forest.
# L'ordre des colonnes doit rester identique.
features = [
    'Employeurs_affilies',
    'Beneficiaire_pension',
    'Salaries_declares',
    'Total_produit(en Million)',
    'Rentiers',
    'Inflation',
    'Croissance_PIB',
    'Esperance_Vie'
]


# =====================================================
# BARRE LATÉRALE DE NAVIGATION
# =====================================================

# Titre du menu placé dans la barre latérale
st.sidebar.title("📊 Menu")

# Création du menu principal de navigation.
# Une seule page sera affichée à la fois.
page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Accueil",
        "🔮 Prédiction personnalisée",
        "📈 Prévisions 1 à 5 ans",
        "📊 Performance du modèle",
        "ℹ️ À propos"
    ]
)

# Ajout d'une séparation visuelle
st.sidebar.divider()

# Petit texte explicatif affiché en bas du menu
st.sidebar.caption(
    "Application de Machine Learning pour "
    "la prévision des charges techniques."
)


# =====================================================
# PAGE 1 : ACCUEIL
# =====================================================

# Ce bloc est exécuté uniquement lorsque
# l'utilisateur sélectionne la page Accueil.
if page == "🏠 Accueil":

    # Titre principal de l'application
    st.title("📊 Prévision des charges techniques")

    # Sous-titre
    st.subheader(
        "Application de Machine Learning pour l'aide "
        "à la prévision des charges"
    )

    # Présentation générale de l'application
    st.write(
        """
        Cette application exploite un modèle de Machine Learning
        afin d'estimer les charges techniques d'un organisme
        de sécurité sociale à partir d'indicateurs
        socio-économiques et opérationnels.
        """
    )

    st.divider()

    # Présentation des principales fonctionnalités
    st.subheader("⚙️ Fonctionnalités")

    # Création de trois colonnes pour présenter
    # les trois principales fonctionnalités
    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(
            """
            ### 🔮 Prédiction personnalisée

            Saisissez manuellement les valeurs des indicateurs
            afin d'obtenir une estimation de la charge technique.
            """
        )

    with col2:
        st.info(
            """
            ### 📈 Prévision pluriannuelle

            Simulez les charges techniques sur un horizon
            allant de 1 à 5 ans.
            """
        )

    with col3:
        st.info(
            """
            ### 📊 Performance

            Consultez les résultats obtenus par les
            différents modèles de Machine Learning.
            """
        )

    st.divider()

    # Présentation du modèle retenu
    st.subheader("🤖 Modèle retenu")

    col1, col2, col3 = st.columns(3)

    # Affichage du nom du meilleur modèle
    col1.metric(
        "Modèle",
        "Random Forest"
    )

    # Affichage du R² obtenu sur le jeu test
    col2.metric(
        "R² Test",
        "97,26 %"
    )

    # Affichage du R² moyen obtenu en validation croisée
    col3.metric(
        "R² moyen K-Fold",
        "97,52 %"
    )

    st.success(
        "Random Forest a été retenu comme modèle final "
        "en raison de ses performances prédictives et "
        "de sa stabilité en validation croisée."
    )


# =====================================================
# PAGE 2 : PRÉDICTION PERSONNALISÉE
# =====================================================

elif page == "🔮 Prédiction personnalisée":

    st.title("🔮 Prédiction personnalisée")

    st.write(
        """
        Renseignez les huit indicateurs ci-dessous.
        Le modèle Random Forest estimera automatiquement
        la charge technique correspondante.
        """
    )

    st.divider()

    # Les champs de saisie sont répartis en deux colonnes
    # pour rendre l'interface plus lisible.
    col1, col2 = st.columns(2)

    # Première colonne : indicateurs principalement opérationnels
    with col1:

        employeurs = st.number_input(
            "Employeurs affiliés",
            min_value=0,
            value=4500
        )

        beneficiaires = st.number_input(
            "Bénéficiaires de pension",
            min_value=0,
            value=6500
        )

        salaries = st.number_input(
            "Salariés déclarés",
            min_value=0,
            value=55000
        )

        total_produit = st.number_input(
            "Total produit (Millions FC)",
            min_value=0.0,
            value=18000.0
        )

    # Deuxième colonne : autres indicateurs
    with col2:

        rentiers = st.number_input(
            "Rentiers",
            min_value=0,
            value=300
        )

        inflation = st.number_input(
            "Inflation",
            value=0.8
        )

        croissance_pib = st.number_input(
            "Croissance du PIB",
            value=0.5
        )

        esperance_vie = st.number_input(
            "Espérance de vie",
            min_value=0.0,
            value=60.5
        )

    st.divider()

    # Le calcul de la prédiction ne sera exécuté
    # que lorsque l'utilisateur clique sur ce bouton.
    if st.button(
        "🔮 Prédire la charge technique",
        use_container_width=True
    ):

        # Création d'un DataFrame contenant une seule observation.
        # Les noms de colonnes doivent être exactement identiques
        # à ceux utilisés lors de l'entraînement.
        nouvelle_donnee = pd.DataFrame({
            'Employeurs_affilies': [employeurs],
            'Beneficiaire_pension': [beneficiaires],
            'Salaries_declares': [salaries],
            'Total_produit(en Million)': [total_produit],
            'Rentiers': [rentiers],
            'Inflation': [inflation],
            'Croissance_PIB': [croissance_pib],
            'Esperance_Vie': [esperance_vie]
        })

        # Le modèle calcule la charge technique prédite
        prediction = model.predict(
            nouvelle_donnee
        )[0]

        st.subheader("Résultat de la prédiction")

        # Affichage du résultat sous deux formes :
        # millions FC et milliards FC.
        col1, col2 = st.columns(2)

        col1.metric(
            "Charge prédite",
            f"{prediction:,.2f} Millions FC"
        )

        col2.metric(
            "Équivalent",
            f"{prediction / 1000:,.2f} Milliards FC"
        )

        st.success(
            "La prédiction a été réalisée avec succès."
        )


# =====================================================
# PAGE 3 : PRÉVISIONS DE 1 À 5 ANS
# =====================================================

elif page == "📈 Prévisions 1 à 5 ans":

    st.title("📈 Prévision pluriannuelle")

    st.write(
        """
        Cette section permet de simuler les charges techniques
        sur un horizon allant de 1 à 5 ans.

        Les valeurs proposées constituent un scénario
        tendanciel et peuvent être modifiées par l'utilisateur.
        """
    )

    st.divider()

    # L'utilisateur choisit le nombre d'années
    # qu'il souhaite afficher : de 1 à 5 ans.
    horizon = st.slider(
        "Nombre d'années à prévoir",
        min_value=1,
        max_value=5,
        value=5
    )

    # Tableau contenant les valeurs projetées
    # des huit variables pour les années 2025 à 2029.
    donnees_futures = pd.DataFrame({

        'Année': [
            2025,
            2026, 
            2027,
            2028,
            2029
        ],

        'Employeurs_affilies': [
            6505.06,
            6819.36,
            7133.67,
            7447.97,
            7762.28
        ],

        'Beneficiaire_pension': [
            5866.61,
            5900.03,
            5933.45,
            5966.86,
            6000.28
        ],

        'Salaries_declares': [
            59048.34,
            60393.25,
            61738.17,
            63083.09,
            64428.00
        ],

        'Total_produit(en Million)': [
            28555.13,
            30179.42,
            31803.71,
            33427.99,
            35052.28
        ],

        'Rentiers': [
            232.40,
            229.80,
            227.20,
            224.60,
            222.01
        ],

        'Inflation': [
            0.4626,
            0.4495,
            0.4365,
            0.4234,
            0.4103
        ],

        'Croissance_PIB': [
            0.4682,
            0.4675,
            0.4668,
            0.4660,
            0.4653
        ],

        'Esperance_Vie': [
            62.45,
            62.86,
            63.26,
            63.66,
            64.06
        ]
    })

    # On conserve uniquement le nombre d'années
    # choisi par l'utilisateur.
    donnees_futures = (
        donnees_futures
        .head(horizon)
        .copy()
    )

    st.subheader(
        "📝 Modifier les indicateurs futurs"
    )

    st.info(
        "Vous pouvez modifier directement les valeurs "
        "du tableau afin de tester différents scénarios."
    )

    # data_editor rend le tableau interactif :
    # l'utilisateur peut modifier les valeurs
    # avant le calcul des prévisions.
    donnees_modifiees = st.data_editor(
        donnees_futures,
        hide_index=True,
        use_container_width=True
    )

    # Sélection uniquement des huit variables
    # attendues par le modèle.
    X_futur = donnees_modifiees[
        features
    ]

    # Calcul des charges techniques futures
    predictions = model.predict(
        X_futur
    )

    # Ajout des prédictions dans le tableau
    donnees_modifiees[
        'Charge_predite'
    ] = predictions

    # Création d'un tableau simplifié
    # contenant seulement l'année et la charge prédite.
    resultats = donnees_modifiees[
        [
            'Année',
            'Charge_predite'
        ]
    ].copy()

    # Arrondissement des résultats à deux décimales
    resultats[
        'Charge_predite'
    ] = resultats[
        'Charge_predite'
    ].round(2)

    # Renommage de la colonne pour l'affichage
    resultats.rename(
        columns={
            'Charge_predite':
            'Charge prédite (Millions FC)'
        },
        inplace=True
    )

    st.divider()

    st.subheader(
        "📋 Résultats des prévisions"
    )

    # Affichage du tableau final des prévisions
    st.dataframe(
        resultats,
        hide_index=True,
        use_container_width=True
    )

    st.subheader(
        "📈 Évolution prévisionnelle"
    )

    # Création du graphique d'évolution
    fig, ax = plt.subplots(
        figsize=(9, 4.5)
    )

    ax.plot(
        resultats['Année'],
        resultats[
            'Charge prédite (Millions FC)'
        ],
        marker='o',
        linewidth=2
    )

    ax.set_xlabel(
        "Année"
    )

    ax.set_ylabel(
        "Charge prédite (Millions FC)"
    )

    ax.set_title(
        "Prévision des charges techniques"
    )

    # Afficher uniquement les années entières
    ax.set_xticks(
        resultats['Année']
    )

    # Ajout d'une grille légère
    ax.grid(
        alpha=0.3
    )

    # Affichage du graphique dans Streamlit
    st.pyplot(fig)

    # Précision méthodologique importante
    st.warning(
        """
        Ces prévisions correspondent à des scénarios construits
        à partir des valeurs projetées des indicateurs.
        Elles ne constituent pas des prévisions institutionnelles
        officielles.
        """
    )


# =====================================================
# PAGE 4 : PERFORMANCE DES MODÈLES
# =====================================================

elif page == "📊 Performance du modèle":

    st.title(
        "📊 Performance des modèles"
    )

    st.write(
        """
        Quatre modèles de régression ont été comparés :
        Régression Linéaire Multiple, Random Forest,
        XGBoost et LightGBM.
        """
    )

    st.divider()

    # Tableau contenant les performances obtenues
    # lors de l'évaluation des quatre modèles.
        # Tableau contenant les performances obtenues
    # lors de l'évaluation des quatre modèles
    performances = pd.DataFrame({

        'Modèle': [
            'Régression Linéaire',
            'Random Forest',
            'XGBoost',
            'LightGBM'
        ],

        'R² Test (%)': [
            59.60,
            97.26,
            96.76,
            96.63
        ],

        'MAE (Millions FC)': [
            875.91,
            179.61,
            208.26,
            211.47
        ],

        'RMSE (Millions FC)': [
            1214.95,
            316.68,
            344.20,
            350.87
        ],

        'R² K-Fold (%)': [
            61.16,
            97.52,
            97.00,
            97.06
        ]

    })
    st.subheader(
        "Résultats comparatifs"
    )

    st.dataframe(
        performances,
        hide_index=True,
        use_container_width=True
    )

    st.divider()

    # Mise en évidence du meilleur modèle
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Meilleur modèle",
        "Random Forest"
    )

    col2.metric(
        "R² Test",
        "97,26 %"
    )

    col3.metric(
        "R² K-Fold",
        "97,52 %"
    )

    st.subheader(
        "Comparaison du R²"
    )

    # Graphique comparatif des performances
    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    ax.bar(
        performances['Modèle'],
        performances['R² Test (%)']
    )

    ax.set_ylabel(
        "R² (%)"
    )

    ax.set_title(
        "Performance des modèles"
    )

    ax.tick_params(
        axis='x',
        rotation=20
    )

    st.pyplot(fig)

    st.success(
        """
        Random Forest a été retenu comme modèle final,
        car il présente le meilleur R² ainsi que
        les erreurs MAE et RMSE les plus faibles.
        """
    )


# =====================================================
# PAGE 5 : À PROPOS
# =====================================================

elif page == "ℹ️ À propos":

    st.title(
        "ℹ️ À propos de l'application"
    )

    st.write(
        """
        ### Objectif

        Cette application a été développée dans le cadre
        d'un projet tutoré portant sur l'application du
        Machine Learning à la prédiction des charges
        techniques d'un organisme de sécurité sociale.

        ### Modèle utilisé

        Le modèle final est un **Random Forest Regressor**.

        Les variables utilisées sont :

        - Employeurs affiliés
        - Bénéficiaires de pension
        - Salariés déclarés
        - Total des produits
        - Rentiers
        - Inflation
        - Croissance du PIB
        - Espérance de vie

        ### Performances obtenues

        - R² Test : **97,26 %**
        - MAE : **179,61 Millions FC**
        - RMSE : **316,68 Millions FC**
        - R² moyen K-Fold : **97,52 %**
        - Écart-type K-Fold : **0,30 %**

        ### Limite importante

        Le modèle a été développé à partir d'un dataset
        synthétique généré sur la base d'observations
        historiques.

        Les résultats doivent donc être interprétés comme
        une **preuve de concept expérimentale** et non comme
        des prévisions institutionnelles officielles.
        """
    )

    st.divider()

    st.caption(
        "Projet tutoré — Machine Learning & Sécurité sociale"
    )
