import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

file = 'data/courbes-de-charges-fictives-res2-6-9.csv'

def load_and_feature_engineering(file):
    df = pd.read_csv(file, sep=',') 
    
    # 1. On calcule des stats par maison (ID)
    df_features = df.groupby('ID').agg({
        'valeur': [
            'mean',    # Consommation moyenne
            'std',     # Variabilité (très élevé pour les résidences secondaires)
            'max',     # Puissance de pointe
            'min'      # Consommation de fond (talon)
        ]
    })
    
    # Nettoyage des noms de colonnes
    df_features.columns = ['mean', 'std', 'max', 'min']
    return df_features

def perform_clustering(df_features, n_clusters=2):
    # 1. Normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_features)
    
    # 2. Clustering
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
    df_features['Cluster'] = kmeans.fit_predict(X_scaled) 
    
    # 3. Mappage des types (on trie par consommation moyenne)
    avg_cons = df_features.groupby('Cluster')['mean'].mean().sort_values()
    res_secondaire_id = avg_cons.index[0]
    
    df_features['Type_Residence'] = df_features['Cluster'].apply(
        lambda x: 'Secondaire' if x == res_secondaire_id else 'Principale'
    )
    
    return df_features

def plot_clusters(resultat):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=resultat, 
        x='mean', 
        y='std', 
        hue='Type_Residence', 
        palette='viridis',
        alpha=0.7
    )
    plt.title('Segmentation des Résidences : Moyenne vs Variabilité')
    plt.xlabel('Consommation moyenne (W)')
    plt.ylabel('Écart-type / Variabilité (W)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('segmentation_residences.png')
    print("\nGraphique sauvegardé sous 'segmentation_residences.png'")


if __name__ == "__main__":
    features = load_and_feature_engineering(file)
    resultat = perform_clustering(features, n_clusters=2)
    print(resultat.head())
    print("\nAnalyse des groupes :")
    print(resultat.groupby('Type_Residence')['mean'].mean())
    
    # Appel de la visualisation
    plot_clusters(resultat)