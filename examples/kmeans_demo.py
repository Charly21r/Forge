import numpy as np
from forge.clustering import KMeans

def main():
    # Seed for reproducibility
    np.random.seed(42)

    # Generate random 2D data
    X = np.random.randn(100, 2)

    # Initialize KMeans with 5 clusters
    model = KMeans(n_clusters=5, random_state=42)
    model.fit(X)

    # Predict cluster labels
    preds = model.predict(X)

    # Compute negative inertia (score)
    score = model.score(X)

    # Print results
    print("Cluster assignments:", preds)
    print(f"Negative inertia (score): {score:.3f}")
    print("Centroids:\n", model.centroids_)

if __name__ == "__main__":
    main()