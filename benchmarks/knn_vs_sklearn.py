import time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from forge.neighbors import KNNClassifier


def run():
    X = np.random.randn(3000, 10)
    y = np.random.randint(0, 3, 3000)

    start = time.time()
    my = KNNClassifier(k=5).fit(X, y)
    my.predict(X)
    my_time = time.time() - start

    start = time.time()
    sk = KNeighborsClassifier(n_neighbors=5).fit(X, y)
    sk.predict(X)
    sk_time = time.time() - start

    print(f"My KNN: {my_time:.4f}s")
    print(f"Sklearn KNN: {sk_time:.4f}s")


if __name__ == "__main__":
    run()