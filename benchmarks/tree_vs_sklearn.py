import time
import numpy as np
from sklearn.tree import DecisionTreeClassifier as SkTree
from forge.tree import DecisionTreeClassifier


def run():
    X = np.random.randn(2000, 10)
    y = np.random.randint(0, 3, 2000)

    start = time.time()
    my = DecisionTreeClassifier(max_depth=10).fit(X, y)
    my.predict(X)
    my_time = time.time() - start

    start = time.time()
    sk = SkTree(max_depth=10).fit(X, y)
    sk.predict(X)
    sk_time = time.time() - start

    print(f"My Tree: {my_time:.4f}s")
    print(f"Sklearn Tree: {sk_time:.4f}s")


if __name__ == "__main__":
    run()