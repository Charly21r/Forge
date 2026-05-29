import time
import numpy as np
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from forge.linear import LogisticRegression


def run():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((3000, 10))
    true_w = rng.standard_normal(10)
    y = (X @ true_w + 0.5 > 0).astype(int)

    start = time.time()
    my = LogisticRegression(max_iter=1000).fit(X, y)
    my_pred = my.predict(X)
    my_time = time.time() - start
    my_acc = float(np.mean(my_pred == y))

    start = time.time()
    sk = SkLogisticRegression(max_iter=1000).fit(X, y)
    sk_pred = sk.predict(X)
    sk_time = time.time() - start
    sk_acc = float(np.mean(sk_pred == y))

    print(f"My LogReg:      {my_time:.4f}s  acc={my_acc:.4f}")
    print(f"Sklearn LogReg: {sk_time:.4f}s  acc={sk_acc:.4f}")


if __name__ == "__main__":
    run()
