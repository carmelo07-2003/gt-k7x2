import numpy as np

def fit(X, y, Xv, yv, H=24, epochs=60, l2=1e-4, lr=0.003, seed=0):
    rng = np.random.default_rng(seed)
    mu = X.mean(0); sd = X.std(0) + 1e-6
    Xt = (X - mu) / sd; Xvv = (Xv - mu) / sd
    d = Xt.shape[1]
    P = [rng.normal(0, np.sqrt(2 / d), (H, d)), np.zeros(H), rng.normal(0, np.sqrt(1 / H), (1, H)), np.zeros(1)]
    M = [np.zeros_like(p) for p in P]; V = [np.zeros_like(p) for p in P]
    pw = min((1 - y.mean()) / y.mean(), 6.0)

    def fwd(A, P):
        h = np.maximum(0, A @ P[0].T + P[1]); return h, (h @ P[2].T + P[3])[:, 0]

    best = (-1, None)
    for ep in range(epochs):
        idx = rng.permutation(len(Xt))
        for s in range(0, len(idx), 256):
            bi = idx[s:s + 256]; Xb, yb = Xt[bi], y[bi]
            h, z = fwd(Xb, P); p = 1 / (1 + np.exp(-z))
            g = (p - yb) * np.where(yb == 1, pw, 1.0) / len(bi)
            G = [None, None, g[None, :] @ h + l2 * P[2], g.sum(keepdims=True)]
            gh = g[:, None] * P[2] * (h > 0)
            G[0] = gh.T @ Xb + l2 * P[0]; G[1] = gh.sum(0)
            for i in range(4):
                M[i] = 0.9 * M[i] + 0.1 * G[i]; V[i] = 0.999 * V[i] + 0.001 * G[i] ** 2
                P[i] -= lr * M[i] / (np.sqrt(V[i]) + 1e-8)
        pv = 1 / (1 + np.exp(-fwd(Xvv, P)[1]))
        f = max(f1(pv, yv, th)[0] for th in np.arange(0.1, 0.95, 0.02))
        if f > best[0]:
            best = (f, [p.copy() for p in P])
    P = best[1]
    pred = lambda A: 1 / (1 + np.exp(-fwd((A - mu) / sd, P)[1]))
    pv = pred(Xv)
    ths = np.arange(0.1, 0.95, 0.02); fs = [f1(pv, yv, t) for t in ths]; bi = int(np.argmax([q[0] for q in fs]))
    net = {'mu': mu.round(5).tolist(), 'sd': sd.round(5).tolist(), 'th': round(float(ths[bi]), 3),
           'layers': [{'W': P[0].round(5).tolist(), 'b': P[1].round(5).tolist(), 'act': 'relu'},
                      {'W': P[2].round(5).tolist(), 'b': P[3].round(5).tolist(), 'act': 'lin'}]}
    return net, pred, fs[bi], ths[bi]


def f1(p, yy, th):
    tp = ((p >= th) & (yy == 1)).sum(); fp = ((p >= th) & (yy == 0)).sum(); fn = ((p < th) & (yy == 1)).sum()
    pr = tp / max(1, tp + fp); rc = tp / max(1, tp + fn)
    return 2 * pr * rc / max(1e-9, pr + rc), pr, rc


