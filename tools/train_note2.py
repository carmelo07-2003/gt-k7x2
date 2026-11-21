"""音の判定器（1段目＋2段目）を学習する。2段目は1段目の結果（まわりの状況）も使う。"""
import json, numpy as np
from collections import defaultdict

rows = json.load(open('/home/claude/harness/note_feats.json'))


dur = {}
for r in rows: dur[r['n']] = max(dur.get(r['n'], 0), r['t'])


def split_name(r):
    n = r['n']
    if n.startswith('user_'):
        return 'utest' if r['t'] > 0.7 * dur[n] else 'train'
    if not (n.startswith('rnd') or n.startswith('mrh')):
        return 'test'
    return 'val' if int(n[3:]) % 6 == 5 else 'train'


for r in rows:
    r['sp'] = split_name(r)
rng = np.random.default_rng(0)


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


def arr(R):
    return np.array([r['f'] for r in R], float), np.array([r['y'] for r in R], float)


tr = [r for r in rows if r['sp'] == 'train']; va = [r for r in rows if r['sp'] == 'val']; te = [r for r in rows if r['sp'] == 'test']; ut = [r for r in rows if r['sp'] == 'utest']
UW = 4  # あなたの録音の重み
import os
Xtr, ytr = arr(tr); Xva, yva = arr(va); Xte, yte = arr(te); Xut, yut = arr(ut)
# 響きの途中の拾い直し（直前1.5秒に同じ高さの正解がある偽物）を重く学習する
RW = int(os.environ.get('RW', '1'))
_gt = {}
def ringing(r):
    if r['y']: return False
    n = r['n']
    if n not in _gt:
        try: _gt[n] = json.load(open(f'/home/claude/harness/data/{n}.json'))['notes']
        except Exception: _gt[n] = []
    t = r['t'] * 256 / 22050 - 0.009; m = r['k'] + 21
    return any(g['m'] == m and 0.06 < t - g['t'] < 1.5 for g in _gt[n])
rep = np.array([(UW if r['n'].startswith('user_') else 1) * (RW if (RW > 1 and ringing(r)) else 1) for r in tr])
OS = lambda X: np.repeat(X, rep, axis=0)
# 1段目：学習データには「2分割で、自分を学習に使っていない判定器」の結果を付ける（本番と同じ条件にするため）
pieces = sorted(set(r['n'] for r in tr)); fold = {n: i % 2 for i, n in enumerate(pieces)}
p1 = np.zeros(len(tr))
for k in (0, 1):
    a = np.array([fold[r['n']] != k for r in tr]); b = ~a
    _, pred, _, _ = fit(np.repeat(Xtr[a], rep[a], axis=0), np.repeat(ytr[a], rep[a]), Xva, yva, seed=k + 1)
    p1[b] = pred(Xtr[b])
import os
SEED = int(os.environ.get('SEED', '0'))
net1, pred1, fv, th1 = fit(OS(Xtr), OS(ytr), Xva, yva, seed=SEED)
print('1段目 val F1 %.3f P %.3f R %.3f th %.2f' % (*fv, th1))
print('1段目 test F1 %.3f P %.3f R %.3f' % f1(pred1(Xte), yte, th1))
for r, v in zip(tr, p1): r['p1'] = v
for r, v in zip(va, pred1(Xva)): r['p1'] = v
for r, v in zip(te, pred1(Xte)): r['p1'] = v
for r, v in zip(ut, pred1(Xut)): r['p1'] = v


# まわりの状況（engine.js の contextFeatures と同じ計算）
def context(R):
    by = defaultdict(list)
    for r in R: by[r['n']].append(r)
    for n, L in by.items():
        byk = defaultdict(list)
        for r in L: byk[r['k']].append(r)
        ts = np.array([r['t'] for r in L]); ps = np.array([r['p1'] for r in L])
        for r in L:
            def near(dk, r0, r1):
                v = 0.0
                for q in byk.get(r['k'] + dk, []):
                    d = q['t'] - r['t']
                    if r0 <= d <= r1 and q is not r: v = max(v, q['p1'])
                return v
            m = (np.abs(ts - r['t']) <= 3) & (ps > 0.5)
            cnt = int(m.sum()) - (1 if r['p1'] > 0.5 else 0)
            r['ctx'] = [r['p1'], near(-12, -2, 2), near(12, -2, 2), near(-19, -2, 2), near(-24, -2, 2), float(np.log(1 + cnt)),
                        near(0, -15, -3), near(-1, -2, 2), near(1, -2, 2),
                        near(0, -40, -16), near(-12, -40, -3), near(12, -40, -3), near(-12, 3, 40), near(0, 3, 40)]


for R in (tr, va, te, ut): context(R)
A = lambda R: np.array([r['f'] + r['ctx'] for r in R], float)
net2, pred2, fv2, th2 = fit(OS(A(tr)), OS(ytr), A(va), yva, H=32, epochs=70, seed=SEED + 7)
print('2段目 val F1 %.3f P %.3f R %.3f th %.2f' % (*fv2, th2))
print('2段目 test F1 %.3f P %.3f R %.3f' % f1(pred2(A(te)), yte, th2))
for un in sorted(set(r['n'] for r in ut)):
    U = [r for r in ut if r['n'] == un]
    if sum(r['y'] for r in U): print('  あなたの録音（後ろ3割）%-18s F1 %.3f P %.3f R %.3f' % ((un,) + f1(pred2(A(U)), np.array([r['y'] for r in U], float), th2)))
try:
    nets = json.load(open('/home/claude/harness/nets.json'))
except Exception:
    nets = {}
nets['note'] = net1; nets['note2'] = net2
json.dump(nets, open(os.environ.get('OUT', '/home/claude/harness/nets.json'), 'w'))
