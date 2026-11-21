"""打音の判定器（あるか・ドンかパンか）を学習する"""
import json, numpy as np
import importlib.util
spec = importlib.util.spec_from_file_location('tn', '/home/claude/harness/train_note2_lib.py')
tn = importlib.util.module_from_spec(spec); spec.loader.exec_module(tn)

rows = json.load(open('/home/claude/harness/perc_feats.json'))
durp = {}
for r in rows: durp[r['n']] = max(durp.get(r['n'], 0), r['time'])
for r in rows:
    n = r['n']
    if n.startswith('user_'): r['sp'] = 'utest' if r['time'] > 0.7 * durp[n] else 'train'
    else: r['sp'] = 'test' if not (n.startswith('rnd') or n.startswith('mrh')) else ('val' if int(n[3:]) % 6 == 5 else 'train')
A = lambda R: (np.array([r['f'] for r in R], float), np.array([r['y'] for r in R], float))
tr = [r for r in rows if r['sp'] == 'train']; va = [r for r in rows if r['sp'] == 'val']; te = [r for r in rows if r['sp'] == 'test']
ut = [r for r in rows if r['sp'] == 'utest']
rep = np.array([4 if r['n'].startswith('user_') else 1 for r in tr])
X, y = A(tr); X = np.repeat(X, rep, axis=0); y = np.repeat(y, rep); Xv, yv = A(va); Xe, ye = A(te)
net, pred, fv, th = tn.fit(X, y, Xv, yv, H=12, epochs=150, l2=1e-3)
print('打音 val F1 %.3f P %.3f R %.3f th %.2f' % (*fv, th), '| test F1 %.3f P %.3f R %.3f' % tn.f1(pred(Xe), ye, th))
for un in sorted(set(r['n'] for r in ut)):
    U = [r for r in ut if r['n'] == un]
    if sum(r['y'] for r in U) or True:
        Xu, yu = A(U); print('  あなたの録音（後ろ3割）%-18s F1 %.3f P %.3f R %.3f  (正解 %d)' % ((un,) + tn.f1(pred(Xu), yu, th) + (int(yu.sum()),)))
# ドン／パン（正例だけで学習。ロジスティック回帰）
pos = [r for r in tr + va if r['y'] == 1 and r['lo'] >= 0]
Xl = np.array([r['f'] for r in pos], float); yl = np.array([r['lo'] for r in pos], float)
mu = Xl.mean(0); sd = Xl.std(0) + 1e-6; Z = (Xl - mu) / sd
w = np.zeros(Z.shape[1]); b = 0.0
for it in range(3000):
    p = 1 / (1 + np.exp(-(Z @ w + b))); g = p - yl
    w -= 0.05 * (Z.T @ g / len(yl) + 1e-3 * w); b -= 0.05 * g.mean()
acc = ((1 / (1 + np.exp(-(Z @ w + b))) >= 0.5) == (yl == 1)).mean()
pe = [r for r in te if r['y'] == 1 and r['lo'] >= 0]
if pe:
    Ze = (np.array([r['f'] for r in pe], float) - mu) / sd
    acce = ((1 / (1 + np.exp(-(Ze @ w + b))) >= 0.5) == (np.array([r['lo'] for r in pe]) == 1)).mean()
else:
    acce = float('nan')
print('ドン/パン 学習 正解率 %.3f | test %.3f' % (acc, acce))
net['lohi'] = {'mu': mu.round(5).tolist(), 'sd': sd.round(5).tolist(), 'w': w.round(5).tolist(), 'b': round(float(b), 5)}
nets = json.load(open('/home/claude/harness/nets.json'))
nets['perc'] = net
json.dump(nets, open('/home/claude/harness/nets.json', 'w'))
