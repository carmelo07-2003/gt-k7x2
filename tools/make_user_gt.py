"""あなたの録音の正解データ（音・打音・学習から外す時刻）を作る"""
import numpy as np, json, sys
sys.path.insert(0, '/home/claude/harness')
from label_user import load, onsets, SR
D = '/home/claude/harness/data'
hz = lambda m: 440 * 2 ** ((m - 69) / 12)
OPEN = {1: 64, 2: 59, 3: 54, 4: 49, 5: 45, 6: 40}  # あなたのチューニング


def E(x, t0, t1, fr):
    a = max(0, int(t0 * SR)); b = min(len(x), int(t1 * SR)); seg = x[a:b]
    if len(seg) < 256: return 1e-12
    seg = seg * np.hanning(len(seg)); k = np.arange(len(seg)); v = 0
    for c in (-30, 0, 30):
        f = fr * 2 ** (c / 1200); v = max(v, abs(np.dot(seg, np.exp(-2j * np.pi * f * k / SR))) ** 2)
    return v + 1e-12


def save(n, notes, perc, ignore):
    json.dump({'notes': notes, 'perc': perc, 'ignore': ignore, 'user': True}, open(f'{D}/user_{n}.json', 'w'), ensure_ascii=False)
    print(f'{n}: 音 {len(notes)} / 打音 {len(perc)} / 学習から外す {len(ignore)}')


# 1) 打音だけの録音：見つけた出来事はすべて打音（肘＝ドン、指と手＝種類は混在なので不明）
for n, lo in (('elbow1', True), ('hand1', None)):
    x = load(n); ts = onsets(x, min_gap=0.1, rel=0.06)
    save(n, [], [{'t': round(t, 3), 'lo': lo} for t in ts], [])

# 2) ハーモニクス：自動判定の結果から
L = json.load(open(f'{D}/user_harm_labels.json'))
for n, ev in L.items():
    notes, ign = [], []
    for e in ev:
        if e['label'] == 'はっきりしない':
            ign.append(e['t']); continue
        s = e['string']; m = e['m']
        if e['label'] == 'ハーモニクス成功':
            notes.append({'t': e['t'], 's': s - 1, 'f': 7 if m - OPEN[s] == 19 else 12, 'm': m, 'h': True})
        elif e['label'] == '押さえた音（失敗）':
            notes.append({'t': e['t'], 's': s - 1, 'f': m - OPEN[s], 'm': m})
        else:
            notes.append({'t': e['t'], 's': s - 1, 'f': 0, 'm': OPEN[s]})
    save(n[5:] if n.startswith('harm_') and False else n, notes, [], ign)

# 3) 開放弦（6弦・5弦）＋肘：その弦の音が大きく立ち上がったら「弾いた」、そうでなければ「肘」
x = load('open_elbow'); ts = [t for t in onsets(x, min_gap=0.1, rel=0.06) if t < 25.6]
notes, perc, ign = [], [], []
for i, t in enumerate(ts):
    nxt = ts[i + 1] if i + 1 < len(ts) else t + 0.5
    t1 = max(t + 0.1, min(t + 0.3, nxt - 0.01))
    best = None
    for s in (6, 5):
        m = OPEN[s]
        r = (E(x, t + 0.04, t1, hz(m)) + E(x, t + 0.04, t1, 2 * hz(m)) + E(x, t + 0.04, t1, 3 * hz(m))) / \
            (E(x, t - 0.12, t - 0.02, hz(m)) + E(x, t - 0.12, t - 0.02, 2 * hz(m)) + E(x, t - 0.12, t - 0.02, 3 * hz(m)))
        if best is None or r > best[1]: best = (s, r)
    s, r = best
    if r > 30:
        notes.append({'t': round(t, 3), 's': s - 1, 'f': 0, 'm': OPEN[s]})
    elif r < 4:
        perc.append({'t': round(t, 3), 'lo': True})
    else:
        ign.append(round(t, 3))
save('open_elbow', notes, perc, ign)
