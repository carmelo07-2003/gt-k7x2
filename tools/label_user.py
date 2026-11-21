"""あなたの録音から、弾いた（叩いた）瞬間を見つけて、何が鳴ったかを自動で判定する（正解ラベル作り）"""
import numpy as np, wave, json, sys
SR = 48000
NAMES = ['ド', 'ド♯', 'レ', 'レ♯', 'ミ', 'ファ', 'ファ♯', 'ソ', 'ソ♯', 'ラ', 'ラ♯', 'シ']


def load(n):
    w = wave.open(f'/home/claude/harness/data/user_{n}.wav')
    return np.frombuffer(w.readframes(w.getnframes()), np.int16) / 32768


def onsets(x, min_gap=0.12, rel=0.08):
    hop = 240; win = 480
    e = np.array([np.sqrt(np.mean(x[i:i + win] ** 2)) for i in range(0, len(x) - win, hop)])
    d = np.diff(x); h = np.array([np.sqrt(np.mean(d[i:i + win] ** 2)) for i in range(0, len(d) - win, hop)])
    thr_e = e.max() * rel; out = []
    for i in range(6, len(e) - 2):
        rise = e[i] / (e[i - 6:i].min() + 1e-9); rh = h[i] / (h[i - 6:i].min() + 1e-9)
        if e[i] > thr_e and (rise > 1.8 or rh > 2.5) and e[i] >= e[i - 1] and (not out or (i - out[-1]) * hop / SR > min_gap):
            out.append(i)
    return [i * hop / SR for i in out]


def spec(x, t0, t1, n=16384):
    a = int(t0 * SR); b = int(t1 * SR); seg = x[max(0, a):max(0, b)]
    if len(seg) < 64: return None, None
    seg = seg * np.hanning(len(seg))
    X = np.abs(np.fft.rfft(seg, n)); f = np.fft.rfftfreq(n, 1 / SR)
    return X, f


def midi(f): return 69 + 12 * np.log2(f / 440)


def name(m): m = int(round(m)); return f'{NAMES[m % 12]}{m // 12 - 1}'


def analyze(n, expect):
    """expect: {'harm': [MIDI...], 'fret': [MIDI...]}  新しく鳴った音の高さで分類する"""
    x = load(n); ts = onsets(x)
    res = []
    for k, t in enumerate(ts):
        nxt = ts[k + 1] if k + 1 < len(ts) else t + 0.4
        Xb, f = spec(x, t - 0.15, t - 0.02)
        Xa, _ = spec(x, t + 0.03, min(t + 0.2, nxt - 0.01) if nxt - t > 0.08 else t + 0.08)
        if Xa is None: continue
        diff = np.maximum(0, Xa - (Xb if Xb is not None else 0))
        band = (f > 150) & (f < 2500)
        kk = np.argmax(diff * band); f0 = f[kk]; m = midi(f0)
        # 澄み具合：その音の基音の強さ ÷ 倍音（2〜5倍）の強さ
        amp = lambda fr: diff[np.argmin(abs(f - fr)) - 3:np.argmin(abs(f - fr)) + 4].max()
        pure = amp(f0) ** 2 / (amp(f0) ** 2 + sum(amp(h * f0) ** 2 for h in range(2, 6) if h * f0 < 20000) + 1e-12)
        # 低域（胴鳴り）の立ち上がりと、音程のはっきり度
        lowb = (f > 60) & (f < 250)
        lowrise = (Xa[lowb].sum() + 1e-9) / ((Xb[lowb].sum() if Xb is not None else 0) + 1e-9)
        peak_share = diff[kk] ** 2 / ((diff[band] ** 2).sum() + 1e-12)
        mr = int(round(m)); label = 'その他'
        if expect:
            if any(abs(mr - h) == 0 for h in expect.get('harm', [])): label = 'ハーモニクス成功'
            elif any(abs(mr - h) == 0 for h in expect.get('fret', [])): label = '押さえた音（失敗）'
        res.append({'t': round(t, 3), 'm': mr, 'pure': round(float(pure), 3), 'lowrise': round(float(lowrise), 2), 'share': round(float(peak_share), 3), 'label': label})
    return res


if __name__ == '__main__':
    cases = {
        'harm_2s7': {'harm': [78], 'fret': [66]},
        'harm_3s7': {'harm': [74], 'fret': [62]},
        'harm_4s7mix': {'harm': [69, 74, 78], 'fret': [57, 62, 66]},
        'harm_2s12': {'harm': [71], 'fret': []},
        'harm_3s12': {'harm': [67], 'fret': []},
    }
    out = {}
    for n, ex in cases.items():
        r = analyze(n, ex); out[n] = r
        from collections import Counter
        c = Counter(e['label'] for e in r); cm = Counter(name(e['m']) for e in r)
        print(f'{n}: 鳴らした回数 {len(r)} | {dict(c)} | 高さの内訳 {dict(cm.most_common(6))}')
        if n.endswith('12'):
            p = sorted(e['pure'] for e in r if e['label'] == 'ハーモニクス成功')
            if p: print('   12Fの澄み具合（低→高）', [round(v, 2) for v in p])
    # 開放弦＋肘
    x = load('open_elbow'); ts = onsets(x)
    r = analyze('open_elbow', None); out['open_elbow'] = r
    print('open_elbow: 出来事', len(r))
    for e in r[:60]:
        print(f"  {e['t']:6.2f}s 新しい音 {name(e['m'])} 澄み{e['pure']:.2f} 低域の立ち上がり×{e['lowrise']:.1f} 音程のはっきり度{e['share']:.2f}")
    json.dump(out, open('/home/claude/harness/data/user_labels.json', 'w'), ensure_ascii=False)
