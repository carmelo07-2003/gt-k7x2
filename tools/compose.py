"""正解つきのテスト音源を作る。
本物のアコギのサンプリング音を使い、同じ弦を弾き直すと前の音が止まる（物理どおり）。
スマホのマイク録音風（低音カット・部屋の響き・雑音）に加工する。
"""
import numpy as np, wave, json, os, sys
SR = 48000
ROOT = os.path.dirname(os.path.abspath(__file__))
OPEN = [64, 59, 55, 50, 45, 40]  # 1弦〜6弦
_cache = {}
rng = np.random.default_rng(1)


def load(inst, m):
    key = (inst, m)
    if key not in _cache:
        p = f"{ROOT}/sf/{inst}/{m}.wav"
        w = wave.open(p)
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
        _cache[key] = x / (np.max(np.abs(x)) + 1e-9)
    return _cache[key]


def knock(lo=True):
    """ボディを叩く音：胴の共鳴（約100Hzなど）＋木を叩くクリック。すぐ消える"""
    t = np.arange(int(SR * 0.45)) / SR
    if lo:
        modes = [(98, 12, 1.0), (192, 20, 0.55), (235, 26, 0.4), (390, 38, 0.2)]
    else:
        modes = [(210, 30, 0.5), (420, 45, 0.4), (900, 60, 0.3)]
    y = np.zeros_like(t)
    for f, d, a in modes:
        y += a * np.sin(2 * np.pi * f * t + rng.random() * 6) * np.exp(-t * d)
    n = int(SR * 0.006)
    click = rng.uniform(-1, 1, n) * np.linspace(1, 0, n)
    if lo:
        click = np.convolve(click, np.ones(8) / 8, 'same')
    y[:n] += click * (0.6 if lo else 1.0)
    return y / np.max(np.abs(y))


def slap():
    """弦を手のひらで叩く「パン」：広い帯域のノイズ＋弦が一瞬だけ鳴ってすぐ止まる"""
    t = np.arange(int(SR * 0.2)) / SR
    noise = rng.uniform(-1, 1, len(t)) * np.exp(-t * 60)
    # 高めの帯域に
    noise = noise - np.convolve(noise, np.ones(6) / 6, 'same')
    y = noise * 0.8
    for m in OPEN:
        f = 440 * 2 ** ((m - 69) / 12)
        y += 0.12 * np.sin(2 * np.pi * f * t) * np.exp(-t * 45)
    return y / np.max(np.abs(y))


class Piece:
    def __init__(self, inst='MusyngKite_acoustic_guitar_steel', length=12.0):
        self.inst = inst
        self.out = np.zeros(int(SR * length))
        self.notes = []  # 正解：t, s, f, m
        self.perc = []   # 正解：t, lo
        self.strings = [None] * 6  # 弦ごとに鳴っている音（開始位置, 長さ, 信号）
        self.voices = []

    def pluck(self, t, s, f, vel=0.8):
        m = OPEN[s] + f
        # 同じ弦が鳴っていたら止める（15msで消える）
        self._damp(s, t)
        x = load(self.inst, m) * vel * (1 + rng.uniform(-0.1, 0.1))
        i = int(t * SR)
        v = {'i': i, 'x': x.copy(), 's': s}
        self.voices.append(v)
        self.strings[s] = v
        self.notes.append({'t': round(t, 4), 's': s, 'f': f, 'm': m})

    def _damp(self, s, t):
        v = self.strings[s]
        if v is None:
            return
        j = int(t * SR) - v['i']
        if 0 <= j < len(v['x']):
            n = int(SR * 0.015)
            env = np.ones(len(v['x']))
            env[j:j + n] = np.linspace(1, 0, len(env[j:j + n]))
            env[j + n:] = 0
            v['x'] *= env
        self.strings[s] = None

    def mute_all(self, t):
        for s in range(6):
            self._damp(s, t)

    def strum(self, t, frets, down=True, gap=0.014, vel=0.6):
        """frets: 6弦→1弦の順（Noneは弾かない）"""
        order = list(range(5, -1, -1)) if down else list(range(6))
        k = 0
        for s in order:
            f = frets[5 - s]
            if f is None:
                continue
            self.pluck(t + k * gap, s, f, vel * (0.85 + 0.15 * rng.random()))
            k += 1

    def hit(self, t, lo=True, vel=0.7, with_slap=False):
        y = (knock(lo) if not with_slap else slap()) * vel
        i = int(t * SR)
        self.voices.append({'i': i, 'x': y, 's': None})
        self.perc.append({'t': round(t, 4), 'lo': bool(lo and not with_slap)})
        # 叩くと開放弦が少し共鳴する
        if lo and not with_slap:
            for s in (5, 4):
                if self.strings[s] is None:
                    x = load(self.inst, OPEN[s]) * 0.04
                    self.voices.append({'i': i, 'x': x, 's': None})

    def render(self, kind='phone'):
        out = np.zeros_like(self.out)
        for v in self.voices:
            i, x = v['i'], v['x']
            n = min(len(x), len(out) - i)
            if n > 0:
                out[i:i + n] += x[:n]
        return simulate(out, kind)

    def save(self, name, kind='phone'):
        y = self.render(kind)
        os.makedirs(f"{ROOT}/data", exist_ok=True)
        w = wave.open(f"{ROOT}/data/{name}.wav", 'wb')
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((y * 32767).astype(np.int16).tobytes()); w.close()
        json.dump({'notes': sorted(self.notes, key=lambda n: n['t']), 'perc': self.perc},
                  open(f"{ROOT}/data/{name}.json", 'w'))


def simulate(x, kind):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    if kind == 'phone':      # スマホのマイク：低音カット、高音少し落ちる
        X *= 1 / (1 + (110 / np.maximum(f, 1)) ** 4) / (1 + (f / 10000) ** 4)
    elif kind == 'speaker':  # 動画をスマホのスピーカーで流して拾う：もっと細い音
        X *= 1 / (1 + (250 / np.maximum(f, 1)) ** 4) / (1 + (f / 7000) ** 6)
    y = np.fft.irfft(X, len(x))
    # 部屋の響き
    ir_t = np.arange(int(SR * 0.35)) / SR
    ir = rng.normal(0, 1, len(ir_t)) * np.exp(-ir_t * 18)
    ir[0] = 0
    wet = np.fft.irfft(np.fft.rfft(y, len(y) + len(ir)) * np.fft.rfft(ir, len(y) + len(ir)))[:len(y)]
    y = y + 0.12 * wet / (np.max(np.abs(wet)) + 1e-9) * np.max(np.abs(y))
    if kind == 'speaker':
        y = np.tanh(y / np.max(np.abs(y)) * 1.6)  # 小さいスピーカーの歪み
    y = y / np.max(np.abs(y)) * 0.7
    y += rng.normal(0, 0.003, len(y))
    return np.clip(y, -1, 1)
