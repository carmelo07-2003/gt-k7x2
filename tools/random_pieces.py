"""学習用のランダムな演奏（正解つき）を大量に作る"""
import sys, numpy as np
from compose import Piece, OPEN
import compose

SHAPES = {
    'C': [None, 3, 2, 0, 1, 0], 'G': [3, 2, 0, 0, 0, 3], 'Am': [None, 0, 2, 2, 1, 0], 'Em': [0, 2, 2, 0, 0, 0],
    'D': [None, None, 0, 2, 3, 2], 'E': [0, 2, 2, 1, 0, 0], 'A': [None, 0, 2, 2, 2, 0], 'Dm': [None, None, 0, 2, 3, 1],
    'F': [1, 3, 3, 2, 1, 1], 'Bm': [None, 2, 4, 4, 3, 2], 'E5': [0, 2, 2, None, None, None], 'G5': [3, 5, 5, None, None, None],
    'Cadd9': [None, 3, 2, 0, 3, 3], 'Em7': [0, 2, 0, 0, 0, 0], 'A7': [None, 0, 2, 0, 2, 0], 'B7': [None, 2, 1, 2, 0, 2],
}
INSTS = ['MusyngKite_acoustic_guitar_steel', 'FluidR3_GM_acoustic_guitar_steel']


def make(seed):
    rng = np.random.default_rng(seed)
    compose.rng = np.random.default_rng(seed + 1000)
    p = Piece(INSTS[seed % 2], 10.0)
    t = 0.5
    ioi = float(rng.choice([0.07, 0.09, 0.11, 0.13, 0.15, 0.18, 0.22, 0.28]))
    texture = rng.choice(['melody', 'finger', 'strum', 'arp', 'mix', 'mix'])
    pos = int(rng.integers(0, 6))  # 手の位置
    ch = rng.choice(list(SHAPES))
    last = None
    while t < 9.0:
        r = rng.random()
        tx = texture if texture != 'mix' else rng.choice(['melody', 'finger', 'strum', 'arp'])
        if rng.random() < 0.08:
            ch = rng.choice(list(SHAPES))
        if rng.random() < 0.1:
            pos = int(np.clip(pos + rng.integers(-3, 4), 0, 9))
        if r < 0.12:  # 打音
            lo = rng.random() < 0.6
            p.hit(t, lo, float(rng.uniform(0.4, 0.9)), with_slap=not lo)
            if rng.random() < 0.4 and tx == 'strum':
                p.strum(t + 0.004, SHAPES[ch], True, float(rng.uniform(0.006, 0.02)), 0.5)
        elif tx == 'melody':
            if last is not None and rng.random() < 0.3:
                s, f = last  # 同じ音の連打
            else:
                s = int(rng.integers(0, 4)); f = int(np.clip(pos + rng.integers(0, 4), 0, 12)) if rng.random() > 0.25 else 0
            p.pluck(t, s, f, float(rng.uniform(0.45, 0.85))); last = (s, f)
        elif tx == 'finger':
            shape = SHAPES[ch]
            bass = [s for s in (5, 4, 3) if shape[5 - s] is not None]
            treble = [s for s in (2, 1, 0) if shape[5 - s] is not None]
            if rng.random() < 0.5 and bass:
                b = int(rng.choice(bass)); p.pluck(t, b, shape[5 - b], float(rng.uniform(0.6, 0.9)))
                if rng.random() < 0.4 and treble:
                    s = int(rng.choice(treble)); p.pluck(t + float(rng.uniform(0, 0.01)), s, shape[5 - s], 0.55)
            elif treble:
                s = int(rng.choice(treble)); p.pluck(t, s, shape[5 - s], float(rng.uniform(0.45, 0.75)))
        elif tx == 'strum':
            down = rng.random() < 0.6
            fr = SHAPES[ch] if down else [None, None] + SHAPES[ch][2:]
            if rng.random() < 0.3:
                fr = [None, None, None] + SHAPES[ch][3:]
            p.strum(t, fr, down, float(rng.uniform(0.006, 0.022)), float(rng.uniform(0.4, 0.75)))
        else:  # arp
            shape = SHAPES[ch]
            ss = [s for s in range(6) if shape[5 - s] is not None]
            s = int(rng.choice(ss)); p.pluck(t, s, shape[5 - s], float(rng.uniform(0.45, 0.75)))
        t += ioi * float(rng.choice([1, 1, 1, 2])) if tx != 'strum' else ioi * float(rng.choice([1, 2]))
        if rng.random() < 0.03:
            p.mute_all(t - 0.02)
    kind = 'speaker' if seed % 5 == 4 else 'phone'
    p.save(f'rnd{seed:03d}', kind)


if __name__ == '__main__':
    a, b = int(sys.argv[1]), int(sys.argv[2])
    for sd in range(a, b):
        make(sd)
    print('made', b - a)


def make_moroha(seed):
    """MOROHA風：速い16分で、ベースの連打・メロディ・ボディを叩く・叩きながらジャーン"""
    rng = np.random.default_rng(seed)
    compose.rng = np.random.default_rng(seed + 5000)
    p = Piece(INSTS[seed % 2], 10.0)
    ioi = float(rng.uniform(0.1, 0.17))
    t = 0.5
    ch = rng.choice(['Em', 'Am', 'C', 'G', 'D', 'E', 'Em7', 'A7'])
    i = 0
    while t < 9.0:
        if rng.random() < 0.06:
            ch = rng.choice(list(SHAPES))
        shape = SHAPES[ch]
        bass = [s for s in (5, 4, 3) if shape[5 - s] is not None] or [3]
        treble = [s for s in (2, 1, 0) if shape[5 - s] is not None] or [0]
        pos16 = i % 16
        r = rng.random()
        if pos16 in (4, 12) and r < 0.8:
            p.hit(t, True, float(rng.uniform(0.5, 0.95)))
            if rng.random() < 0.45:
                fr = shape if rng.random() < 0.6 else [None, None] + shape[2:]
                p.strum(t + float(rng.uniform(0, 0.01)), fr, True, float(rng.uniform(0.006, 0.016)), 0.5)
        elif r < 0.12:
            p.hit(t, False, float(rng.uniform(0.4, 0.8)), with_slap=True)
        elif pos16 % 2 == 0:
            b = int(rng.choice(bass)); p.pluck(t, b, shape[5 - b] if shape[5 - b] is not None else 0, float(rng.uniform(0.6, 0.9)))
            if rng.random() < 0.35:
                s = int(rng.choice(treble)); p.pluck(t + float(rng.uniform(0, 0.008)), s, shape[5 - s] if shape[5 - s] is not None else 0, 0.55)
        else:
            s = int(rng.choice(treble)); f = shape[5 - s] if shape[5 - s] is not None else 0
            if rng.random() < 0.25:
                f = min(12, f + int(rng.integers(1, 4)))
            p.pluck(t, s, f, float(rng.uniform(0.45, 0.75)))
        t += ioi; i += 1
    kind = 'speaker' if seed % 4 == 3 else 'phone'
    p.save(f'mrh{seed:03d}', kind)


def run_moroha(a, b):
    for sd in range(a, b):
        make_moroha(sd)
