"""テスト曲（正解つき）を作る"""
from compose import Piece
import numpy as np

STEEL = 'MusyngKite_acoustic_guitar_steel'
STEEL2 = 'FluidR3_GM_acoustic_guitar_steel'


def fast_melody(ioi=0.1, inst=STEEL, name='fast_melody'):
    p = Piece(inst, 6.0)
    seq = [(0, 0), (0, 3), (1, 3), (1, 3), (1, 0), (2, 2), (2, 0), (2, 0), (2, 2), (1, 0), (1, 3), (0, 0), (0, 0), (0, 3),
           (0, 5), (0, 3), (1, 5), (1, 3), (1, 3), (1, 0), (2, 2), (2, 0), (3, 2), (3, 0), (3, 0), (2, 0), (2, 2), (1, 0),
           (0, 0), (0, 0)]
    t = 0.6
    for s, f in seq:
        p.pluck(t, s, f, 0.7)
        t += ioi
    p.save(name)
    return p


CH = {  # 6弦→1弦
    'C': [None, 3, 2, 0, 1, 0], 'G': [3, 2, 0, 0, 0, 3], 'Am': [None, 0, 2, 2, 1, 0], 'Em': [0, 2, 2, 0, 0, 0],
    'D': [None, None, 0, 2, 3, 2], 'E': [0, 2, 2, 1, 0, 0],
}


def fret_of(ch, s):
    return CH[ch][5 - s]


def travis(ioi=0.2, inst=STEEL, name='travis'):
    p = Piece(inst, 14.0)
    plan = [('C', (4, 3)), ('G', (5, 2)), ('Am', (4, 3)), ('Em', (5, 3))]
    t = 0.6
    for ch, (b1, b2) in plan:
        for bar in range(2):
            steps = [[b1, 0], [2], [b2], [1], [b1], [2], [b2], [1]]
            for st in steps:
                for s in st:
                    f = fret_of(ch, s)
                    if f is not None:
                        p.pluck(t, s, f, 0.75 if s >= 3 else 0.6)
                t += ioi
    p.save(name)
    return p


def moroha_like(ioi=0.15, inst=STEEL, name='moroha_like'):
    p = Piece(inst, 12.0)
    t = 0.6
    for bar in range(4):
        b = 5 if bar % 2 == 0 else 4      # 6弦開放 / 5弦開放
        top = [(0, 0), (1, 0), (0, 3), (1, 3)] if bar < 2 else [(0, 0), (1, 1), (0, 2), (1, 3)]
        for i in range(16):
            tt = t + i * ioi
            if i in (0, 8):
                p.pluck(tt, b, 0, 0.8); p.pluck(tt, top[0][0], top[0][1], 0.6)
            elif i in (2, 6, 10, 14):
                p.pluck(tt, b, 0, 0.75)        # 同じ弦の速い連打
            elif i in (1, 5, 9):
                p.pluck(tt, top[1][0], top[1][1], 0.6)
            elif i in (3, 7, 11, 15):
                p.pluck(tt, top[2][0], top[2][1], 0.6) if i != 11 else p.pluck(tt, top[3][0], top[3][1], 0.6)
            elif i == 4:
                p.hit(tt, True, 0.8)
            elif i == 12:
                p.hit(tt, True, 0.8)           # 叩きながらジャーン
                p.strum(tt + 0.005, CH['Em'], True, 0.012, 0.55)
            elif i == 13:
                p.hit(tt, False, 0.7, with_slap=True)
        t += 16 * ioi
    p.save(name)
    return p


def strums(inst=STEEL, name='strums'):
    p = Piece(inst, 14.0)
    t = 0.6
    for ch in ['Em', 'C', 'G', 'D', 'Am']:
        pat = [True, False, True, False]   # ダウン・アップ
        for k, down in enumerate(pat * 2):
            fr = CH[ch] if down else [None, None] + CH[ch][2:]
            p.strum(t, fr, down, 0.013 if down else 0.01, 0.6 if down else 0.45)
            t += 0.25
    # 16分の速いストローク
    for k in range(8):
        p.strum(t, CH['G'] if k % 2 == 0 else [None, None] + CH['G'][2:], k % 2 == 0, 0.008, 0.5)
        t += 0.125
    p.save(name)
    return p


def arpeggio(ioi=0.07, inst=STEEL, name='arpeggio'):
    p = Piece(inst, 8.0)
    t = 0.6
    for ch in ['Am', 'C', 'Am', 'C']:
        order = [4, 3, 2, 1, 0, 1, 2, 3]
        for s in order:
            f = fret_of(ch, s)
            if f is not None:
                p.pluck(t, s, f, 0.6)
            t += ioi
    p.save(name)
    return p


if __name__ == '__main__':
    fast_melody()
    fast_melody(0.125, STEEL2, 'fast_melody_b')
    travis()
    moroha_like()
    moroha_like(0.13, STEEL2, 'moroha_like_b')
    strums()
    arpeggio()
    # 動画をスピーカーで流して拾った想定
    p = moroha_like(0.15, STEEL, 'moroha_spk_tmp'); p.save('moroha_speaker', 'speaker')
    import os; [os.remove(f) for f in ['data/moroha_spk_tmp.wav', 'data/moroha_spk_tmp.json']]
    print('ok')
