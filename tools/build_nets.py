"""学習した重みを、アプリが読み込む nets.js に書き出す"""
import json
n = json.load(open('/home/claude/harness/nets.json'))
def rnd(o):
    if isinstance(o, float): return round(o, 4)
    if isinstance(o, list): return [rnd(v) for v in o]
    if isinstance(o, dict): return {k: rnd(v) for k, v in o.items()}
    return o
js = '/* 耳コピ解析の判定器の重み（テスト音源で学習）。engine.js の後に読み込む */\n'
js += 'TabEngine.setNets(' + json.dumps(rnd(n.get('note')), separators=(',', ':')) + ',' + json.dumps(rnd(n.get('perc')), separators=(',', ':')) + ',' + json.dumps(rnd(n.get('note2')), separators=(',', ':')) + ',' + json.dumps(rnd(n.get('harm')), separators=(',', ':')) + ');\n'
open('/home/claude/gt-k7x2/nets.js', 'w').write(js)
print('nets.js', len(js), 'bytes')
