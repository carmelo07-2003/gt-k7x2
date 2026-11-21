// テスト曲で精度を測る：音（高さ＋タイミング）・打音・弦の正解率
const fs = require('fs');
require('/home/claude/gt-k7x2/engine.js');
const E = globalThis.TabEngine;
const TM = require('/home/claude/bp/node_modules/@spotify/basic-pitch/cjs/toMidi.js');
const nets = fs.existsSync('/home/claude/harness/nets.json') ? JSON.parse(fs.readFileSync('/home/claude/harness/nets.json')) : {};
E.setNets(nets.note || null, nets.perc || null, nets.note2 || null);
const args = process.argv.slice(2);
const verbose = args.includes('-v'); const names = args.filter(a => a !== '-v');
const TUN = [64, 59, 55, 50, 45, 40];

function match(pred, gt, tol = 0.05) { // 同じ高さ・±50ms
  const used = new Set(); let tp = 0; const pairs = [];
  for (const p of pred) { let best = -1, bd = tol; gt.forEach((g, i) => { if (used.has(i) || g.m !== p.m) return; const d = Math.abs(g.t - p.t); if (d < bd) { bd = d; best = i } }); if (best >= 0) { used.add(best); tp++; pairs.push([p, gt[best]]) } }
  return { tp, fp: pred.length - tp, fn: gt.length - tp, pairs };
}
const tot = { base: [0, 0, 0], eng: [0, 0, 0], perc: [0, 0, 0], str: [0, 0], lohi: [0, 0] };
for (const n of names) {
  const bp = JSON.parse(fs.readFileSync(`/home/claude/harness/data/${n}.bp.json`)), gt = JSON.parse(fs.readFileSync(`/home/claude/harness/data/${n}.json`));
  const x = Float32Array.from(bp.x22);
  // 基準：Basic Pitch 標準の取り出し方
  const base = TM.noteFramesToTime(TM.outputToNotesPoly(bp.frames, bp.onsets, 0.5, 0.3, 5)).filter(q => q.pitchMidi >= 39).map(q => ({ t: q.startTimeSeconds, m: q.pitchMidi }));
  const b = match(base, gt.notes);
  // 新エンジン
  const res = E.transcribe(bp, x, { sens: 1, tuning: TUN });
  const ev = E.groupEvents(res, { style: 'auto' });
  const pos = E.assignTab(ev, { tuning: TUN, capo: 0, frets: 15 });
  const pred = [];
  ev.forEach((e, i) => {
    if (e.k === 'n') pred.push({ t: e.t / 1000, m: e.m, pos: pos[i] });
    else if (e.k === 'g') e.ms.forEach((m, j) => { const pp = (pos[i] || []).find((q, qi) => q && TUN[q.s] + q.f === m); pred.push({ t: e.t / 1000, m, pos: pp }); });
  });
  // まとめた出来事の時刻はかたまりの先頭なので、元の音の時刻で照合する
  const predRaw = res.notes.map(q => ({ t: q.time, m: q.m }));
  const r = match(predRaw, gt.notes);
  const g2 = match(pred, gt.notes, 0.2);
  let sOk = 0, sN = 0;
  for (const [p, g] of g2.pairs) { if (p.pos) { sN++; if (p.pos.s === g.s) sOk++; } }
  const pm = { tp: 0, fp: 0, fn: 0 }; const usedP = new Set(); let lohiOk = 0;
  for (const q of res.perc) { let best = -1, bd = 0.05; gt.perc.forEach((g, i) => { if (usedP.has(i)) return; const d = Math.abs(g.t - q.time); if (d < bd) { bd = d; best = i } }); if (best >= 0) { usedP.add(best); pm.tp++; if (!!gt.perc[best].lo === !!q.lo) lohiOk++; } else pm.fp++; }
  pm.fn = gt.perc.length - pm.tp;
  const F = (tp, fp, fn) => (2 * tp / Math.max(1, 2 * tp + fp + fn)).toFixed(3);
  console.log(n.padEnd(15), 'BP標準 F', F(b.tp, b.fp, b.fn), `(誤${b.fp}/漏${b.fn})`, '| 新 F', F(r.tp, r.fp, r.fn), `(誤${r.fp}/漏${r.fn})`, '| 打音 F', F(pm.tp, pm.fp, pm.fn), `(誤${pm.fp}/漏${pm.fn})`, '| 弦', sN ? (sOk / sN).toFixed(2) : '-');
  if (verbose) {
    const fpList = predRaw.filter(p => !r.pairs.some(([q]) => q === p)).map(p => `${p.m}@${p.t.toFixed(2)}`);
    const fnList = gt.notes.filter(g => !r.pairs.some(([, q]) => q === g)).map(g => `${g.m}@${g.t.toFixed(2)}`);
    console.log('   誤:', fpList.slice(0, 20).join(' '), '\n   漏:', fnList.slice(0, 20).join(' '));
  }
  for (const [k, v] of [['base', b], ['eng', r]]) { tot[k][0] += v.tp; tot[k][1] += v.fp; tot[k][2] += v.fn; }
  tot.perc[0] += pm.tp; tot.perc[1] += pm.fp; tot.perc[2] += pm.fn; tot.str[0] += sOk; tot.str[1] += sN; tot.lohi[0] += lohiOk; tot.lohi[1] += pm.tp;
}
const F = a => (2 * a[0] / Math.max(1, 2 * a[0] + a[1] + a[2])).toFixed(3);
console.log('合計  BP標準 F', F(tot.base), '| 新エンジン F', F(tot.eng), `(適合 ${(tot.eng[0] / Math.max(1, tot.eng[0] + tot.eng[1])).toFixed(3)} 再現 ${(tot.eng[0] / Math.max(1, tot.eng[0] + tot.eng[2])).toFixed(3)})`, '| 打音 F', F(tot.perc), '| ドン/パン正解', (tot.lohi[0] / Math.max(1, tot.lohi[1])).toFixed(2), '| 弦の正解率', (tot.str[0] / Math.max(1, tot.str[1])).toFixed(3));
