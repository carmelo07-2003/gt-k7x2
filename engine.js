/* ギター相棒：耳コピ解析エンジン
 * Basic Pitch（AI）の生の出力（鳴り始めの確率・鳴っている確率）と録音そのものを使って、
 *  1) 本当に弾いた音を学習済みの判定器で選ぶ（倍音・響きの拾い直し・混線を除く）
 *  2) ボディを叩いた音（打音）を見つける
 *  3) ジャラーンと鳴らした音をまとめる
 *  4) 弦の物理（同じ弦は同時に1音、響いている弦は弾いていない）を使って、TAB譜の場所を決める
 * アプリ（ブラウザ）とテスト（Node）の両方で使う。
 */
(function (G) {
  'use strict';
  const SR = 22050, HOP = 256, ANNOT = 172, MIDI0 = 21;
  const WOFF = (HOP / SR) * (ANNOT - (SR * 2 - HOP) / HOP) + 0.0018;
  const f2t = f => f * HOP / SR - WOFF * Math.floor(f / ANNOT);
  const hz = m => 440 * Math.pow(2, (m - 69) / 12);
  const sig = z => 1 / (1 + Math.exp(-z));
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const log10 = v => Math.log(v) / Math.LN10;

  /* ---------- 信号の道具 ---------- */
  // ハン窓つきゲーツェル法：ある周波数のエネルギー
  function goertzel(x, start, len, f) {
    const k = 2 * Math.cos(2 * Math.PI * f / SR);
    let s1 = 0, s2 = 0;
    const a = Math.max(0, start), b = Math.min(x.length, start + len);
    for (let j = a; j < b; j++) {
      const i = j - start, w = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (len - 1));
      const s0 = x[j] * w + k * s1 - s2; s2 = s1; s1 = s0;
    }
    return Math.max(0, s1 * s1 + s2 * s2 - k * s1 * s2);
  }
  // ある高さの音の、倍音を含めたエネルギー
  function partialE(x, t, m, dur) {
    const st = Math.round(t * SR), len = Math.round(dur * SR), f0 = hz(m);
    let e = 0;
    for (let h = 1; h <= 6; h++) { const f = h * f0; if (f > 8000) break; e += goertzel(x, st, len, f) / h; }
    return e + 1e-12;
  }
  // 高い周波数の変化（カツッという出だし）の強さの時間変化
  function hfEnvelope(x) {
    const hop = 128, win = 256, env = [];
    for (let i = 0; i + win < x.length; i += hop) {
      let e = 0;
      for (let j = 1; j < win; j++) { const d = x[i + j] - x[i + j - 1]; e += d * d; }
      env.push(Math.sqrt(e / win));
    }
    return { env, dt: hop / SR };
  }
  const lowBand = (x, st) => goertzel(x, st, 1102, 90) + goertzel(x, st, 1102, 110) + goertzel(x, st, 1102, 130) + goertzel(x, st, 1102, 165) + 1e-12;

  /* ---------- 1) 音の候補と特徴 ---------- */
  const NF = 23;
  function onsetCandidates(bp, x, lo, hi, minO) {
    const O = bp.onsets, F = bp.frames, T = O.length, C = [];
    const kLo = Math.max(0, lo - MIDI0), kHi = Math.min(87, hi - MIDI0);
    for (let k = kLo; k <= kHi; k++) for (let t = 6; t < T - 6; t++) {
      const o = O[t][k];
      if (o >= minO && o >= O[t - 1][k] && o > O[t + 1][k]) C.push({ t, k, m: k + MIDI0, o, time: f2t(t) });
    }
    return C;
  }
  function features(c, bp, x, hf, eRef, openSet) {
    const O = bp.onsets, F = bp.frames, t = c.t, k = c.k, T = O.length;
    const fAt = (tt, kk) => (kk < 0 || kk > 87 || tt < 0 || tt >= T) ? 0 : F[tt][kk];
    const oMax = (kk, r) => { if (kk < 0 || kk > 87) return 0; let v = 0; for (let d = -r; d <= r; d++) { const tt = t + d; if (tt >= 0 && tt < T) v = Math.max(v, O[tt][kk]); } return v; };
    const pre = (fAt(t - 3, k) + fAt(t - 4, k) + fAt(t - 5, k)) / 3;
    const post = (fAt(t + 1, k) + fAt(t + 2, k) + fAt(t + 3, k) + fAt(t + 4, k)) / 4;
    const tc = c.time - 0.009; // AIは実際より約9ms遅れて反応する
    const eb = partialE(x, tc - 0.05, c.m, 0.04), ea = partialE(x, tc - 0.002, c.m, 0.04), el = partialE(x, tc + 0.045, c.m, 0.04);
    // 倍音の聞き間違い：低い音（1・1.5・2・2.5オクターブ下）の出だしや響きとの比較
    let ghostO = 0, ghostF = 0;
    for (const d of [12, 19, 24, 28, 31]) { ghostO = Math.max(ghostO, oMax(k - d, 2)); ghostF = Math.max(ghostF, fAt(t + 2, k - d)); }
    const upO = oMax(k + 12, 2), nbO = Math.max(oMax(k - 1, 1), oMax(k + 1, 1));
    // 同じ時刻のほかの候補
    let simMax = 0, simN = 0;
    for (let kk = 0; kk < 88; kk++) { if (kk === k) continue; const v = oMax(kk, 2); if (v > 0.3) simN++; simMax = Math.max(simMax, v); }
    // 打音らしさ：その時刻の高域の出だし
    const hi = Math.round(tc / hf.dt);
    let hfPk = 0, hfMin = 1e9;
    for (let j = hi - 2; j <= hi + 3; j++) if (j >= 0 && j < hf.env.length) hfPk = Math.max(hfPk, hf.env[j]);
    for (let j = hi - 8; j <= hi - 3; j++) if (j >= 0 && j < hf.env.length) hfMin = Math.min(hfMin, hf.env[j]);
    // オクターブの見分け：低い音なら3倍・5倍の音があり、1オクターブ上の音なら1.5倍・2.5倍の位置には何もない
    const wl = clamp(6 / hz(c.m), 0.04, 0.08), s0 = Math.round((tc - 0.002) * SR), L = Math.round(wl * SR), f0 = hz(c.m);
    const g = mult => goertzel(x, s0, L, f0 * mult) + 1e-12;
    const g1 = g(1), g2 = g(2), g3 = g(3), g4 = g(4), g5 = f0 * 5 < 10500 ? g(5) : 1e-12, gh1 = g(1.5), gh2 = g(2.5);
    const o = c.o;
    return [
      o, post, pre, post - pre,
      clamp(log10(ea / eb), -3, 3), clamp(log10(ea / eRef), -6, 1), clamp(log10(el / ea), -3, 2),
      ghostO - o, ghostF, (upO - o), nbO - o,
      o - simMax, Math.log(1 + simN), (c.m - 60) / 12,
      pre * clamp(log10(ea / eb), -3, 3), clamp(Math.log((hfPk + 1e-6) / (hfMin + 1e-6)), -2, 6),
      o * o, ghostO * ghostF,
      clamp(log10(g1 / g2), -4, 4), clamp(log10((g3 + g5) / (g2 + g4)), -4, 4), clamp(log10((gh1 + gh2) / (g1 + g2)), -4, 4),
      openSet && openSet.has(c.m) ? 1 : 0, openSet && openSet.has(c.m - 12) ? 1 : 0,
    ];
  }

  /* 学習済みの判定器（小さなニューラルネット）。重みはテストで学習して埋め込む */
  let NOTE_NET = null, NOTE_NET2 = null, PERC_NET = null;
  function mlp(net, v) {
    const z = v.map((x, i) => (x - net.mu[i]) / net.sd[i]);
    let h = z;
    for (const L of net.layers) {
      const out = new Array(L.b.length);
      for (let j = 0; j < L.b.length; j++) { let s = L.b[j]; const w = L.W[j]; for (let i = 0; i < h.length; i++) s += w[i] * h[i]; out[j] = L.act === 'relu' ? Math.max(0, s) : s; }
      h = out;
    }
    return sig(h[0]);
  }
  // 学習前の仮の判定（重みがない時）
  function fallbackNote(f) { return sig(9 * (f[0] - 0.45) + 1.2 * f[4] - 2.5 * Math.max(0, f[7]) - 1.5 * Math.max(0, f[9])); }

  // 2段目の判定に使う「まわりの状況」
  function contextFeatures(cands) {
    const byK = new Map();
    for (const c of cands) { if (!byK.has(c.k)) byK.set(c.k, []); byK.get(c.k).push(c); }
    const near = (c, dk, r0, r1) => { const L = byK.get(c.k + dk); if (!L) return 0; let v = 0; for (const q of L) { const d = q.t - c.t; if (d >= r0 && d <= r1 && q !== c) v = Math.max(v, q.p1); } return v; };
    const sorted = [...cands].sort((a, b) => a.t - b.t);
    return cands.map(c => {
      let cnt = 0;
      for (const q of sorted) { if (q.t < c.t - 3) continue; if (q.t > c.t + 3) break; if (q !== c && q.p1 > 0.5) cnt++; }
      return [c.p1, near(c, -12, -2, 2), near(c, 12, -2, 2), near(c, -19, -2, 2), near(c, -24, -2, 2), Math.log(1 + cnt), near(c, 0, -15, -3), near(c, -1, -2, 2), near(c, 1, -2, 2),
        near(c, 0, -40, -16), near(c, -12, -40, -3), near(c, 12, -40, -3), near(c, -12, 3, 40), near(c, 0, 3, 40)];
    });
  }

  /* ---------- 2) 打音の候補と特徴 ---------- */
  // 低い帯域（胴の「ボン」）の大きさの時間変化。簡単なローパスフィルタを2段かける
  function lowEnvelope(x) {
    const a = Math.exp(-2 * Math.PI * 260 / SR), y = new Float32Array(x.length);
    let p1 = 0, p2 = 0;
    for (let i = 0; i < x.length; i++) { p1 = (1 - a) * x[i] + a * p1; p2 = (1 - a) * p1 + a * p2; y[i] = p2; }
    const hop = 128, win = 256, env = [];
    for (let i = 0; i + win < y.length; i += hop) { let e = 0; for (let j = 0; j < win; j++) e += y[i + j] * y[i + j]; env.push(Math.sqrt(e / win)); }
    return env;
  }
  function percCandidates(x, hf, le) {
    const H = hf.env, L = le, n = Math.min(H.length, L.length), hm = Math.max(...H), lm = Math.max(...L), C = [];
    let last = -99;
    const rise = (E, i) => E[i] / (Math.min(E[i - 1], E[i - 2], E[i - 3], E[i - 4], E[i - 5], E[i - 6]) + 1e-9);
    for (let i = 6; i < n - 12; i++) {
      const rh = rise(H, i), rl = rise(L, i);
      const okH = H[i] >= hm * 0.03 && rh >= 1.5 && H[i] >= H[i - 1] && H[i] > H[i + 1];
      const okL = L[i] >= lm * 0.04 && rl >= 1.3 && L[i] >= L[i - 1] && L[i] >= L[i + 1];
      if (!okH && !okL) continue;
      if ((i - last) * hf.dt < 0.04) continue;
      last = i; C.push({ i, time: i * hf.dt + 0.003, pk: H[i], mn: H[i] / rh, rl, rh, lv: L[i] / lm });
    }
    return C;
  }
  function percFeatures(c, x, hf, notes, hfMax, le, cands) {
    const st = Math.round(c.time * SR), env = hf.env, i = c.i;
    const lb = lowBand(x, st - 1400), la = lowBand(x, st + 100), ll = lowBand(x, st + 2300);
    const decay = (env[Math.min(env.length - 1, i + 7)] + 1e-6) / (c.pk + 1e-6);
    const ldecay = ((le[Math.min(le.length - 1, i + 14)] || 0) + 1e-6) / ((le[i] || 0) + 1e-6);
    let nearP = 0, nearN = 0, nearLow = 1, nearSus = 0, nearHiP = 0;
    for (const n of notes) {
      if (Math.abs(n.time - c.time) < 0.035) { nearP = Math.max(nearP, n.p); nearN++; if (n.m > 52) { nearLow = 0; nearHiP = Math.max(nearHiP, n.p); } nearSus = Math.max(nearSus, n.sus); }
    }
    // AIが「音程のある音」としては不採用にした反応（胴を叩くとバラバラの高さで弱く反応する）
    let rejN = 0, rejLo = 0, rejHi = 0, accN = 0;
    if (cands) for (const q of cands) {
      const d = (q.time - 0.009) - c.time;
      if (d < -0.03 || d > 0.04) continue;
      if (q.acc) { accN++; continue; }
      if (q.o > 0.25) rejN++;
      if (q.m <= 52) rejLo = Math.max(rejLo, q.o); else rejHi = Math.max(rejHi, q.o);
    }
    return [
      clamp(Math.log(c.rh), 0, 8), clamp(log10(c.pk / hfMax), -3, 0),
      clamp(log10(la / lb), -3, 3), clamp(log10(ll / la), -3, 2), clamp(log10(decay), -3, 1),
      nearP, Math.log(1 + nearN), nearLow, nearSus,
      clamp(Math.log(c.rl), 0, 8), clamp(log10(c.lv + 1e-6), -3, 0), clamp(log10(ldecay), -3, 1), nearHiP,
      Math.log(1 + rejN), rejLo, rejHi, Math.log(1 + accN),
    ];
  }
  function fallbackPerc(f) { return sig(0.8 * (f[0] - 2) + 1.5 * f[9] + 2 * f[2] - 4 * f[12] - 2 * f[4]); }

  /* ---------- 解析の本体 ---------- */
  function transcribe(bp, x, opt) {
    opt = opt || {};
    const tuning = opt.tuning || [64, 59, 55, 50, 45, 40], capo = opt.capo || 0;
    const lo = Math.min(...tuning) + capo - 1, hi = Math.max(...tuning) + 24 + 4;
    const hf = hfEnvelope(x);
    const openSet = new Set(tuning.map(m => m + capo));
    const cands = onsetCandidates(bp, x, lo, hi, 0.15);
    // 音量の基準（強い候補の大きさ）
    const strong = cands.filter(c => c.o > 0.5).map(c => partialE(x, c.time - 0.011, c.m, 0.04)).sort((a, b) => a - b);
    const eRef = strong.length ? strong[Math.floor(strong.length * 0.9)] : 1;
    const base = NOTE_NET2 && NOTE_NET2.th ? NOTE_NET2.th : NOTE_NET && NOTE_NET.th ? NOTE_NET.th : 0.5;
    const thr = [Math.min(0.95, base + 0.1), base, Math.max(0.2, base - 0.15)][opt.sens == null ? 1 : opt.sens];
    const notes = [];
    for (const c of cands) {
      c.f = features(c, bp, x, hf, eRef, openSet);
      c.p1 = NOTE_NET ? mlp(NOTE_NET, c.f) : fallbackNote(c.f);
    }
    // 2段目：まわりの判定結果（1オクターブ下や上が本物か、直前に同じ音があるか）も見て決め直す
    if (NOTE_NET2) { const ctx = contextFeatures(cands); cands.forEach((c, i) => { c.p = mlp(NOTE_NET2, c.f.concat(ctx[i])); }); }
    else cands.forEach(c => { c.p = c.p1; });
    // 「ハーモニクスを弾いている」がONの時は、ハーモニクスで出せる高さ（開放弦の1・1.5・2オクターブ上）の基準をゆるめる
    const harmSet = new Set();
    if (opt.harmPref) tuning.forEach(o => [12, 19, 24].forEach(a => harmSet.add(o + capo + a)));
    const hThr = Math.max(0.2, thr - (opt.harmBoost == null ? 0.3 : opt.harmBoost));
    for (const c of cands) if (c.p >= (harmSet.has(c.m) ? hThr : thr)) { notes.push(c); c.acc = true; }
    // 同じ高さで近すぎる（35ms以内）ものは強い方だけ
    notes.sort((a, b) => b.p - a.p);
    const kept = [];
    for (const n of notes) { if (!kept.some(q => q.m === n.m && Math.abs(q.time - n.time) < 0.035)) kept.push(n); }
    kept.sort((a, b) => a.time - b.time);
    // 終わり：鳴っている確率が下がるまで、または同じ高さが次に弾かれるまで
    const F = bp.frames, T = F.length;
    for (const n of kept) {
      const next = kept.find(q => q.m === n.m && q.time > n.time + 0.02);
      const tEnd = next ? next.t : T - 1;
      let t = n.t + 1, miss = 0;
      while (t < tEnd && miss < 6) { if (F[t][n.k] < 0.3) miss++; else miss = 0; t++; }
      n.end = f2t(Math.min(tEnd, t - miss));
      let a = 0, c = 0; for (let j = n.t; j <= Math.min(T - 1, n.t + 8); j++) { a += F[j][n.k]; c++; }
      n.amp = a / c;
      n.sus = n.f[6];
      n.time = n.time - 0.009;
    }
    // 打音
    let perc = [];
    if (opt.perc !== false) {
      const hfMax = Math.max(...hf.env), le = lowEnvelope(x);
      for (const c of percCandidates(x, hf, le)) {
        const f = percFeatures(c, x, hf, kept, hfMax, le, cands);
        const p = PERC_NET ? mlp(PERC_NET, f) : fallbackPerc(f);
        if (p >= (opt.percThr || (PERC_NET && PERC_NET.th) || 0.5)) {
          let lo = f[2] > 0.25;
          if (PERC_NET && PERC_NET.lohi) { const L = PERC_NET.lohi; let z = L.b; for (let i = 0; i < f.length; i++) z += L.w[i] * (f[i] - L.mu[i]) / L.sd[i]; lo = z >= 0; }
          perc.push({ time: c.time, lo, p, f });
        }
      }
      // 1回の叩きが長く響いて何度も見つかるのを防ぐ：0.09秒以内は一番確かなもの1つに
      perc.sort((a, b) => b.p - a.p);
      const pk = [];
      for (const q of perc) if (!pk.some(r => Math.abs(r.time - q.time) < 0.09)) pk.push(q);
      perc = pk.sort((a, b) => a.time - b.time);
      // 打音と同時の、低くてすぐ消える「音」は胴鳴りなので消す
      for (const q of perc) {
        for (let i = kept.length - 1; i >= 0; i--) {
          const n = kept[i];
          if (Math.abs(n.time - q.time) < 0.04 && n.m <= 52 && n.sus < -0.25) kept.splice(i, 1);
        }
      }
    }
    const out = { notes: kept.map(n => ({ time: n.time, m: n.m, p: n.p, amp: n.amp, end: n.end, sus: n.sus })), perc: perc.map(q => ({ time: q.time, lo: q.lo, p: q.p })) };
    if (opt.keepCands) out.cands = cands;
    return out;
  }

  /* ---------- 3) ジャラーンをまとめて「出来事」にする ---------- */
  function groupEvents(res, opt) {
    opt = opt || {};
    const style = opt.style || 'auto';
    const gap = style === 'chord' ? 0.05 : style === 'melody' ? 0.02 : 0.035;
    const ns = [...res.notes].sort((a, b) => a.time - b.time), groups = [];
    for (const n of ns) {
      const g = groups[groups.length - 1];
      if (g && n.time - g.last <= gap && n.time - g.t0 <= 0.16 && g.ns.length < 6 && !g.ns.some(q => q.m === n.m)) { g.ns.push(n); g.last = n.time; }
      else groups.push({ t0: n.time, last: n.time, ns: [n] });
    }
    const ev = [];
    for (const g of groups) {
      let ns2 = g.ns;
      if (style === 'melody' && ns2.length > 1) ns2 = [ns2.reduce((a, b) => (b.p * b.amp > a.p * a.amp ? b : a))];
      const t = Math.round(g.t0 * 1000);
      if (ns2.length === 1) ev.push({ k: 'n', m: ns2[0].m, t, end: Math.round(ns2[0].end * 1000) });
      else {
        const s = [...ns2].sort((a, b) => a.m - b.m);
        ev.push({ k: 'g', ms: s.map(n => n.m), ends: s.map(n => Math.round(n.end * 1000)), t });
      }
    }
    for (const q of res.perc) ev.push({ k: 'p', lo: q.lo, t: Math.round(q.time * 1000) });
    ev.sort((a, b) => a.t - b.t);
    return ev;
  }

  /* ---------- 4) TAB譜の場所を決める（ビームサーチ） ---------- */
  const HARM = [[12, 12], [7, 19], [5, 24], [19, 19], [4, 28], [9, 28]];
  function options(m, ctx, harm) {
    const r = [];
    for (let s = 0; s < 6; s++) {
      const open = ctx.tuning[s] + ctx.capo;
      const f = m - ctx.tuning[s];
      if (m === open) r.push({ s, f: 0, h: false });
      else if (f > ctx.capo && f <= ctx.frets) r.push({ s, f, h: false });
      if (harm) for (const [fr, add] of HARM) { const ff = ctx.capo + fr; if (ff <= 19 && open + add === m) r.push({ s, f: ff, h: true }); }
    }
    const lim = ctx.inRange ? r.filter(ctx.inRange) : r;
    return lim.length ? lim : r;
  }
  // 1つの出来事の押さえ方の候補（同時に鳴らす音は違う弦・指が届く範囲）
  function eventOptions(e, ctx) {
    const ms = e.k === 'n' ? [e.m] : e.ms;
    // 開放弦と同じ高さは「開放弦が鳴った」ことが多いので、ハーモニクス優先の対象にしない
    const harm = e.k === 'n' && (e.h || (ctx.harmPref && !ctx.tuning.some(o => o + ctx.capo === e.m)));
    const per = ms.map(m => options(m, ctx, harm));
    if (e.k === 'n' && e.fix) {
      const f = per[0].filter(p => p.s === e.fix.s && p.f === e.fix.f && !!p.h === !!e.fix.h);
      if (f.length) per[0] = f;
    }
    const out = [], cur = [], used = new Set();
    (function rec(i) {
      if (out.length > 400) return;
      if (i === ms.length) {
        const fr = cur.filter(p => p.f > ctx.capo && !p.h).map(p => p.f);
        const span = fr.length ? Math.max(...fr) - Math.min(...fr) : 0;
        if (span > 5) return;
        let order = 0; // 低い音ほど太い弦が普通
        for (let a = 0; a < cur.length; a++) for (let b = a + 1; b < cur.length; b++) if (ms[a] < ms[b] && cur[a].s < cur[b].s) order++;
        const avg = fr.length ? fr.reduce((x, y) => x + y, 0) / fr.length : null;
        let c = 0.45 * Math.max(0, span - 3) + 0.15 * span + 0.6 * order;
        for (const p of cur) c += p.h ? (harm ? -0.6 : 3) + ([12, 7, 5].includes(p.f - ctx.capo) ? 0 : 0.5) : (p.f > ctx.capo ? 0.035 * (p.f - ctx.capo) : 0) + (harm && e.h ? 0.8 : 0);
        out.push({ pos: cur.slice(), c, avg });
        return;
      }
      for (const p of per[i]) { if (used.has(p.s)) continue; used.add(p.s); cur.push(p); rec(i + 1); cur.pop(); used.delete(p.s); }
      if (ms.length > 1 && cur.length + (ms.length - i - 1) >= 1 && e.k === 'g') rec(i + 1); // 届かない音は省く（罰あり）
    })(0);
    for (const o of out) o.c += 2.5 * (ms.length - o.pos.length);
    out.sort((a, b) => a.c - b.c);
    return { ms, opts: out.slice(0, 24) };
  }
  function assignTab(events, ctx) {
    const W = 48;
    let beams = [{ c: 0, hand: null, occ: [null, null, null, null, null, null], hist: [] }];
    events.forEach((e, idx) => {
      if (e.k !== 'n' && e.k !== 'g') { beams.forEach(b => b.hist.push(null)); return; }
      const { ms, opts } = eventOptions(e, ctx);
      if (!opts.length) { beams.forEach(b => b.hist.push(null)); return; }
      const t = e.t / 1000, ends = e.k === 'n' ? [e.end] : (e.ends || []);
      const next = [];
      for (const b of beams) for (const o of opts) {
        let c = b.c + o.c;
        // 手の移動
        if (o.avg != null && b.hand != null) c += 0.22 * Math.max(0, Math.abs(o.avg - b.hand) - 1.5);
        for (const p of o.pos) {
          const oc = b.occ[p.s];
          if (!oc) continue;
          const m = ms[o.pos.indexOf(p)];
          if (oc.m === m) { c -= 0.25; continue; } // 同じ音は同じ弦で弾き直すのが自然（ハーモニクスも同じ）
          // 前の音がまだ響いている弦を弾いたことになる → 不自然
          if (oc.end != null && oc.end > t + 0.05) c += 1.6 * Math.min(1, (oc.end - t) / 0.3);
        }
        next.push({ b, o, c });
      }
      next.sort((x, y) => x.c - y.c);
      const nb = [], seen = new Set();
      for (const n of next) {
        const key = n.o.pos.map(p => p.s + ':' + p.f).join(',') + '|' + (n.o.avg == null ? (n.b.hand == null ? '-' : Math.round(n.b.hand)) : '');
        if (seen.has(key)) continue; seen.add(key);
        const occ = n.b.occ.slice();
        n.o.pos.forEach((p, i) => { const en = ends[i] != null ? ends[i] / 1000 : null; occ[p.s] = { m: ms[i], end: en == null ? t + 0.3 : en }; });
        const hand = n.o.avg != null ? (n.b.hand == null ? n.o.avg : 0.4 * n.b.hand + 0.6 * n.o.avg) : n.b.hand;
        nb.push({ c: n.c, hand, occ, hist: n.b.hist.concat([e.k === 'n' ? (n.o.pos[0] || null) : n.o.pos]) });
        if (nb.length >= W) break;
      }
      beams = nb;
    });
    const best = beams.reduce((a, b) => (b.c < a.c ? b : a));
    return best.hist;
  }

  G.TabEngine = {
    transcribe, groupEvents, assignTab, features, percFeatures, onsetCandidates, percCandidates, hfEnvelope, lowEnvelope, partialE, f2t,
    setNets(n, p, n2) { NOTE_NET = n; PERC_NET = p; NOTE_NET2 = n2 || null; }, contextFeatures, mlp, NF,
  };
})(typeof self !== 'undefined' ? self : globalThis);
