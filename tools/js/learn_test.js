// アプリの「修正→確認済み→学習」を、正解データを修正として使って再現する
const fs=require('fs');require('/home/claude/gt-k7x2/engine.js');require('/home/claude/gt-k7x2/nets.js');const E=TabEngine;
const TUN=[64,59,54,49,45,40];
const files=['elbow1','hand1','open_elbow','harm_2s7','harm_2s12','harm_3s7','harm_3s12','harm_4s7mix'];
function histOf(gt,t0,t1){const h=[];for(const g of gt.notes)if(g.t>=t0&&g.t<t1)h.push({k:'n',m:g.m,t:Math.round(g.t*1000)});for(const p of gt.perc)if(p.t>=t0&&p.t<t1)h.push({k:'p',lo:p.lo,t:Math.round(p.t*1000)});return h.sort((a,b)=>a.t-b.t)}
const recsN=[],recsP=[];
for(const n of files){
  const bp=JSON.parse(fs.readFileSync(`../data/user_${n}.bp.json`)),gt=JSON.parse(fs.readFileSync(`../data/user_${n}.json`));
  const x=Float32Array.from(bp.x22),dur=x.length/22050;
  const res=E.transcribe(bp,x,{tuning:TUN,keepCands:true,harmPref:n.startsWith('harm')});
  // 前7割だけを「修正済みの録音」として使う
  const cut=0.7*dur;res.cands=res.cands.filter(c=>c.time<cut);res.percCands=res.percCands.filter(c=>c.time<cut);
  const rows=E.learnRows(res,histOf(gt,0,cut));recsN.push(rows.note);recsP.push(rows.perc);
  console.log(n.padEnd(12),'学習用の行 音',rows.note.length,'(正解',rows.note.filter(r=>r.y).length,') 打音',rows.perc.length,'(正解',rows.perc.filter(r=>r.y).length,')');
}
const t0=Date.now();const rn=E.learnFromRows('note',recsN),rp=E.learnFromRows('perc',recsP);
console.log('学習時間',Date.now()-t0,'ms');
console.log('音   ：確認用 F1',rn.before&&rn.before.toFixed(3),'→',rn.after&&rn.after.toFixed(3),rn.ok?'採用':'不採用',rn.reason||'');
console.log('打音 ：確認用 F1',rp.before&&rp.before.toFixed(3),'→',rp.after&&rp.after.toFixed(3),rp.ok?'採用':'不採用',rp.reason||'');
fs.writeFileSync('/tmp/corr.json',JSON.stringify({note:rn.model,perc:rp.model}));
