// 学習用：候補ごとの特徴と正解ラベルを書き出す
const fs=require('fs');require('/home/claude/gt-k7x2/engine.js');const E=globalThis.TabEngine;
const mode=process.argv[2];const names=process.argv.slice(3);
const out=[];
const nets=fs.existsSync('/home/claude/harness/nets.json')?JSON.parse(fs.readFileSync('/home/claude/harness/nets.json')):null;
if(nets)E.setNets(nets.note,nets.perc||null,nets.note2||null);
for(const n of names){
  const bp=JSON.parse(fs.readFileSync(`/home/claude/harness/data/${n}.bp.json`)),gt=JSON.parse(fs.readFileSync(`/home/claude/harness/data/${n}.json`));
  const x=Float32Array.from(bp.x22);
  const hf=E.hfEnvelope(x);
  if(mode==='note'){
    const C=E.onsetCandidates(bp,x,39,92,0.15);
    const strong=C.filter(c=>c.o>0.5).map(c=>E.partialE(x,c.time-0.011,c.m,0.04)).sort((a,b)=>a-b);
    const eRef=strong.length?strong[Math.floor(strong.length*0.9)]:1;
    const used=new Set();
    for(const c of [...C].sort((a,b)=>b.o-a.o)){let best=-1,bd=0.05;gt.notes.forEach((g,i)=>{if(used.has(i)||g.m!==c.m)return;const d=Math.abs(g.t-(c.time-0.009));if(d<bd){bd=d;best=i}});c.y=best>=0?1:0;if(best>=0)used.add(best)}
    const openSet=new Set(gt.user?[64,59,54,49,45,40]:[64,59,55,50,45,40]);
    const ign=gt.ignore||[];
    for(const c of C){if(ign.some(t=>Math.abs(t-(c.time-0.009))<0.06))continue;out.push({n,y:c.y,t:c.t,k:c.k,f:E.features(c,bp,x,hf,eRef,openSet)})}
  }else{
    const res=E.transcribe(bp,x,{perc:false,keepCands:true,tuning:gt.user?[64,59,54,49,45,40]:undefined});
    const notes=res.notes;
    const hfMax=Math.max(...hf.env),le=E.lowEnvelope(x);
    const C=E.percCandidates(x,hf,le);const used=new Set();
    for(const c of C){let best=-1,bd=0.04;gt.perc.forEach((g,i)=>{if(used.has(i))return;const d=Math.abs(g.t-c.time);if(d<bd){bd=d;best=i}});c.y=best>=0?1:0;c.lo=best>=0?(gt.perc[best].lo==null?-1:(gt.perc[best].lo?1:0)):-1;if(best>=0)used.add(best)}
    const ign2=gt.ignore||[];
    for(const c of C){if(ign2.some(t=>Math.abs(t-c.time)<0.06))continue;out.push({n,y:c.y,lo:c.lo,time:c.time,f:E.percFeatures(c,x,hf,notes,hfMax,le,res.cands)})};
  }
}
fs.writeFileSync(`/home/claude/harness/${mode}_feats.json`,JSON.stringify(out));
const pos=out.filter(o=>o.y).length;console.log(mode,'rows',out.length,'pos',pos);
