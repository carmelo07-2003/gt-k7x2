const fs=require('fs');require('/home/claude/gt-k7x2/engine.js');const N=JSON.parse(fs.readFileSync(process.argv[2]||'../nets.json'));TabEngine.setNets(N.note,N.perc,N.note2,N.harm);
let fpR=0,fpO=0,tp=0,gtn=0;
const files=[['fast_melody',0],['fast_melody_b',0],['travis',0],['moroha_like',0],['moroha_like_b',0],['strums',0],['arpeggio',0],['moroha_speaker',0],['user_open6to1',1],['user_open_elbow',1],['user_harm_2s7',1],['user_harm_3s7',1],['user_harm_2s12',1],['user_harm_3s12',1],['user_harm_4s7mix',1]];
for(const [n,u] of files){
 const bp=JSON.parse(fs.readFileSync(`../data/${n}.bp.json`)),gt=JSON.parse(fs.readFileSync(`../data/${n}.json`));const x=Float32Array.from(bp.x22);
 const r=TabEngine.transcribe(bp,x,{tuning:u?[64,59,54,49,45,40]:[64,59,55,50,45,40]});const ign=gt.ignore||[];
 const used=new Set();
 for(const q of r.notes){if(ign.some(t=>Math.abs(t-q.time)<0.08))continue;const i=gt.notes.findIndex((g,j)=>!used.has(j)&&g.m===q.m&&Math.abs(g.t-q.time)<0.06);if(i>=0){used.add(i);tp++;continue}
  if(gt.notes.some(g=>g.m===q.m&&q.time-g.t>0.06&&q.time-g.t<1.5))fpR++;else fpO++}
 gtn+=gt.notes.length;
}
console.log('正解',gtn,'当たり',tp,'| 余計な音のうち「響きの途中の拾い直し」',fpR,'／それ以外',fpO);
