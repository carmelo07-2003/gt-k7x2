// あなたの録音の後ろ3割で、音・打音の当たり具合を測る
const fs=require('fs');require('/home/claude/gt-k7x2/engine.js');
const nf=process.argv[2]||'/home/claude/gt-k7x2/nets.js';
if(nf.endsWith('.json')){const n=JSON.parse(fs.readFileSync(nf));TabEngine.setNets(n.note,n.perc,n.note2,n.harm)}else require(nf);
const TUN=[64,59,54,49,45,40];const tot={n:[0,0,0],p:[0,0,0]};
for(const n of ['open6to1','elbow1','hand1','open_elbow','harm_2s7','harm_2s12','harm_3s7','harm_3s12','harm_4s7mix']){
  const bp=JSON.parse(fs.readFileSync(`/home/claude/harness/data/user_${n}.bp.json`)),gt=JSON.parse(fs.readFileSync(`/home/claude/harness/data/user_${n}.json`));
  const x=Float32Array.from(bp.x22),dur=x.length/22050,t0=0.7*dur,ign=gt.ignore||[];
  const HP=n.startsWith('harm');const r=TabEngine.transcribe(bp,x,{tuning:TUN,harmPref:HP&&process.env.HP!=='0',harmBoost:process.env.HB?+process.env.HB:undefined});const ev=TabEngine.groupEvents(r,{});const pos=TabEngine.assignTab(ev,{tuning:TUN,capo:0,frets:15});
  const keep=t=>t>=t0&&!ign.some(q=>Math.abs(q-t)<0.08);
  const P=r.notes.filter(q=>keep(q.time)),G=gt.notes.filter(g=>keep(g.t));
  const used=new Set();let tp=0;for(const p of P){const i=G.findIndex((g,j)=>!used.has(j)&&g.m===p.m&&Math.abs(g.t-p.time)<0.06);if(i>=0){used.add(i);tp++}}
  const PP=r.perc.filter(q=>keep(q.time)),GP=gt.perc.filter(g=>keep(g.t));const u2=new Set();let tp2=0;for(const p of PP){const i=GP.findIndex((g,j)=>!u2.has(j)&&Math.abs(g.t-p.time)<0.06);if(i>=0){u2.add(i);tp2++}}
  // ハーモニクスが正しい高さで、TAB譜でもハーモニクスとして出たか（ハーモニクス優先ON）
  let hOk=0,hN=0;const posH=TabEngine.assignTab(ev,{tuning:TUN,capo:0,frets:15,harmPref:true});
  for(const g of G.filter(g=>g.h)){hN++;const j=ev.findIndex(e=>e.k==='n'&&e.m===g.m&&Math.abs(e.t/1000-g.t)<0.08);if(j>=0&&posH[j]&&posH[j].h)hOk++}
  const F=(a,b,c)=>(2*a/Math.max(1,2*a+b+c)).toFixed(2);
  console.log(n.padEnd(12),'音 F',G.length||P.length?F(tp,P.length-tp,G.length-tp):'-',`(正解${G.length} 当たり${tp} 余計${P.length-tp})`,'| 打音 F',GP.length||PP.length?F(tp2,PP.length-tp2,GP.length-tp2):'-',`(正解${GP.length} 当たり${tp2} 余計${PP.length-tp2})`,hN?`| ハーモニクス ${hOk}/${hN}`:'');
  tot.n[0]+=tp;tot.n[1]+=P.length-tp;tot.n[2]+=G.length-tp;tot.p[0]+=tp2;tot.p[1]+=PP.length-tp2;tot.p[2]+=GP.length-tp2;
}
const F=a=>(2*a[0]/Math.max(1,2*a[0]+a[1]+a[2])).toFixed(3);console.log('合計 音 F',F(tot.n),'| 打音 F',F(tot.p));
