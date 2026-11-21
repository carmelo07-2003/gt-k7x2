// AIの onset 行列の候補と正解を突き合わせ、見分けに使える特徴を調べる
const fs=require('fs');
const FPS=22050/256, ANNOT=86*2, WOFF=(256/22050)*(ANNOT-(22050*2-256)/256)+0.0018;
const f2t=f=>f*256/22050-WOFF*Math.floor(f/ANNOT);
function goertzelE(x,start,len,f,sr=22050){ // Hann窓つき
  const k=2*Math.cos(2*Math.PI*f/sr);let s1=0,s2=0;
  for(let i=0;i<len;i++){const j=start+i;if(j<0||j>=x.length)continue;const w=0.5-0.5*Math.cos(2*Math.PI*i/(len-1));const s0=x[j]*w+k*s1-s2;s2=s1;s1=s0}
  return Math.max(0,s1*s1+s2*s2-k*s1*s2);
}
function partialE(x,t,f0,dur,sr=22050){const st=Math.round(t*sr),len=Math.round(dur*sr);let e=0;for(let h=1;h<=6;h++){const f=h*f0;if(f>8000)break;e+=goertzelE(x,st,len,f)/h}return e}
const names=process.argv.slice(2);
const rows=[];
for(const n of names){
  const bp=JSON.parse(fs.readFileSync(`/home/claude/harness/data/${n}.bp.json`)),gt=JSON.parse(fs.readFileSync(`/home/claude/harness/data/${n}.json`));
  const O=bp.onsets,F=bp.frames,x=Float32Array.from(bp.x22),T=O.length;
  // 候補：各音高の onset の山
  const cands=[];
  for(let k=15;k<70;k++)for(let t=6;t<T-4;t++){const o=O[t][k];if(o>=0.1&&o>=O[t-1][k]&&o>O[t+1][k])cands.push({t,k,m:k+21,o,time:f2t(t)})}
  // 正解との対応（同じ高さ・±50ms）
  const used=new Set();
  for(const c of cands.sort((a,b)=>b.o-a.o)){let best=-1,bd=0.05;gt.notes.forEach((g,i)=>{if(used.has(i)||g.m!==c.m)return;const d=Math.abs(g.t-c.time);if(d<bd){bd=d;best=i}});c.ok=best>=0;if(best>=0){used.add(best);c.dt=c.time-gt.notes[best].t}}
  for(const c of cands){
    const pre=(F[c.t-3][c.k]+F[c.t-4]?.[c.k]+F[c.t-5]?.[c.k])/3;
    const post=(F[c.t+1][c.k]+F[c.t+2][c.k]+F[c.t+3][c.k])/3;
    const f0=440*2**((c.m-69)/12);
    const eb=partialE(x,c.time-0.05,f0,0.04),ea=partialE(x,c.time+0.0,f0,0.04);
    rows.push({n,ok:c.ok,o:c.o,pre,post,rise:Math.log10((ea+1e-9)/(eb+1e-9)),m:c.m,dt:c.dt});
  }
  console.log(n,'gt',gt.notes.length,'matched',used.size,'cands',cands.length);
}
const q=(a,p)=>{if(!a.length)return NaN;const s=[...a].sort((x,y)=>x-y);return +s[Math.floor(p*(s.length-1))].toFixed(3)};
for(const grp of [['fresh(pre<0.3)',r=>r.pre<0.3],['sounding(pre>=0.3)',r=>r.pre>=0.3]]){
  console.log('\n==',grp[0]);
  for(const ok of [true,false]){
    const R=rows.filter(r=>r.ok===ok&&grp[1](r));
    console.log(ok?'TRUE ':'FALSE',R.length,'o q10/50/90',q(R.map(r=>r.o),.1),q(R.map(r=>r.o),.5),q(R.map(r=>r.o),.9),'| rise',q(R.map(r=>r.rise),.1),q(R.map(r=>r.rise),.5),q(R.map(r=>r.rise),.9),'| post',q(R.map(r=>r.post),.1),q(R.map(r=>r.post),.5));
  }
}
const dts=rows.filter(r=>r.ok).map(r=>r.dt);console.log('\nonset time offset (AI-GT) median',q(dts,.5),'q10',q(dts,.1),'q90',q(dts,.9));
