import sys,json,glob,os
from playwright.sync_api import sync_playwright
os.chdir('/home/claude/harness')
names=sys.argv[1:] or sorted(os.path.basename(p)[:-4] for p in glob.glob('data/*.wav'))
with sync_playwright() as p:
    b=p.chromium.launch(executable_path='/usr/bin/google-chrome',args=['--enable-unsafe-swiftshader'])
    pg=b.new_page()
    pg.goto('http://localhost:8766/bp.html')
    for n in names:
        s=pg.evaluate(f"runBP('{n}')")
        open(f'data/{n}.bp.json','w').write(s)
        d=json.loads(s);print(n,len(d['frames']),'frames',round(d['ms']),'ms')
    b.close()
