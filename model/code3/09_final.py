import warnings; warnings.filterwarnings('ignore')
from jr2 import *
import pickle
rng=np.random.default_rng(2027)
P0=dict(alpha=2000.,xm=4.,whp=1.)
Fr={1.0:build(1.0)}
for ls in (0.5,1.5,2.0): Fr[ls]=build(ls)
old={'bat':pd.read_pickle('../work2/bat_v.pkl').hp,'bowl':pd.read_pickle('../work2/bowl_v.pkl').hp}
def mk(fr,kind):
    q=lambda p:fr.groupby('innings').PI.transform(lambda s:s.quantile(p))
    return {'base':fr.hp,'top30':fr.PI>=q(.70),'top15':fr.PI>=q(.85),'excl overs 1-3':(fr.PI>=q(.80))&(fr.overid>3),'late & close (WP 25-75%)':(fr.balls_left<=48)&fr.wp.between(.25,.75),'previous definition (v2)':old[kind].reindex(fr.index)}
hpv={'bat':mk(Fr[1.0][0],'bat'),'bowl':mk(Fr[1.0][1],'bowl')}
FM={'bat':('striker_id','bowler_id','striker_name'),'bowl':('bowler_id','striker_id','bowler_name')}
Ds={}
def getD(kind,hn,ls):
    k=(kind,hn,ls)
    if k not in Ds:
        fr=Fr[ls][0] if kind=='bat' else Fr[ls][1]; own,opp,_=FM[kind]; Ds[k]=prep(fr,own,opp,'y',hp=hpv[kind][hn].reindex(fr.index))
    return Ds[k]
fr0={'bat':Fr[1.0][0],'bowl':Fr[1.0][1]}
CNT={};POOL={};names={}
for kind in FM:
    own=FM[kind][0]; g=fr0[kind][fr0[kind].hp].groupby(own).agg(n=('hp','size'),m=('match_id','nunique')); CNT[kind]=g; POOL[kind]=g[(g.n>=150)&(g.m>=12)].index
    names[kind]=fr0[kind].groupby(own)[FM[kind][2]].agg(lambda x:x.mode().iloc[0]); print(kind,'eligible',len(POOL[kind]))
def ato_series(fr,own,bowler):
    f=fr.sort_values(['match_id','innings','legal_before']).copy()
    f['wp_next']=f.groupby(['match_id','innings']).wp.shift(-1); last=f.groupby(['match_id','innings']).cumcount(ascending=False)==0
    f.loc[last&(f.innings==2),'wp_next']=f.loc[last&(f.innings==2),'won']; f=f[f.wp_next.notna()]
    d=f.wp_next-f.wp
    if bowler: mask=f.wp>0.65; d=-d
    else: mask=f.wp<0.35
    g=pd.DataFrame({'o':f[own],'w':d})[mask].groupby('o').w.agg(['sum','size']); return g['sum']/(g['size']+120.)
ATO={k:ato_series(fr0[k],FM[k][0],k=='bowl') for k in FM}
def rate(kind,hn='base',ls=1.0,alpha=None,xm=None,whp=None,pool=None,fwm=None,mult=None,ret=False,ato_w=0.10):
    alpha=P0['alpha'] if alpha is None else alpha; xm=P0['xm'] if xm is None else xm; whp=P0['whp'] if whp is None else whp
    D=getD(kind,hn,ls); fw=None
    if fwm=='ipl': fw=D['ipl'].astype(float)
    elif fwm=='t20i': fw=(~D['ipl']).astype(float)
    elif fwm=='t20i_half': fw=np.where(D['ipl'],1.0,0.5)
    rm=None if mult is None else mult.reindex(D['match']).fillna(0).values
    L,X=fit(D,alpha,xm,whp,rowmult=rm,fw=fw); L=pd.Series(L,index=D['ids']); X=pd.Series(X,index=D['ids'])
    pool=POOL[kind] if pool is None else pool; z=lambda v:(v-v.mean())/v.std()
    pl=(L+X)[pool]; sc=((1-ato_w)*z(pl)+ato_w*z(ATO[kind].reindex(pool).fillna(0))).sort_values(ascending=False)
    return (sc,L,X) if ret else sc
SPEC={'base':{},'wicket cost x0.5':dict(ls=.5),'wicket cost x1.5':dict(ls=1.5),'wicket cost x2':dict(ls=2.0),'pressure wt 2x':dict(whp=2),'pressure wt 3x':dict(whp=3),
 'X penalty 1x':dict(xm=1),'X penalty 4x':dict(xm=4),'X penalty 16x':dict(xm=16),'X off (level only)':dict(xm=1e6),'shrinkage 1000':dict(alpha=1000.),'shrinkage 5000':dict(alpha=5000.),
 'pressure = top 30%':dict(hn='top30'),'pressure = top 15%':dict(hn='top15'),'pressure excl. overs 1-3':dict(hn='excl overs 1-3'),'pressure = late & close':dict(hn='late & close (WP 25-75%)'),'T20I weight 0.5':dict(fwm='t20i_half'),'against-the-odds 0%':dict(ato_w=0.0),'against-the-odds 20%':dict(ato_w=0.2),'previous pressure definition (v2)':dict(hn='previous definition (v2)')}
V={}
for kind in FM:
    V[kind]={k:rate(kind,**p) for k,p in SPEC.items()}; base=V[kind]['base']; top=list(base.index[:10])
    print('\n=====',kind.upper(),'sensitivity (same eligible pool) =====')
    for k,s in V[kind].items():
        ix=base.index.intersection(s.index); print('%-26s top10 overlap %2d/10  rank corr %.2f'%(k,len(set(s.index[:10])&set(top)),pd.concat([base[ix],s[ix]],axis=1).corr('spearman').iloc[0,1]))
    fr=fr0[kind]; own=FM[kind][0]
    for lab,fm,mask in (('IPL only','ipl',fr.is_ipl==1),('T20I only','t20i',fr.is_ipl==0)):
        n=fr[fr.hp&mask].groupby(own).size(); pl=n[n>=100].index; s=rate(kind,fwm=fm,pool=pl); V[kind][lab]=s; ix=base.index.intersection(s.index)
        print('%-26s pool %3d; top10 overlap w/ base %2d/10; rank corr %.2f'%(lab,len(pl),len(set(s.index[:10])&set(top)),pd.concat([base[ix],s[ix]],axis=1).corr('spearman').iloc[0,1]))
    for lab,(mn,mm) in {'eligibility 120 balls':(120,10),'eligibility 200 balls':(200,12)}.items():
        g=CNT[kind]; pl=g[(g.n>=mn)&(g.m>=mm)].index; s=rate(kind,pool=pl); V[kind][lab]=s; print('%-26s pool %3d; top10 overlap w/ base %2d/10'%(lab,len(pl),len(set(s.index[:10])&set(top))))
# bootstrap (fixed pool)
ms=np.array(sorted(Fr[1.0][0].match_id.unique())); B=300; BOOT={}
for kind in FM:
    pool=POOL[kind]; R=pd.DataFrame(index=pool,columns=range(B),dtype=float)
    for b in range(B):
        c=pd.Series(np.bincount(rng.integers(0,len(ms),len(ms)),minlength=len(ms)).astype(float),index=ms); R[b]=rate(kind,mult=c).rank(ascending=False).reindex(pool)
    BOOT[kind]=pd.DataFrame({'boot_median':R.median(axis=1),'p05':R.quantile(.05,axis=1),'p95':R.quantile(.95,axis=1),'P_top10':(R<=10).mean(axis=1),'P_top5':(R<=5).mean(axis=1),'P_top3':(R<=3).mean(axis=1)})
# validity vs baselines
m=pd.read_csv('/home/claude/data/matches.csv').set_index('match_id'); order=m.loc[ms].sort_values('match_date').index.values
splits={'odd>even':(order[::2],order[1::2]),'even>odd':(order[1::2],order[::2]),'early>late':(order[:len(order)//2],order[len(order)//2:]),'late>early':(order[len(order)//2:],order[:len(order)//2])}
VAL=[]
for kind in FM:
    D=getD(kind,'base',1.0); hp=D['hp']; df=pd.DataFrame({'o':D['own'],'y':D['y']})
    for sp_,(A,Bm) in splits.items():
        inA=np.isin(D['match'],A); inB=np.isin(D['match'],Bm); tb=df[inB&hp].groupby('o').y.agg(['mean','size']); tb=tb[tb['size']>=50]
        nA=pd.Series(D['own'][inA&hp]).value_counts(); idx=[i for i in tb.index if nA.get(i,0)>=30]; act=tb.loc[idx,'mean']
        r={'role':kind,'split':sp_,'n_players':len(idx)}
        for lab,xm in (('final model (L+X)',P0['xm']),('level only (X off)',1e6)):
            L,X=fit(D,P0['alpha'],xm,1.0,rowmult=inA.astype(float)); r[lab]=spearmanr((L+X)[idx],act)[0]
        r['raw pressure average']=spearmanr(df[inA&hp].groupby('o').y.mean().reindex(idx).fillna(0),act)[0]
        r['raw all-ball average']=spearmanr(df[inA].groupby('o').y.mean().reindex(idx).fillna(0),act)[0]
        VAL.append(r)
VAL=pd.DataFrame(VAL); VAL.to_csv('validity2.csv',index=False); pd.set_option('display.width',220)
print('\nOut-of-sample validity (Spearman, predicted vs held-out pressure net value):'); print(VAL.groupby('role')[['final model (L+X)','level only (X off)','raw pressure average','raw all-ball average']].agg(['mean','min','max']).round(2).to_string())
pickle.dump(dict(V=V,BOOT=BOOT,POOL=POOL,names=names,CNT=CNT),open('stage2.pkl','wb')); print('done')
