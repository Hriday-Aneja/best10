import warnings; warnings.filterwarnings('ignore')
exec(open('setup_model.py').read())
import pickle
from scipy.stats import spearmanr
ms=np.array(sorted(Fr[1.0][0].match_id.unique()))
M=pd.read_csv('/home/claude/data/matches.csv').set_index('match_id'); M['match_date']=pd.to_datetime(M.match_date)
order=M.loc[ms].sort_values('match_date').index.values
z=lambda v:(v-v.mean())/v.std()
# ---------- per-ball + per-player-match tables for ATO and CLUTCH-WIN component ----------
Q={};PM={}
for kind in FM:
    fr=fr0[kind]; own=FM[kind][0]; bowl=kind=='bowl'
    f=fr[['match_id','innings','legal_before','wp','won',own]].sort_values(['match_id','innings','legal_before']).copy()
    f['wpn']=f.groupby(['match_id','innings']).wp.shift(-1); last=f.groupby(['match_id','innings']).cumcount(ascending=False)==0
    l2=last&(f.innings==2); f.loc[l2,'wpn']=f.loc[l2,'won']; f=f[f.wpn.notna()]
    sgn=-1 if bowl else 1
    f['ow']=f.wp if not bowl else 1-f.wp; f['d']=sgn*(f.wpn-f.wp); f['ownwon']=f.won if not bowl else 1-f.won
    f=f[f.ow<0.35].rename(columns={own:'o'})          # own team disadvantaged (<35% win prob)
    Q[kind]=f[['o','match_id','d']]
    g=f.groupby(['o','match_id']).agg(n=('d','size'),wpa=('d','sum'),minow=('ow','min'),won=('ownwon','first')).reset_index()
    g=g[g.n>=6]; g['val']=np.where((g.won==1)&(g.wpa>0),g.wpa*(1-g.minow),0.0)   # hard-won innings: contribution x difficulty
    PM[kind]=g
def comp(kind,mult=None):
    q=Q[kind]; g=PM[kind]
    wq=np.ones(len(q)) if mult is None else mult.reindex(q.match_id).fillna(0).values
    wg=np.ones(len(g)) if mult is None else mult.reindex(g.match_id).fillna(0).values
    a=pd.DataFrame({'o':q.o.values,'s':q.d.values*wq,'n':wq}).groupby('o').sum(); ato=a.s/(a.n+120.)
    c=pd.DataFrame({'o':g.o.values,'s':g.val.values*wg,'n':wg,'h':((g.val.values>0)*wg)}).groupby('o').sum(); clu=c.s/(c.n+8.)
    return ato,clu,c.h
def blend(pl,ato,clu,w,pool):
    cl=0.5*z(ato.reindex(pool).fillna(0))+0.5*z(clu.reindex(pool).fillna(0))
    return ((1-w)*z(pl.reindex(pool))+w*cl)
WTS={'orig: ATO 10%':None,'clutch 20%':.2,'clutch 30%':.3,'clutch 40%':.4}
def score(kind,L,X,ato,clu,wname,pool):
    pl=L+X
    if WTS[wname] is None: return (0.9*z(pl.reindex(pool))+0.1*z(ato.reindex(pool).fillna(0))).sort_values(ascending=False)
    return blend(pl,ato,clu,WTS[wname],pool).sort_values(ascending=False)
OUT={}
# ================= PART 1: rankings under higher clutch weights =================
for kind in FM:
    pool=POOL[kind]; _,L,X=rate(kind,ret=True); ato,clu,hn=comp(kind)
    R={k:score(kind,L,X,ato,clu,k,pool) for k in WTS}; OUT[kind]=dict(R=R,L=L,X=X,ato=ato,clu=clu,hn=hn)
    print('\n=====',kind.upper(),'top-10 by clutch weight =====')
    print(pd.DataFrame({k:[names[kind][i] for i in s.index[:10]] for k,s in R.items()},index=range(1,11)).to_string())
    base=R['orig: ATO 10%']
    for k,s in R.items(): print('%-16s top10 overlap vs orig %2d/10 ; rank corr %.2f'%(k,len(set(s.index[:10])&set(base.index[:10])),pd.concat([base,s],axis=1).corr('spearman').iloc[0,1]))
# ================= PART 2: validation =================
sp={'odd>even':(order[::2],order[1::2]),'even>odd':(order[1::2],order[::2]),'early>late':(order[:len(order)//2],order[len(order)//2:]),'late>early':(order[len(order)//2:],order[:len(order)//2])}
pre=M.loc[order][M.loc[order].match_date<'2026-01-01'].index.values; post=M.loc[order][M.loc[order].match_date>='2026-01-01'].index.values
sp['pre2026>2026']=(pre,post); print('\nmatches pre2026 %d, 2026 %d'%(len(pre),len(post)))
rows=[]; top_rows=[]
for kind in FM:
    D=getD(kind,'base',1.0); hp=D['hp']; df=pd.DataFrame({'o':D['own'],'y':D['y']}); ids=D['ids']
    for sn,(A,B) in sp.items():
        inA=np.isin(D['match'],A); inB=np.isin(D['match'],B)
        mA=pd.Series(1.0,index=A); 
        L,X=fit(D,P0['alpha'],P0['xm'],1.0,rowmult=inA.astype(float)); pl=pd.Series(L+X,index=np.arange(len(ids)))
        ato,clu,_=comp(kind,mA); ato.index=pd.Index(ato.index); 
        # map player ids -> index positions
        pos=pd.Series(np.arange(len(ids)),index=ids)
        tb=df[inB&hp].groupby('o').y.agg(['mean','size']); tb=tb[tb['size']>=50]
        nA=pd.Series(D['own'][inA&hp]).value_counts(); idx=[i for i in tb.index if nA.get(i,0)>=30]
        act=tb.loc[idx,'mean']; pid=ids[idx]
        a_=ato.reindex(pid).fillna(0).values; c_=clu.reindex(pid).fillna(0).values; p_=pl[idx].values
        r={'role':kind,'split':sn,'n':len(idx),'L+X':spearmanr(p_,act)[0],'ATO only':spearmanr(a_,act)[0],'clutch-win only':spearmanr(c_,act)[0]}
        zz=lambda v:(v-v.mean())/v.std()
        r['orig 90/10']=spearmanr(0.9*zz(p_)+0.1*zz(a_),act)[0]
        for w in (.2,.3,.4): r['clutch %d%%'%(w*100)]=spearmanr((1-w)*zz(p_)+w*(0.5*zz(a_)+0.5*zz(c_)),act)[0]
        r['raw all-ball avg']=spearmanr(df[inA].groupby('o').y.mean().reindex(idx).fillna(0),act)[0]
        rows.append(r)
        # top-10 precision: held-out pressure value of model's top-10 vs eligible-pool mean
        sc=pd.Series(0.9*zz(p_)+0.1*zz(a_),index=idx).sort_values(ascending=False)
        top_rows.append({'role':kind,'split':sn,'top10_heldout_per100':act[sc.index[:10]].mean()*100,'pool_heldout_per100':act.mean()*100,'bottom10_per100':act[sc.index[-10:]].mean()*100})
V=pd.DataFrame(rows); T=pd.DataFrame(top_rows); pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
print('\nOUT-OF-SAMPLE Spearman (pred from train matches vs held-out pressure net value):'); print(V.round(3).to_string(index=False))
print('\nMEAN by role:'); print(V.groupby('role').mean(numeric_only=True).drop(columns='n').round(3).to_string())
print('\nTop-10 vs pool vs bottom-10 held-out net runs/100 pressure balls:'); print(T.round(2).to_string(index=False)); print(T.groupby('role').mean(numeric_only=True).round(2).to_string())
V.to_csv('validity_new.csv',index=False); T.to_csv('top10_precision.csv',index=False)
# split-half reliability (odd vs even matches)
print('\nSPLIT-HALF RELIABILITY (odd vs even matches, pool players with HP balls>=30 each half)')
for kind in FM:
    D=getD(kind,'base',1.0); ids=D['ids']; A,B=sp['odd>even']; pos=pd.Series(np.arange(len(ids)),index=ids)
    res={}
    for tag,S in (('A',A),('B',B)):
        inS=np.isin(D['match'],S); L,X=fit(D,P0['alpha'],P0['xm'],1.0,rowmult=inS.astype(float)); mS=pd.Series(1.0,index=S); ato,clu,_=comp(kind,mS)
        cnt=pd.Series(D['own'][inS&D['hp']]).value_counts(); res[tag]=(pd.Series(L+X,index=ids),ato,clu,cnt)
    ok=[i for i in res['A'][3].index if res['A'][3].get(i,0)>=30 and res['B'][3].get(i,0)>=30]; pid=ids[ok]
    for nm,j in (('L+X rating',0),('ATO',1),('clutch-win',2)):
        a=res['A'][j].reindex(pid).fillna(0) if j else res['A'][0].reindex(pid); b=res['B'][j].reindex(pid).fillna(0) if j else res['B'][0].reindex(pid)
        r=spearmanr(a,b)[0]; print('%s %-11s n=%d  half-half rho %.2f -> Spearman-Brown full-sample reliability %.2f'%(kind,nm,len(pid),r,2*r/(1+r)))
# ================= PART 3: does the pressure flag separate pressure from ordinary? =================
print('\nPRESSURE DEFINITION CHECK: high-pressure (HP) vs ordinary balls')
for kind in FM:
    fr=fr0[kind].sort_values(['match_id','innings','legal_before']).copy(); fr['wpn']=fr.groupby(['match_id','innings']).wp.shift(-1); fr=fr[fr.wpn.notna()]
    fr['lev']=(fr.wpn-fr.wp).abs()
    if kind=='bat': fr['rr']=fr.runs; fr['wk']=fr.batter_out
    else: fr['rr']=fr.charged; fr['wk']=fr.bowler_wkt
    t=fr.groupby('hp').agg(balls=('rr','size'),runs_per_ball=('rr','mean'),wkts_per_100=('wk',lambda s:s.mean()*100),leverage_abs_dWP=('lev','mean'),mean_wp=('wp','mean'),close_share=('wp',lambda s:s.between(.25,.75).mean()))
    print(kind); print(t.round(4).to_string())
pickle.dump(dict(OUT=OUT,V=V,T=T),open('clutch_out.pkl','wb')); print('DONE part1-3')
