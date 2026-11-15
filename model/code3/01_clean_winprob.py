import pandas as pd, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
D='/home/claude/data/'
b=pd.read_csv(D+'ball_by_ball.csv',low_memory=False)
m=pd.read_csv(D+'matches.csv').merge(pd.read_csv(D+'competitions.csv'),on='comp_id')

# ---------- STEP 1: CLEAN ----------
log=[]
m['is_ipl']=m.comp_name.str.startswith('IPL')
n0=len(m); FULL={18,25,28,17,21,22,27,20,19,72,78,23}   # ICC full members
m=m[m.is_ipl|((m.match_format=='T20I')&m.home_team_id.isin(FULL)&m.away_team_id.isin(FULL))]; log.append(('IPL + full-member T20I only',len(m)))
m=m[m.match_id.isin(b.match_id)]; log.append(('has ball-by-ball',len(m)))
m=m[m.win_team_id.notna()]; log.append(('has a winner (no tie/NR)',len(m)))
rev=m.revised_target.notna()|m.first_inn_revised_overs.notna()|m.second_inn_revised_overs.notna()
m=m[~rev]; log.append(('not rain-revised (DLS)',len(m)))
b=b[b.match_id.isin(m.match_id)&(b.innings<=2)].copy()   # drop super overs
b=b.sort_values(['match_id','innings','innings_ball_num']).reset_index(drop=True)

b['legal']=~b.extras_type.isin(['wide','Overstep','other'])
extra={'wide':1,'Overstep':1,'other':1,'byes':1,'legbyes':1,'Penalty':5}
b['team_runs']=b.runs+b.extras_type.map(extra).fillna(0)
b['wkt']=b.is_wicket.astype(int)
g=b.groupby(['match_id','innings'])
b['cum_runs']=g.team_runs.cumsum()-b.team_runs           # before this ball
b['cum_wkts']=g.wkt.cumsum()-b.wkt
b['legal_before']=g.legal.cumsum()-b.legal
b['balls_left']=120-b.legal_before
tot=g.agg(runs=('team_runs','sum'),wk=('wkt','sum'),lb=('legal','sum')).reset_index()
# keep only matches whose both innings look complete
t1=tot[tot.innings==1].set_index('match_id'); t2=tot[tot.innings==2].set_index('match_id')
ok=t1.join(t2,lsuffix='1',rsuffix='2',how='inner')
ok=ok[((ok.lb1>=118)|(ok.wk1>=10))]                       # inn1 completed
m=m.set_index('match_id'); w=m.win_team_id
bt2=b[b.innings==2].groupby('match_id').batting_team_id.first()
chase_won=(w.reindex(ok.index)==bt2.reindex(ok.index))
ok=ok[~(chase_won&False)]
good=ok.index
log.append(('both innings present & inn1 complete',len(good)))
b=b[b.match_id.isin(good)].copy()
inn1=ok.runs1
b['target']=b.match_id.map(inn1)+1
i2=b.innings==2
b['need']=np.where(i2,b.target-b.cum_runs,np.nan)
b['rrr']=np.where(i2,b.need/np.maximum(b.balls_left,1)*6,np.nan)
b['crr']=b.cum_runs/np.maximum(b.legal_before,1)*6
b['wkts_in_hand']=10-b.cum_wkts
b['won']=(b.batting_team_id==b.match_id.map(w)).astype(int)
b['is_ipl']=b.match_id.map(m.is_ipl).astype(int)
b['stage_ko']=b.match_id.map(m.knock_out).fillna(0).astype(int)
b['season']=b.match_id.map(m.season)
print(pd.DataFrame(log,columns=['filter','matches_left']).to_string(index=False))
print('balls kept',len(b),'matches',b.match_id.nunique(),'IPL',b.groupby('match_id').is_ipl.first().sum())
# score reconstruction sanity: winner by runs

# ---------- STEP 2 (v4): WIN PROB WITH TEAM + VENUE CONTEXT ----------
from sklearn.metrics import roc_auc_score, log_loss
# (a) pre-match Elo from earlier matches only
mm=m.copy(); mm['match_date']=pd.to_datetime(mm.match_date)
mm=mm.sort_values(['match_date']).reset_index()
elo={}; pre={}
for r in mm.itertuples():
    h,a=r.home_team_id,r.away_team_id
    eh,ea=elo.get(h,1500.),elo.get(a,1500.)
    pre[(r.match_id,h)]=eh; pre[(r.match_id,a)]=ea
    if r.win_team_id in (h,a):
        exp=1/(1+10**((ea-eh)/400)); res=1.0 if r.win_team_id==h else 0.0
        elo[h]=eh+20*(res-exp); elo[a]=ea-20*(res-exp)
b['bat_elo']=[pre.get((i,t),1500.) for i,t in zip(b.match_id,b.batting_team_id)]
b['bowl_elo']=[pre.get((i,t),1500.) for i,t in zip(b.match_id,b.bowling_team_id)]
b['elo_diff']=b.bat_elo-b.bowl_elo
# (b) venue par from earlier matches at same venue & format (shrunk to global mean)
mm['inn1']=mm.match_id.map(inn1)
mm=mm[mm.inn1.notna()]
vp={}; seen={}
for r in mm.itertuples():
    key=(r.venue_id,bool(r.is_ipl)); gm=mm[mm.is_ipl==r.is_ipl].inn1.mean()
    lst=seen.get(key,[]); vp[r.match_id]=(sum(lst)+4*gm)/(len(lst)+4)
    seen.setdefault(key,[]).append(r.inn1)
b['venue_par']=b.match_id.map(vp)
# (c) base features
b['crr']=(b.cum_runs+8*2)/(b.legal_before+12)*6
b['rr_gap']=b.rrr-b.crr
b['resources']=b.wkts_in_hand*b.balls_left/120
b['proj']=b.cum_runs+b.crr*b.balls_left/6*(0.5+0.05*b.wkts_in_hand)
gkf=GroupKFold(n_splits=5)
# (d) learned projected final score (innings 1 only, out-of-fold)
pf=['balls_left','cum_runs','cum_wkts','crr','resources','venue_par','elo_diff','is_ipl']
b['inn1_total']=b.match_id.map(inn1); b['proj_final']=np.nan
i1=b.index[b.innings==1]
for tr,te in gkf.split(b.loc[i1,pf],groups=b.loc[i1,'match_id']):
    r_=HistGradientBoostingRegressor(max_iter=150,learning_rate=0.05,max_leaf_nodes=8,min_samples_leaf=500,random_state=0)
    r_.fit(b.loc[i1[tr],pf],b.loc[i1[tr],'inn1_total']); b.loc[i1[te],'proj_final']=r_.predict(b.loc[i1[te],pf])
b['proj_vs_par']=b.proj_final-b.venue_par
b['target_vs_par']=np.where(b.innings==2,b.target-b.venue_par,np.nan)
print('proj_final MAE (inn1, OOF): %.1f runs'%(b[b.innings==1].proj_final-b[b.innings==1].inn1_total).abs().mean())
def run(feats,tag):
    X=b[feats]; y=b.won; out=np.full(len(b),np.nan)
    for tr,te in gkf.split(X,y,b.match_id):
        c=HistGradientBoostingClassifier(max_iter=150,learning_rate=0.05,max_leaf_nodes=8,min_samples_leaf=800,l2_regularization=5.0,random_state=0).fit(X.iloc[tr],y.iloc[tr])
        out[te]=c.predict_proba(X.iloc[te])[:,1]
    o=pd.Series(out,index=b.index)
    print('%-22s AUC all %.3f | inn1 %.3f | inn2 %.3f | logloss %.3f'%(tag,roc_auc_score(y,o),roc_auc_score(y[b.innings==1],o[b.innings==1]),roc_auc_score(y[b.innings==2],o[b.innings==2]),log_loss(y,o)))
    return o
base=['innings','balls_left','cum_wkts','crr','need','rrr','rr_gap','resources','proj','is_ipl']
run(base,'old (state only)')
run(base+['elo_diff'],'+ team Elo')
run(base+['venue_par','target_vs_par'],'+ venue par')
b['wp']=run(base+['elo_diff','venue_par','target_vs_par','proj_final','proj_vs_par'],'+ all (final)')
print(b.groupby(pd.cut(b.wp,[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1])).agg(pred=('wp','mean'),actual=('won','mean')).round(3).T.to_string())
print(b[b.innings==1].groupby(pd.cut(b[b.innings==1].wp,[0,.2,.35,.5,.65,.8,1])).agg(pred=('wp','mean'),actual=('won','mean'),n=('won','size')).round(3).T.to_string())
b.to_pickle('balls_pressure_v4.pkl')
