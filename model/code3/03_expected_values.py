import pandas as pd, numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
b=pd.read_pickle('balls_pi.pkl')
LAM=20.0   # runs value of one wicket (sensitivity tested later)
b['ps']=(b.pace_or_spin=='Pace').astype(float).where(b.pace_or_spin.notna())
b['batter_out']=((b.wkt==1)&(b.dismissed_player_id==b.striker_id)).astype(int)
b['bowler_wkt']=((b.wkt==1)&(~b.dismissal_type.isin(['RunOut','Retire Out','Obstracting the field','Obstructing the field']))).astype(int)
b['charged']=b.runs+b.extras_type.isin(['wide','Overstep','other']).astype(int)   # runs charged to bowler (no byes/leg-byes)
ctx=['innings','legal_before','balls_left','cum_wkts','crr','rrr','need','is_ipl','venue_par','elo_diff','PI','wp']
def fit_pred(df,feats,target,clf=False):
    M=(HistGradientBoostingClassifier if clf else HistGradientBoostingRegressor)(max_iter=120,learning_rate=0.06,max_leaf_nodes=12,min_samples_leaf=500,l2_regularization=3,random_state=0)
    M.fit(df[feats],df[target]); return M.predict_proba(df[feats])[:,1] if clf else M.predict(df[feats])
# ---- batting: faced deliveries (all but wides)
bat=b[b.extras_type!='wide'].copy()
f=ctx+['ps']
bat['E_r']=fit_pred(bat,f,'runs'); bat['E_d']=fit_pred(bat,f,'batter_out',True)
bat['v']=(bat.runs-bat.E_r)-LAM*(bat.batter_out-bat.E_d)
# ---- bowling: all deliveries
bw=b.copy(); bw['PI']=bw.PI_bowl; bw['hp']=bw.hp_bowl; f2=ctx+['batting_order']
bw['E_r']=fit_pred(bw,f2,'charged'); bw['E_w']=fit_pred(bw,f2,'bowler_wkt',True)
bw['v']=(bw.E_r-bw.charged)+LAM*(bw.bowler_wkt-bw.E_w)
bat.to_pickle('bat_v.pkl'); bw.to_pickle('bowl_v.pkl')
for nm,d,pid in (('BAT',bat,'striker_id'),('BOWL',bw,'bowler_id')):
    g=d[d.hp].groupby(pid).agg(n=('v','size'),m=('match_id','nunique'))
    print(nm,'players with HP balls >=',{t:int((g.n>=t).sum()) for t in (30,60,100,150,200,300)})
    print('  per-ball v: mean %.4f sd %.3f ; HP mean %.4f ; ordinary mean %.4f'%(d.v.mean(),d.v.std(),d[d.hp].v.mean(),d[~d.hp].v.mean()))
