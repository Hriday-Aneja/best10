import warnings; warnings.filterwarnings('ignore')
from jr import prep, fit
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
bat=pd.read_pickle('bat_v.pkl'); bw=pd.read_pickle('bowl_v.pkl')
# wicket cost in runs = expected remaining runs lost when a wicket falls (resource approach), by phase, from innings-1 balls
i1=bw[bw.innings==1].copy(); i1['rem']=i1.groupby('match_id').team_runs.transform('sum')-i1.cum_runs
fr=['balls_left','cum_wkts','venue_par','is_ipl']
rr=HistGradientBoostingRegressor(max_iter=200,learning_rate=0.06,max_leaf_nodes=12,min_samples_leaf=400,monotonic_cst=[1,-1,0,0],random_state=0).fit(i1[fr],i1.rem)
x1=i1[fr].copy(); x1['cum_wkts']=np.minimum(x1.cum_wkts+1,10); i1['wc']=rr.predict(i1[fr])-rr.predict(x1)
LAM=i1.groupby('phase',observed=True).wc.mean().to_dict()
def build(lscale=1.0):
    lb=bat.phase.map(LAM).astype(float)*lscale; lw=bw.phase.map(LAM).astype(float)*lscale
    b=bat.copy(); w=bw.copy()
    b['y']=(b.runs-b.E_r)-lb*(b.batter_out-b.E_d); w['y']=(w.E_r-w.charged)+lw*(w.bowler_wkt-w.E_w)
    return b,w
