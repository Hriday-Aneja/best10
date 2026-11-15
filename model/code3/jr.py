import numpy as np, pandas as pd, scipy.sparse as sp, time
from scipy.sparse.linalg import lsqr
from scipy.stats import spearmanr
ALPHA={('bat','runs'):647.,('bat','risk'):1480.,('bowl','runs'):372.,('bowl','risk'):1337.}   # k = sigma^2/tau^2 measured earlier
def load():
    bat=pd.read_pickle('bat_v.pkl'); bw=pd.read_pickle('bowl_v.pkl')
    bat['y_runs']=bat.runs-bat.E_r;  bat['y_risk']=-(bat.batter_out-bat.E_d)
    bw['y_runs']=bw.E_r-bw.charged;  bw['y_risk']=bw.bowler_wkt-bw.E_w
    return bat,bw
def prep(frame,own,opp,ycol,hp=None):
    oi,own_idx=np.unique(frame[own],return_inverse=True); qi,opp_idx=np.unique(frame[opp],return_inverse=True)
    h=(frame.hp if hp is None else hp).values.astype(bool)
    return dict(ids=oi,own=own_idx,opp=opp_idx,nopp=len(qi),y=frame[ycol].values.astype(float),hp=h,match=frame.match_id.values,ipl=frame.is_ipl.values.astype(bool))
def fit(D,alpha,xmult,whp,rowmult=None,fw=None):
    P=len(D['ids']); n=len(D['y']); w=np.where(D['hp'],whp,1.0)
    if rowmult is not None: w=w*rowmult
    if fw is not None: w=w*fw
    sw=np.sqrt(w); sX=np.sqrt(1.0/xmult); rr=np.arange(n); H=D['hp']
    A=sp.csr_matrix((np.concatenate([np.ones(n),np.full(H.sum(),sX),np.ones(n)]),(np.concatenate([rr,rr[H],rr]),np.concatenate([D['own'],P+D['own'][H],2*P+D['opp']]))),shape=(n,2*P+D['nopp']))
    sol=lsqr(sp.diags(sw)@A,sw*D['y'],damp=np.sqrt(alpha),atol=1e-7,btol=1e-7,iter_lim=400)[0]
    return sol[:P],sol[P:2*P]*sX
