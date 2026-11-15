import itertools
from jr2 import *
print('wicket cost (runs) by phase:',{k:round(v,1) for k,v in LAM.items()})
b,w=build(); FR={'bat':(b,'striker_id','bowler_id'),'bowl':(w,'bowler_id','striker_id')}
m=pd.read_csv('/home/claude/data/matches.csv').set_index('match_id'); order=m.loc[b.match_id.unique()].sort_values('match_date').index.values
splits={'odd>even':(order[::2],order[1::2]),'even>odd':(order[1::2],order[::2]),'early>late':(order[:len(order)//2],order[len(order)//2:]),'late>early':(order[len(order)//2:],order[:len(order)//2])}
Ds={k:prep(fr_,own,opp,'y') for k,(fr_,own,opp) in FR.items()}
def score(kind,alpha,xm,whp,sp_):
    D=Ds[kind]; A,B=splits[sp_]; inA=np.isin(D['match'],A); inB=np.isin(D['match'],B)
    L,X=fit(D,alpha,xm,whp,rowmult=inA.astype(float)); PL=L+X
    df=pd.DataFrame({'o':D['own'],'y':D['y']}); tb=df[inB&D['hp']].groupby('o').y.agg(['mean','size']); tb=tb[tb['size']>=50]
    nA=pd.Series(D['own'][inA&D['hp']]).value_counts(); idx=[i for i in tb.index if nA.get(i,0)>=30]
    return spearmanr(PL[idx],tb.loc[idx,'mean'])[0]
rows=[]
for a,xm,wh in itertools.product((200,500,1000,2000,5000,10000),(1,4,16,1e6),(1,2,3)):
    r={'alpha':a,'xmult':xm,'whp':wh}
    for kind in FR: r[kind]=np.mean([score(kind,a,xm,wh,s) for s in splits])
    r['both']=r['bat']+r['bowl']; rows.append(r)
G=pd.DataFrame(rows); G.to_pickle('grid2.pkl'); pd.set_option('display.width',200)
print(G.sort_values('both',ascending=False).head(14).round(3).to_string(index=False))
print('\nmarginals (mean over other params):')
for c in ('alpha','xmult','whp'): print(G.groupby(c)[['bat','bowl','both']].mean().round(3).T.to_string())
