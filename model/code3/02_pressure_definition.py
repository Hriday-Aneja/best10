import pandas as pd, numpy as np
b=pd.read_pickle('balls_pressure_v4.pkl'); clip=lambda s:np.clip(s,0,1)
b['phase']=pd.cut(b.overid,[0,6,15,20],labels=['PP','Mid','Death'])
b['C']=1-(2*b.wp-1).abs()                                   # contest still live
b['stage']=1-b.balls_left/120; b['W']=clip(b.cum_wkts/10); b['Wh']=1-b.W
bins=pd.cut(b.legal_before,range(0,125,6),right=False); m1=(b.innings==1)
b['par_w']=bins.map(b[m1].groupby(bins[m1]).cum_wkts.mean()).astype(float); b['collapse']=clip((b.cum_wkts-b.par_w)/3)
share=(b[m1].cum_runs/b[m1].inn1_total).groupby(bins[m1]).mean()
b['par_cum']=b.venue_par*bins.map(share).astype(float)      # venue-adjusted par score at this stage
b['behind']=clip((b.par_cum-b.cum_runs)/20); b['ahead']=clip((b.cum_runs-b.par_cum)/20)
b['demand']=clip((b.rrr/(b.venue_par/20)-1)/0.5)            # chase: required rate vs what this ground normally scores
b['flat']=clip((b.venue_par-165)/45)                        # flat / high-scoring ground: tempo expected from ball 1
# BATTING pressure: chase demand vs par, behind par pace, collapse, flat-pitch tempo
b['PI']=np.where(b.innings==2,b.C*(0.25*b.stage+0.45*b.demand+0.30*b.W),b.C*(0.20*b.stage+0.30*b.behind+0.25*b.collapse+0.25*b.flat)).astype(float)
# BOWLING pressure: batting side on top and needs breaking, in every phase; death overs weigh most (stage^1.5)
b['Cb']=clip(1-(b.wp-0.6).abs()/0.6); b['bat_top']=clip((b.wp-0.4)/0.4); sb=b.stage**1.5
b['PI_bowl']=np.where(b.innings==2,b.Cb*(0.35*sb+0.35*b.bat_top+0.30*b.Wh),b.C*(0.30*sb+0.30*b.ahead+0.25*b.Wh+0.15*b.flat)).astype(float)
for c,h in (('PI','hp'),('PI_bowl','hp_bowl')): b[h]=b.groupby('innings')[c].transform(lambda s:s>=s.quantile(.8))
b['legal_f']=b.legal.astype(bool); b.to_pickle('balls_pi.pkl')
print('flagged share within each phase (%):'); print(pd.concat({'bat':b.groupby(['innings','phase'],observed=True).hp.mean()*100,'bowl':b.groupby(['innings','phase'],observed=True).hp_bowl.mean()*100},axis=1).round(1).to_string())
