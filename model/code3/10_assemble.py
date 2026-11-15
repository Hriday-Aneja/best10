import warnings; warnings.filterwarnings('ignore')
src=open('09_final.py').read().split('SPEC={')[0]; ns={}; exec(src,ns)      # reuse model setup (no refits of sensitivity/bootstrap)
import pickle, numpy as np, pandas as pd
S=pickle.load(open('stage2.pkl','rb')); V,BOOT=S['V'],S['BOOT']; rate=ns['rate']; fr0=ns['fr0']; FM=ns['FM']; names=ns['names']; LAM=ns['LAM']
SPECS=['base','wicket cost x0.5','wicket cost x1.5','wicket cost x2','pressure wt 2x','pressure wt 3x','X penalty 1x','X penalty 4x','X penalty 16x','X off (level only)','shrinkage 1000','shrinkage 5000','pressure = top 30%','pressure = top 15%','pressure excl. overs 1-3','pressure = late & close','T20I weight 0.5','against-the-odds 0%','against-the-odds 20%','previous pressure definition (v2)']
pd.set_option('display.width',260); FIN={}
for kind in FM:
    fr=fr0[kind]; own=FM[kind][0]; base=V[kind]['base']; pool=base.index
    s,L,X=rate(kind,ret=True); ranks=pd.DataFrame({k:V[kind][k].rank(ascending=False) for k in SPECS}).reindex(pool)
    t=pd.DataFrame({'name':names[kind].reindex(pool).values,'base_rank':range(1,len(pool)+1),'rating_per100':(L+X).reindex(pool).values*100,'ato_wpa_per_ball':ns['ATO'][kind].reindex(pool).fillna(0).values,'L_per100':L.reindex(pool).values*100,'X_per100':X.reindex(pool).values*100},index=pool)
    t['mean_rank_variants']=ranks.mean(axis=1); t['n_variants_top10']=(ranks<=10).sum(axis=1); t=t.join(BOOT[kind])
    t['consensus']=0.5*t.mean_rank_variants.rank()+0.5*t.boot_median.rank(); t=t.sort_values(['consensus','base_rank']); t['final_rank']=range(1,len(t)+1)
    h=fr[fr.hp]; n=fr[~fr.hp]
    if kind=='bat':
        g=lambda d:d.groupby(own).agg(balls=('runs','size'),runs=('runs','sum'),outs=('batter_out','sum')); a,b=g(h),g(n)
        t['pressure_balls']=a.balls; t['pressure_SR']=a.runs/a.balls*100; t['usual_SR']=b.runs/b.balls*100; t['pressure_runs']=a.runs; t['pressure_outs']=a.outs; t['pressure_dis%']=a.outs/a.balls*100; t['usual_dis%']=b.outs/b.balls*100
    else:
        g=lambda d:d.assign(Lg=(~d.extras_type.isin(['wide','Overstep','other']))).groupby(own).agg(legal=('Lg','sum'),r=('charged','sum'),w=('bowler_wkt','sum')); a,b=g(h),g(n)
        t['pressure_balls']=a.legal; t['pressure_econ']=a.r/a.legal*6; t['usual_econ']=b.r/b.legal*6; t['pressure_wkts']=a.w; t['pressure_wkts_per_over']=a.w/a.legal*6; t['usual_wkts_per_over']=b.w/b.legal*6
    ip=h.groupby([own,'is_ipl']).size().unstack(fill_value=0); t['IPL_pressure_deliveries']=ip.get(1); t['T20I_pressure_deliveries']=ip.get(0)
    hv=h.groupby([own,'match_id']).agg(s=('y','sum'),nb=('y','size')).reset_index(); hv=hv[hv.nb>=6]
    t['consistency%']=hv.groupby(own).s.apply(lambda x:(x>0).mean()*100); t['pressure_innings']=hv.groupby(own).size()
    FIN[kind]=t; t.to_csv(f'final2_{kind}_full.csv')
cols=['final_rank','name','base_rank','pressure_balls','IPL_pressure_deliveries','T20I_pressure_deliveries','L_per100','X_per100','rating_per100','consistency%','P_top10','P_top5','boot_median','p05','p95','n_variants_top10']
for kind in FM: print('\n',kind.upper()); print(FIN[kind][cols].head(14).round(2).to_string(index=False))
for kind in FM: print('\n',kind,'desc'); t=FIN[kind]; print(t.head(10)[['name']+[c for c in t.columns if c.startswith(('pressure_','usual_')) and c!='pressure_balls']+['pressure_balls']].round(2).to_string(index=False))
# cross-check of other report's picks
allnames={k:names[k] for k in FM}
tg={'bat':['Vaibhav Sooryavanshi','Abhishek Sharma','Phil Salt','Shubman Gill','Sai Sudharsan','Travis Head','Virat Kohli'],'bowl':['Mohammad Nawaz','Mahedi Hasan','Mujeeb Ur Rahman','Yuzvendra Chahal']}
for kind in FM:
    s150=V[kind]['eligibility 120 balls']; cn=ns['CNT'][kind].n; print('\n',kind.upper(),'other-report picks under our model:')
    for tname in tg[kind]:
        ids=allnames[kind][allnames[kind]==tname].index
        if len(ids)==0: print('  ',tname,'not in pool'); continue
        i=ids[0]; print('   %-22s pressure balls %4s  rank %s'%(tname,cn.get(i,0),(list(s150.index).index(i)+1) if i in s150.index else 'below 150-ball bar'))
pickle.dump(FIN,open('FIN2.pkl','wb'))
