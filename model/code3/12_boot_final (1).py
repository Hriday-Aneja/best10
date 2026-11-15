import warnings; warnings.filterwarnings('ignore')
src=open('11_clutch_validate.py').read().split('# ================= PART 2')[0]; exec(src)
import pickle
rng=np.random.default_rng(2027); B=300; res={}
up_b=pd.read_csv('/mnt/user-data/uploads/final_top_batters.csv').set_index('player_id'); up_w=pd.read_csv('/mnt/user-data/uploads/final_top_bowlers.csv').set_index('player_id')
for kind in FM:
    pool=POOL[kind]; R={w:pd.DataFrame(index=pool,columns=range(B),dtype=float) for w in ('clutch 20%','clutch 30%')}
    for b in range(B):
        c=pd.Series(np.bincount(rng.integers(0,len(ms),len(ms)),minlength=len(ms)).astype(float),index=ms)
        _,L,X=rate(kind,mult=c,ret=True); ato,clu,_=comp(kind,c)
        for w in R: R[w][b]=score(kind,L,X,ato,clu,w,pool).rank(ascending=False).reindex(pool)
    for w in R:
        base=OUT[kind]['R'][w]; t=pd.DataFrame({'player':names[kind].reindex(pool),'base_rank':base.rank(ascending=False).reindex(pool),
          'boot_median':R[w].median(axis=1),'rank_p05':R[w].quantile(.05,axis=1),'rank_p95':R[w].quantile(.95,axis=1),'P_top10':(R[w]<=10).mean(axis=1),'P_top5':(R[w]<=5).mean(axis=1)})
        t['consensus']=0.5*t.base_rank.rank()+0.5*t.boot_median.rank(); t=t.sort_values(['consensus','base_rank']); t['final_rank']=range(1,len(t)+1)
        o=OUT[kind]; t['L_plus_X_per100']=((o['L']+o['X']).reindex(t.index)*100); t['ato_wpa']=o['ato'].reindex(t.index); t['clutch_win_score']=o['clu'].reindex(t.index); t['hard_won_innings']=o['hn'].reindex(t.index)
        t['disadv_innings']=PM[kind].groupby('o').size().reindex(t.index)
        up=up_b if kind=='bat' else up_w
        for c_ in (['pressure_SR','usual_SR','pressure_dis%','usual_dis%'] if kind=='bat' else ['pressure_econ','usual_econ','pressure_wkts_per_over','usual_wkts_per_over']): t[c_]=up[c_].reindex(t.index)
        t['pressure_balls']=up['pressure_legal_balls'].reindex(t.index); t['old_final_rank']=up['final_rank'].reindex(t.index)
        res[(kind,w)]=t; t.reset_index().rename(columns={'index':'player_id','o':'player_id'}).to_csv('/home/claude/code/code3/final_%s_%s.csv'%(kind,w.replace(' ','_').replace('%','')),index=False)
        print('\n=====',kind.upper(),w); print(t.head(14)[['final_rank','player','old_final_rank','boot_median','rank_p05','rank_p95','P_top10','P_top5','hard_won_innings','disadv_innings','clutch_win_score']].round(3).to_string(index=False))
pickle.dump(res,open('final_clutch.pkl','wb')); print('DONE')
