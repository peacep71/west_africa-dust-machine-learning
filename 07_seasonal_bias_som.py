# -*- coding: utf-8 -*-
"""Group 12 model-season validation bias fields with a separate 3x3 SOM.

Based on the manuscript's described method and supplied seasonal-bias source.
The final published node labels/order require verification with original runs.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from minisom import MiniSom

OUT = Path("outputs/SOM_seasonal_bias")
OUT.mkdir(parents=True,exist_ok=True)

def main():
    v = pd.read_parquet("outputs/validation_predictions_2019_2020.parquet")
    v['time'] = pd.to_datetime(v['time'])
    v['season'] = v['time'].dt.month.map({12:'DJF',1:'DJF',2:'DJF',3:'MAM',4:'MAM',5:'MAM',6:'JJA',7:'JJA',8:'JJA',9:'SON',10:'SON',11:'SON'})
    tags = [('a','LR','DJF','pred_lr'),('b','LR','MAM','pred_lr'),('c','LR','JJA','pred_lr'),('d','LR','SON','pred_lr'),
            ('e','RF','DJF','pred_rf'),('f','RF','MAM','pred_rf'),('g','RF','JJA','pred_rf'),('h','RF','SON','pred_rf'),
            ('i','LightGBM','DJF','pred_lgbm'),('j','LightGBM','MAM','pred_lgbm'),('k','LightGBM','JJA','pred_lgbm'),('l','LightGBM','SON','pred_lgbm')]
    maps=[]; cams=[]; stats=[]
    for tag,model,season,pcol in tags:
        subset=v[v.season==season].copy()
        subset['error']=subset[pcol]-subset['dust_aod']
        x=subset.set_index(['time','latitude','longitude'])[['dust_aod','error']].to_xarray()
        means=x.mean('time',skipna=True)
        maps.append(means['error'].expand_dims(tag=[tag]))
        cams.append(means['dust_aod'].expand_dims(tag=[tag]))
        vals=subset['error'].to_numpy()
        stats.append({'tag':tag,'model':model,'season':season,
            'mean_bias':float(np.nanmean(vals)), 'RMSE':float(np.sqrt(np.nanmean(vals**2)))})
    b=xr.concat(maps,'tag'); c=xr.concat(cams,'tag')
    mask=np.isfinite(b.values).all(axis=0)
    X=b.values[:,mask].astype('float32')
    # featurewise standardisation of spatial maps; zero-variance cells handled.
    mean=X.mean(axis=0); sd=X.std(axis=0)
    Z=(X-mean)/np.where(sd==0,1,sd)
    som=MiniSom(3,3,Z.shape[1],sigma=0.8,learning_rate=0.5,random_seed=42)
    som.random_weights_init(Z)
    som.train_random(Z,10000)
    node=np.array([som.winner(z)[0]*3+som.winner(z)[1]+1 for z in Z])
    for record, label in zip(stats,node):record['node']=int(label)
    pd.DataFrame(stats).to_csv(OUT/'model_season_assignments.csv',index=False)
    bias_composites=[]; cams_composites=[]
    for n in range(1,10):
        idx=np.where(node==n)[0]
        bd=b.isel(tag=idx).mean('tag',skipna=True) if len(idx) else xr.full_like(b.isel(tag=0),np.nan)
        cd=c.isel(tag=idx).mean('tag',skipna=True) if len(idx) else xr.full_like(c.isel(tag=0),np.nan)
        bias_composites.append(bd.expand_dims(node=[n]))
        cams_composites.append(cd.expand_dims(node=[n]))
    xr.Dataset({'mean_bias':xr.concat(bias_composites,'node'),
                'CAMS_dust_AOD':xr.concat(cams_composites,'node')}).to_netcdf(OUT/'som_node_composites.nc')
    print(pd.DataFrame(stats))

if __name__ == '__main__':main()
