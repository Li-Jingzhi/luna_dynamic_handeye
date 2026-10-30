"""Same fixed-view Pro-equation method on current-robot batches; no dynamic head chain."""
from pathlib import Path
import json
import sys
import numpy as np

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];sys.path.insert(0,str(ROOT))
from reports.latest_handeye_20261030.calibrate import read_data
from dynamic_handeye.registration import rigid_fit
from dynamic_handeye.empirical import fit_fields,correct
from dynamic_handeye.calibration import metrics


def fit(c,t,test):
    r,shift=rigid_fit(c,t);base=c@r.T+shift;x=test@r.T+shift
    field=fit_fields(base,t,.06,[3.])[('inverse',3.)]
    return correct(field,x),field


def main():
    a,d=read_data();results=[]
    for batch in sorted(set(d['batch'])):
        for head in sorted(set(d['heads'][d['batch']==batch])):
            mask=(d['batch']==batch)&(d['heads']==head)
            all_ids=np.flatnonzero(mask);all_ids=all_ids[np.argsort([d['rows'][i]['waypoint_id'] for i in all_ids])]
            c=d['camera'][all_ids];t=d['target'][all_ids];ids=np.arange(len(c));train_pred,field=fit(c,t,c)
            errors=[];scales=[]
            for test in np.array_split(ids,5):
                train=ids[(ids<test[0]-2)|(ids>test[-1]+2)]
                prediction,m=fit(c[train],t[train],c[test]);errors.extend(prediction-t[test]);scales.append(m['within_pro_scale_bounds'])
            results.append(dict(batch=batch,head=head,n=len(c),training=metrics(train_pred-t),blocked_cv=metrics(errors),
                within_pro_scale_bounds=field['within_pro_scale_bounds'] and all(scales)))
    (OUT/'fixed_view_comparison.json').write_text(json.dumps(dict(assets=a.hashes,sources=d['sources'],spec='joint inverse affine/Gaussian, length60mm/ridge3, fixed pF and original FK',
        note='Fixed-head per-batch fits, 5 trajectory blocks with 2-neighbor purge. Different coverage from Pro; development diagnostics.',groups=results),indent=2)+'\n')
    print([(g['batch'],g['head'],round(g['blocked_cv']['p95_mm'],3)) for g in results])


if __name__=='__main__':main()
