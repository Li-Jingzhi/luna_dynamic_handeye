"""Export residuals, source verification, Pro-style comparison model and human report."""
from pathlib import Path
import csv
import json
import sys
import numpy as np

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];sys.path.insert(0,str(ROOT))
from dynamic_handeye.empirical import base_points,correct,predict_pixels,fit_fields
from dynamic_handeye.geometry import pose,apply
from dynamic_handeye.calibration import metrics


def summarize(a,d):
    r=json.loads((OUT/'validation.json').read_text());model=json.loads((OUT/'model.json').read_text())
    base=base_points(model['baseline'],a,d['camera'],d['head'])
    pred=correct(model['correction'],base);delta=pred-d['target'];distance=np.linalg.norm(delta,axis=1)*1000
    r['full_fit_by_batch']={b:metrics(delta[d['batch']==b]) for b in sorted(set(d['batch']))}
    r['full_fit_by_head']={h:metrics(delta[d['heads']==h]) for h in sorted(set(d['heads']))}
    r['full_fit_below_1mm_fraction']=float(np.mean(distance<1))
    for name in r['summary']:
        r['summary'][name]['by_protocol']={kind:dict(
            min_p95_mm=min(f['scores'][name]['p95_mm'] for f in r['folds'].values() if f['kind']==kind),
            max_p95_mm=max(f['scores'][name]['p95_mm'] for f in r['folds'].values() if f['kind']==kind),
            max_error_mm=max(f['scores'][name]['max_mm'] for f in r['folds'].values() if f['kind']==kind))
            for kind in ('trajectory','spatial','configuration','head')}
    head=a.head(d['head']);_,inv,count=np.unique(d['group'],return_inverse=True,return_counts=True)
    def residual(x):
        z=apply(head@pose(x),d['camera']);means=np.array([np.bincount(inv,weights=z[:,i])/count for i in range(3)]).T
        return ((z-means[inv])*1000).ravel()
    x=np.array(model['baseline']['x'][6:]);eps=1e-5
    jac=np.column_stack([(residual(x+np.eye(6)[i]*eps)-residual(x-np.eye(6)[i]*eps))/(2*eps) for i in range(6)])
    singular=np.linalg.svd(jac,compute_uv=False)
    r['relative_mount_observability']=dict(rank=int(sum(singular>singular[0]*1e-7)),singular_values=singular.tolist(),
        note='Local data-only numerical rank; not proof of physical correctness or full head range.')
    (OUT/'validation.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    with (OUT/'training_predictions.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['batch','head','waypoint','source_sha256','pred_x_m','pred_y_m','pred_z_m','fk_x_m','fk_y_m','fk_z_m','error_mm'])
        for i,row in enumerate(d['rows']):w.writerow([d['batch'][i],row['head_key'],row['waypoint_id'],row['source_sha256'],*pred[i],*d['target'][i],distance[i]])
    with (OUT/'candidate_ranking.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['candidate','valid_pro_scale','worst_fold_p95_mm','mean_fold_p95_mm','full_fit_p95_mm'])
        for n,v in sorted(r['summary'].items(),key=lambda z:z[1]['worst_p95_mm']):w.writerow([n,v['valid_pro_scale'],v['worst_p95_mm'],v['mean_p95_mm'],r['training_candidates'][n]['p95_mm']])
    example=dict(model='reports/latest_handeye_20261030/model.json',pixels=d['rows'][0]['corrected_pixels'],head=d['rows'][0]['head_angles_deg'],diagnostic=True)
    example['result']=predict_pixels(model,a,example['pixels'],example['head'],diagnostic=True)
    (OUT/'example.json').write_text(json.dumps(example,indent=2)+'\n')
    # Export a direct Pro-equation counterpart with the same reference head mapping.
    # It is a fixed-pF single-arm restriction, not Pro's free-pF two-arm optimization.
    if model['baseline']['kind']=='camera_relative_head':
        pro=dict(model);pro['candidate']='relative:inverse_l60_r3'
        pro['correction']=fit_fields(base,d['target'],.06,[3.])[('inverse',3.)]
        pro['validation']=dict(worst_p95_mm=r['summary'][pro['candidate']]['worst_p95_mm'],acceptance='not passed')
        (OUT/'pro_style_model.json').write_text(json.dumps(pro,indent=2,allow_nan=False)+'\n')
    write_readme(r,model,example)


def write_readme(r,model,example):
    selected=r['selected'];best=r['summary'][selected];baseline=r['summary']['fk:geometry_l0_r0'];pro=r['summary']['relative:inverse_l60_r3']
    text=f'''# 本机最新手眼标定：固定原FK、仿射与高斯补偿

2026-10-30，基于仓库99419e7及本机最新相机参数。完成40候选、19个外层划分及训练内8候选选择流程，生成完整离线标定模型。**当前尚不能达到Pro约1.5mm的分块留出水平，也没有达到任意头姿最大误差小于1mm。**

## 输入、参考与当前数据

只使用本机新参数和新数据：轨迹A1254、轨迹B407、新九姿前四轮586，共{r['n']}条有效观测、{r['physical_waypoints']}个分组物理球位、12个源日志。旧机2026-10-18数据未加入。原始拒绝保留在各源日志，未按拟合误差筛点。

6个已有头姿为[−10,−25]、[−10,−20]、[0,−25]、[0,−20]、[10,−25]、[10,−20]；尚无pitch=−15完整层，不能声明全头部行程。C前四轮已被诊断研究使用，本次也加入开发标定，不能再叫独立验收。

FK采用当前冻结MDH和躯干安装，pF固定[0,0,101]mm，七轴额外零位修正全0。目标是FK预测球心位置。pF虽固定作为本次参考，物理尺寸、方向与法兰原点仍须独立核对；现有配置note与后续报告对测量状态的文字不完全一致，本次不利用自由pF吸收误差。

相机模型、FK、安装哈希随模型保存。逐日志MDH/相机参数/校正像素契约检查通过；从球心修正后的四像素重新三角化。相机为固定5ms/增益1/白平衡关闭的现采集配置；本次没有操作设备。

## 算法

1. 有效12参数头链或相机相对头链：同物理球位跨头姿一致性拟合相机安装X；相对法不使用FK标签优化X，名义头根仅作坐标规范。
2. 比较几何、完整仿射、联合前向仿射+高斯，以及Pro方程形式的逆向仿射+高斯。核长度30/60/100mm，ridge=0.1/1/3，共40候选。
3. 前向模型为p=p0+F(p0)B+K(p0,P)W；逆向模型联合解D(target)M=p0+KW，再逆仿射输出目标位置。后者固定本机pF，只使用右臂，与Pro可拟合共享pF的双臂完整问题不同。
4. 逆向仿射候选的奇异值需在0.9–1.1范围才可参与最终排名。前向场是经验补偿，不把其参数解释为物理安装或新相机内参。

核中心只来自每个训练折。运行只输入双目同名点四像素和实际yaw/pitch；q7只形成训练标签及构型划分，运行不读取q7、pF、轨迹ID或点号。

## 验证与选型

外层19项：整轨迹A/B/C分别留出3项、空间块5项、构型块5项、完整头姿6项。非头姿划分将同球位所有头姿整体分组，避免球位泄漏；整头姿留出允许训练中的其他头姿使用同球位，故并非同时未见位置。C负yaw头姿仅来自C，头姿/批次仍有混杂。

每个外层训练域再按同类分块，仅在训练数据中比较8个预设几何/仿射/前向60mm-ridge1/逆向60mm-ridge3候选（两种头链），按内层最差P95、再平均P95选择，之后预测外层。其结果与“40候选在开发外层上的固定候选排名”分别保存。候选设计参考历史研究，所以这些仍是开发验证，不是独立实物验收。

全量最终模型在40候选中按开发19折最差P95最小、再平均P95选择：**{selected}**，即相机相对头链＋联合仿射/高斯，长度60mm、ridge3。用全部2247条观测重新拟合后的模型包含2247个核中心及6741个高斯权重，不能称为只有6或12个参数。

## 结果（P95，mm）

|测试协议|有效12参数纯几何|选中的联合仿射/高斯|Pro式逆向60mm/ridge3|
|---|---:|---:|---:|
'''
    for kind,title in [('trajectory','整轨迹留出'),('spatial','空间块留出'),('configuration','构型块留出'),('head','整头姿留出')]:
        text+=f'|{title}|'
        for summary in (baseline,best,pro):
            v=summary['by_protocol'][kind];text+=f"{v['min_p95_mm']:.3f}–{v['max_p95_mm']:.3f}|"
        text+='\n'
    improvement=(baseline['worst_p95_mm']-best['worst_p95_mm'])/baseline['worst_p95_mm']*100
    text+=f'''
最差开发折P95由{baseline['worst_p95_mm']:.3f}降到{best['worst_p95_mm']:.3f}mm（改善{improvement:.1f}%）；并非所有子项都改善。新C与负yaw覆盖扩大，不能直接与此前A/B16折7.xmm的最差值比较。

全部开发点重拟合：平均{r['training']['mean_mm']:.3f}、RMS{r['training']['rms_mm']:.3f}、P95 **{r['training']['p95_mm']:.3f}**、最大 **{r['training']['max_mm']:.3f}mm**；只有{r['full_fit_below_1mm_fraction']*100:.2f}%训练点小于1mm。更小核候选的训练P95可更低，但开发留出变差，未据训练误差选它。

|整轨迹测试|选定候选P95 mm|8候选训练内选型流程P95 mm|
|---|---:|---:|
'''
    for label,f in r['folds'].items():
        if f['kind']=='trajectory':text+=f"|{label}|{f['scores'][selected]['p95_mm']:.3f}|{f['nested_test']['p95_mm']:.3f}|\n"
    text+='''
不能把训练内选型流程各折选出的不同模型拼成一个运行模型。最终model.json是全开发数据拟合的一个固定候选，其最终独立精度仍须新数据验收。分块误差由各折独立拟合模型得到，不是假称全量模型经过同批独立验证。

## 当前差距与下一步

Pro一点几mm是其固定视角内部留出结果；本机B固定视角也能拟合很低，但跨空间、构型和整轨迹误差明显增大。算法同形不能保证不同机械臂、相机和夹具得到同样残差。仅增加残差场复杂度没有解决本批的稳定系统不一致。

优先做：同一q的厂家SDK FK/本项目FK对照，核对法兰及球偏移定义；精密固定球心、多肘腕构型检查原FK是否漂移；精密已知距离靶分离双目度量误差。若物理约束确认FK偏差，再建立明确的新FK版本。固定原FK时不能把改标签后的低误差报告成原目标达标。

完成剩余头姿需现场恢复条件，历史暂停记录不是当前机械臂状态。本次没有自动续采。当前A/B/C均为开发数据，模型冻结后需全新会话、轨迹、空间球位、未见头姿及左右接近方向验收；左臂没有本次精度证据。最终检查最大误差、超限率和有效率，不只看P95。

## 模型与运行

- `model.json`：所比较候选中按上述规则选定的全量离线模型，approved=false。
- `pro_style_model.json`：相同相对头链、Pro式逆向仿射/高斯60mm-ridge3对照模型。
- `validation.json`：源哈希、全部40候选分数、19个外层划分及8候选内层记录。
- `candidate_ranking.csv`、`training_predictions.csv`：排名和逐训练点残差。
- `verification.json`、`example.json`：落盘四像素推理往返验证与可运行示例。

正式调用会拒绝，因为尚未验收；离线需显式diagnostic，返回xyz_m/S_MODEL及角度凸包、50mm空间支持标识。即使支持标识为true，也不能推导1mm保证。标定球使用已完成球心投影修正的像素；普通目标需要同一物理点的对应像素，不能套用球半径修正，迁移效果另验。

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python reports/latest_handeye_20261030/calibrate.py
.venv/bin/python reports/latest_handeye_20261030/predict.py \\
  --pixels 796.984164052486 605.2666373303944 746.7779201285618 605.3636777305901 \\
  --head 0 -25.001 --diagnostic
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -v
```

Python入口为dynamic_handeye.empirical.predict_pixels(model,Assets(),pixels,head,diagnostic=True)。本次新增4项测试覆盖前向/逆向联合映射的未见点恢复、序列化四像素接口、参数变更及未批准模型拒绝；主测试58项，52通过、6历史资产测试跳过。25个真实数据点的模型落盘接口与全量预测数值往返检查通过。VPI/OpenCV5的独立棋盘格标定兼容问题仍按此前报告单独存在，不影响本次已校正像素三角化及标定。
'''
    fixed_path=OUT/'fixed_view_comparison.json'
    if fixed_path.exists():
        fixed=json.loads(fixed_path.read_text())
        if fixed.get('assets')==r['assets'] and fixed.get('sources')==r['sources']:
            text+='\n## 固定视角Pro同口径补充对照\n\n各批次各头姿独立拟合，不使用动态头链；固定pF、零位全0，联合逆向仿射/高斯60mm/ridge3，五折轨迹分段并剔除两侧2个邻接位置。\n\n|本机批次|固定视角分块留出P95范围 mm|\n|---|---:|\n'
            for batch in ('A','B','C'):
                values=[g['blocked_cv']['p95_mm'] for g in fixed['groups'] if g['batch']==batch]
                text+=f'|{batch}|{min(values):.3f}–{max(values):.3f}|\n'
            text+='\n即使固定视角，三批也没有全部达到1.5mm；B约2.7mm，A约9–10mm，C约5.6–6.2mm。这比直接把Pro固定视角1.5mm与本机动态头姿最差折比较更公平。不同点云和构型覆盖仍不是同工况硬件对照。完整分头姿结果见fixed_view_comparison.json。\n'
    (OUT/'README.md').write_text(text)


if __name__=='__main__':
    from reports.latest_handeye_20261030.calibrate import read_data
    assets,data=read_data();summarize(assets,data)
