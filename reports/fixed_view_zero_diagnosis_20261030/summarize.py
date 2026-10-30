"""Export fixed-view hypothesis diagnostics, verification and a reference-explicit report."""
from pathlib import Path
import json
import sys
import numpy as np

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];sys.path.insert(0,str(ROOT))
from reports.fixed_view_zero_diagnosis_20261030.predict import predict_pixels
from reports.fixed_view_zero_diagnosis_20261030.analyze import target_fk
from dynamic_handeye.calibration import metrics
from dynamic_handeye.geometry import fixed_geometry_hashes


def summarize(a,d):
    r=json.loads((OUT/'results.json').read_text());model=json.loads((OUT/'fixed_head_best_candidate.json').read_text())
    model['fixed_geometry_hashes']=fixed_geometry_hashes(a)
    basic_path=OUT/'fixed_head_candidate.json'
    if basic_path.exists():
        basic=json.loads(basic_path.read_text());basic['fixed_geometry_hashes']=fixed_geometry_hashes(a)
        basic_path.write_text(json.dumps(basic,indent=2)+'\n')
    ids=np.flatnonzero(d['heads']=='0,-25');pred=[]
    for i in ids:pred.append(predict_pixels(model,a,d['rows'][i]['corrected_pixels'],d['rows'][i]['head_angles_deg'],diagnostic=True)['xyz_m'])
    pred=np.array(pred);target=target_fk(a,d['q7'][ids],np.array(model['effective_joint_offsets_deg']))
    model['training_corrected_fk']=metrics(pred-target);model['training_original_fk']=metrics(pred-d['target'][ids])
    r['primary_selected_training']=dict(corrected_fk=model['training_corrected_fk'],original_fk=model['training_original_fk'])
    (OUT/'fixed_head_best_candidate.json').write_text(json.dumps(model,indent=2,allow_nan=False)+'\n')
    (OUT/'results.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    reference=dict(schema='luna_diagnostic_fk_reference_v1',approved=False,
        reference_version='fixed_0m25_software_fk_20261030',original_assets=a.hashes,
        fixed_geometry_hashes=fixed_geometry_hashes(a),
        q_input='Original controller fb_pos in degrees; hardware encoder reference unchanged',
        formula='T_S_M @ FK_MDH(q_feedback + effective_joint_offsets_deg) @ pF',
        effective_joint_offsets_deg=model['effective_joint_offsets_deg'],pf_m=a.pf.tolist(),
        hardware_encoder_reference_changed=False,
        note='Separately versioned software hypothesis; do not write these angles as encoder reset values.')
    (OUT/'calibrated_fk_reference.json').write_text(json.dumps(reference,indent=2)+'\n')
    i=int(ids[0]);example=dict(pixels=d['rows'][i]['corrected_pixels'],head=d['rows'][i]['head_angles_deg'],
        result=predict_pixels(model,a,d['rows'][i]['corrected_pixels'],d['rows'][i]['head_angles_deg'],diagnostic=True))
    (OUT/'example.json').write_text(json.dumps(example,indent=2)+'\n')
    verification=dict(samples=len(ids),training_corrected_fk=model['training_corrected_fk'],training_original_fk=model['training_original_fk'],
        effective_offsets_deg=model['effective_joint_offsets_deg'],camera_scale=model['camera_scale'],
        unapproved_call_rejected=False,wrong_head_rejected=False)
    try:predict_pixels(model,a,example['pixels'],example['head'])
    except ValueError:verification['unapproved_call_rejected']=True
    try:predict_pixels(model,a,example['pixels'],[0,-20],diagnostic=True)
    except ValueError:verification['wrong_head_rejected']=True
    assert verification['unapproved_call_rejected'] and verification['wrong_head_rejected']
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    write_report(r,model)


def write_report(r,model):
    p=r['scopes']['fixed_0m25'];best=model['candidate'];block=p[best];transfer=p[best+'_transfer']
    text='''# 先固定头姿：大误差原因与软件几何校准方案

2026-10-30。本轮只使用本机当前相机参数、原MDH和12份已有A/B/C日志，不包含另一台旧Luna的数据；不连接机器人、不回零、不重置编码器、不改正式相机/MDH。第一阶段限定固定头姿[0,−25]，615个有效位置，三条轨迹共用同一套相机外参。第二诊断范围为六个已有固定头姿，各自独立外参，2247有效位置，不代表动态头链标定。

## 关键结论

现有数据强烈支持存在稳定的**关节/工具几何参考不一致**。有效关节修正对独立留出的整条轨迹也有效，明显强于仅更换视觉外参或仿射/高斯补偿。但是不能据此唯一认定编码器零位错误：末轴偏置与工具点定义存在精确等价，其他几何误差、挠曲和相机系统偏差也可能进入有效参数。

控制器命令全零并读取全零，仅说明控制器在自己的参考中到零，不能证明机械轴对齐MDH零位。若在当前反馈已为0时再重置为0，不会自动消除模型中的约1°几何偏差。真正的编码器校零需已知的机械零位基准和厂家的角度/编码器定义；本报告不提供可直接写入编码器的数值。

**现有数据下可行的软件方案**：保持原SDK反馈和编码器参考，在明确版本的离线FK中使用q+δ，重新标定固定相机外参，再做小幅尺度/残差修正。相机位置预测本身只读取像素和固定头角门控，不读取当前q7；q7仅用于计算对照FK。当前原FK结果全部保留，不能将修正FK闭环成绩当作原控制器FK精度提升。

## 固定[0,−25]的同口径对照

每个训练折重新拟合参数；5折沿每条轨迹的物理球位连续分段并剔除两侧2个邻接位置。跨轨迹实验依次用两条轨迹训练、第三条整条测试，**相机外参也完全冻结**，不在目标轨迹重新估计外参。615个位置均有跨轨迹测试预测。三批现已研究过，因此仍是开发诊断，不是独立实物验收。

|方案|误差参考|训练P95 mm|分块留出P95 mm|跨整条轨迹P95 mm|
|---|---|---:|---:|---:|
'''
    for kind,label in [('nominal','原FK＋刚体外参'),('scale','原FK＋全局相机尺度'),('zeros','有效关节修正＋刚体外参'),('zeros_scale','有效关节修正＋全局尺度')]:
        v=p['fits'][kind];ref='原FK' if kind in ('nominal','scale') else '修正FK'
        text+=f"|{label}|{ref}|{v['training_corrected_fk']['p95_mm']:.3f}|{p['blocked'][kind]['corrected_fk']['p95_mm']:.3f}|{p['cross_trajectory'][kind]['corrected_fk']['p95_mm']:.3f}|\n"
    text+=f"|有效关节修正＋Pro式仿射/高斯|修正FK|—|{p['zeros_plus_pro_gaussian']['corrected_fk']['p95_mm']:.3f}|{p['zeros_plus_pro_gaussian_transfer']['corrected_fk']['p95_mm']:.3f}|\n"
    text+=f"|有效关节修正＋尺度＋Pro式仿射/高斯|修正FK|{model['training_corrected_fk']['p95_mm']:.3f}|{block['corrected_fk']['p95_mm']:.3f}|{transfer['corrected_fk']['p95_mm']:.3f}|\n"
    text+=f'''
原FK不变的完整仿射及Pro式60mm/ridge3联合高斯，在同一615点分块协议下P95分别为{p['original_fk_visual_compensation']['affine']['original_fk']['p95_mm']:.3f}/{p['original_fk_visual_compensation']['pro_gaussian']['original_fk']['p95_mm']:.3f}mm。固定头姿仍然有大误差，动态头链和普通噪声不是唯一原因。

选中方案按四个几何诊断变体的最差开发折P95选择：{best}。训练P95={model['training_corrected_fk']['p95_mm']:.3f}mm；分块留出P95={block['corrected_fk']['p95_mm']:.3f}mm、最大{block['corrected_fk']['max_mm']:.3f}mm；跨整条轨迹P95={transfer['corrected_fk']['p95_mm']:.3f}mm、最大{transfer['corrected_fk']['max_mm']:.3f}mm。最差单个开发折P95为{model['validation']['worst_dev_fold_p95_mm']:.3f}mm。

这支持固定头姿下“P95约1mm”的阶段方案，尚未满足全部位置<1mm，也不能证明物理绝对误差相同。跨轨迹的三次P95：

|整条测试轨迹|P95 mm|最大 mm|
|---|---:|---:|
'''
    for f in transfer['folds']:text+=f"|{f['label']}|{f['corrected_fk']['p95_mm']:.3f}|{f['corrected_fk']['max_mm']:.3f}|\n"
    text+='''
## 有效关节修正具体是什么

原计算θ_i=Θ_INI,i+q_feedback,i；诊断模型为θ_i=Θ_INI,i+q_feedback,i+δ_i。仅优化J2–J7，每轴通过2*tanh限制在±2°，0.5°尺度正则；J1固定0以去除与相机/基座旋转的规范自由度，**不能解释为J1物理误差已确认是0**。相机尺度限制±2%，用0.5%尺度正则约束。

选中全量固定头姿模型参数（度）：

|关节|J1|J2|J3|J4|J5|J6|J7|
|---|---:|---:|---:|---:|---:|---:|---:|
'''
    text+='|δ|'+'|'.join(f'{x:+.6f}' for x in model['effective_joint_offsets_deg'])+'|\n'
    text+=f'''
全局相机三维尺度s={model['camera_scale']:.9f}，约+{(model['camera_scale']-1)*100:.3f}%。这是有效尺度诊断量，未改写内参/基线，也不代表实际基线已测量出同样偏差。映射p0=R(s*p_C)+t，再接联合逆向仿射/高斯60mm/ridge3。

上述δ是与该原始反馈坐标和工具定义绑定的软件修正，**不是控制器编码器重置值**，也不是命令角需直接加上的运动指令。控制器仍按原角度控制；在独立计算FK参考时加δ。若硬件重置改变反馈参考，本模型和既有日志的角度基准需重新建立，不能继续混用。

主模型的零位加尺度数据Jacobian在本假设下为7/7秩；六零位模型为6/6秩。这说明在固定pF和刚体相机假设下可局部辨识，不能证明物理成因唯一。

## 为什么更像关节相关几何问题

1. 三条构型/空间不同轨迹中，J3约+0.63至+0.69°、J4约−1.39至−1.46°方向稳定；整轨迹留出没有重新调目标轨迹相机外参，仍约1mm。比仅在一条轨迹拟合低误差更有解释力。
2. 只允许相机统一缩放，原FK误差仍约7–8mm；完整仿射/高斯也未达到1mm。关节修正将跨轨迹从约8.86降到约1.41mm，再加尺度/残差场约1.06mm。不能把这解读为相机完全无误差。
3. 每个头姿独立相机外参，因此没有使用动态头链解释本结果。历史采集审计的窗口抖动约0.1–0.3mm量级，显著小于原闭环残差；时间/检测仍可能影响尾部，但不能仅用普通抖动解释主差距。
4. 逐个关节强制δ=0、重新拟合其余轴的训练P95如下，只是消融诊断：

|强制不修正的轴|训练P95 mm|
|---|---:|
'''
    for k,v in r['primary_zero_ablations'].items():text+=f"|{k}|{v['training_corrected_fk']['p95_mm']:.3f}|\n"
    ambiguity=r['J7_tool_point_exact_ambiguity']
    text+=f'''
J3/J4/J6/J7更有影响，J5改善很弱。所有消融都保留pF固定，因此并不能排除工具定义与末轴偏置耦合。

## 不能据此证明“应该重置编码器”的精确反例

本机固定法兰F下，末轴δ7可与工具点变化pF'=F⁻¹ Rz(δ7) F pF精确等价。当前J7={ambiguity['J7_deg']:.6f}°，pF=[0,0,101]mm，与J7不修正但pF≈[{ambiguity['equivalent_pf_m'][0]*1000:.6f},{ambiguity['equivalent_pf_m'][1]*1000:.6f},{ambiguity['equivalent_pf_m'][2]*1000:.6f}]mm产生相同球心轨迹。615个真实q的数值最大差{ambiguity['equivalence_max_mm']:.3g}mm。

若工具点原点/方向没有独立确认，仅凭这些图像与q不能区分上述两种物理解释。即使pF向量已精密测量，其余连杆/轴线、柔性与相机共同偏差仍可能被有效角吸收。因此现有数据可以支持软件几何校准方案，不能支持直接改编码器的唯一因果判断。

## 原FK和修正FK必须分开

同一个选中相机模型在全量615点上，对修正FK训练P95为{model['training_corrected_fk']['p95_mm']:.3f}mm，对原FK却是{model['training_original_fk']['p95_mm']:.3f}mm。这是参考坐标/模型改变的直接结果，不是原FK视觉精度从8mm变成0.6mm。

若必须继续把厂商原FK输出当唯一标签，现有数据尚没有约1mm泛化证据；不能用修正FK成绩替代。如果接受建立明确版本的标定FK，现有数据已支持先把固定头姿做到P95约1mm。实际物理绝对准确性、工具方向和编码器基准仍需独立验证。

## 第一阶段和第二阶段的顺序

第一阶段现在可离线执行：冻结本报告模型和原反馈参考，以FK_calibrated(q)=T_S_M*FK_MDH(q+δ)*pF作为新参考；固定[0,−25]调用相机模型。保存原FK与标定FK两列误差，比较新会话时不重新拟合δ或外参。当前数据已研究过，现场验收仍需未参与拟合的新轨迹/会话；约1mmP95与最大<1mm分别判断。

第二阶段等固定视角验收后再做：冻结经过验证的FK几何版本，补齐yaw/pitch交叉覆盖，再拟合一个连续头链并做未见头姿测试。六个固定视角各自独立外参的低残差不能当作任意头姿连续映射已通过。本次没有提前重训动态链或续采硬件。

## 文件和运行

- `calibrated_fk_reference.json`：版本化的软件FK修正，原反馈角单位为度，不是硬件参数写入文件。
- `fixed_head_best_candidate.json`：选定固定[0,−25]相机模型、软件FK修正、尺度与场；approved=false。
- `fixed_head_candidate.json`：不加相机尺度/场的基础零位诊断模型。
- `results.json`：全部假设、各折参数、原FK/修正FK指标、数据Jacobian和工具点等价反例。
- `verification.json`、`example.json`：615个真实点的落盘像素接口复核及调用示例。

模型不连接SDK，视觉推理不输入q7。仅显式诊断模式可用，要求反馈头角在[0,−25]附近0.02°以内，资产改变会拒绝。普通点需双目真实同名点，黄球需已经做完球心投影修正；没有验证任意目标检测系统偏差的迁移。

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python reports/fixed_view_zero_diagnosis_20261030/analyze.py
.venv/bin/python reports/fixed_view_zero_diagnosis_20261030/predict.py \\
  --pixels 796.984164052486 605.2666373303944 746.7779201285618 605.3636777305901 \\
  --head 0 -25.001 --diagnostic
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -v
```

合成验证包括已知零位恢复、已知尺度恢复、物理球位分块隔离，以及固定头姿/参数改变/未批准推理拒绝。全部报告是利用现有数据的开发诊断，硬件编码器未重置，正式MDH和相机参数未覆盖。
'''
    # Six fixed views share zero/scale but each has a separate camera pose, not a dynamic head chain.
    v=r['scopes']['all_fixed_views']
    text+='\n## 六个独立固定视角的旁证\n\n共享六轴有效偏置、每个头姿独立刚体外参，2247点：\n\n|假设|训练P95|分块留出P95|跨轨迹P95|\n|---|---:|---:|---:|\n'
    for kind in ('nominal','zeros','zeros_scale'):
        text+=f"|{kind}|{v['fits'][kind]['training_corrected_fk']['p95_mm']:.3f}|{v['blocked'][kind]['corrected_fk']['p95_mm']:.3f}|{v['cross_trajectory'][kind]['corrected_fk']['p95_mm']:.3f}|\n"
    text+='\n跨轨迹C的负yaw头姿在A/B没有对应固定视角，297条无法预测，明确不计入该跨轨迹汇总；其余1950条有对应固定视角。主结论使用[0,−25]615条全覆盖结果，不受此遗漏影响。\n'
    (OUT/'README.md').write_text(text)


if __name__=='__main__':
    from reports.latest_handeye_20261030.calibrate import read_data
    assets,data=read_data();summarize(assets,data)
