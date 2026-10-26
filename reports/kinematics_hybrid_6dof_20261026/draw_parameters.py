from pathlib import Path
import sys, json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root))
from dynamic_handeye.geometry import Assets, torso_from_arm
assets=Assets()
out=root/'reports/kinematics_hybrid_6dof_20261026'
out.mkdir(parents=True, exist_ok=True)
im=Image.new('RGB',(2400,2050),'#f4f7fb');d=ImageDraw.Draw(im)
font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
f=ImageFont.truetype(font,27);h=ImageFont.truetype(font,33);title=ImageFont.truetype(font,45);small=ImageFont.truetype(font,24)
ink='#172b45';known='#dff0e9';fit='#fff0d4';computed='#e4eafa'
def text(x,y,s,ft=f): d.text((x,y),s,font=ft,fill=ink)
def card(box,heading,lines,color=known):
 d.rounded_rectangle(box,18,fill=color,outline='#b3c2d2',width=2)
 text(box[0]+20,box[1]+12,heading,h)
 for i,line in enumerate(lines):text(box[0]+20,box[1]+63+i*39,line)
def arrow(a,b):
 d.line([a,b],fill='#56718f',width=4)
 v=np.array(b,float)-a;v/=np.linalg.norm(v);n=np.array([-v[1],v[0]])
 d.polygon([tuple(b),tuple(np.array(b)-v*17+n*8),tuple(np.array(b)-v*17-n*8)],fill='#56718f')
text(55,22,'旧版 Luna V2.0｜混合六参数 · 双目双臂运动学',title)
text(55,86,'当前头链：Rz(yaw) Ry(-pitch)，不使用 D=640 mm；绿色固定、橙色拟合；mm / °')
card((900,145,1500,270),'躯干 S_MODEL',['Leg_J6_Link 原点；不是地面世界系'],computed)
for dest in [(400,330),(1200,330),(2000,330)]:arrow((1200,270),dest)
for side,x,sign,rx,label in [('left',55,'+','−','左臂'),('right',1655,'−','+','右臂')]:
 t=torso_from_arm(assets,side)
 card((x,330,x+690,520),f'{label} MDH 父系 M_{"L" if side=="left" else "R"}',[
  f'T_S_M：p ≈ [0, {sign}56, 391.501] mm',f'R ≈ Rx({rx}90°)；名义 CAD / MDH 对齐','首关节安装位置为 y=±214.6 mm，勿混用'])
 arrow((x+345,520),(x+345,570))
 card((x,570,x+690,735),f'{label} 7 关节 + 固定法兰',[
  'T_S_F = T_S_M · FK_MDH(q7)', 'q7 来自对应臂实际反馈；包含 87 mm 法兰'])
 arrow((x+345,735),(x+345,785))
 lines=['p_F = [0, 0, 101] mm（右臂实测）','884条有效记录，实际参加本次拟合'] if side=='right' else ['左球心偏置必须单独实测并显式输入','联合标定接口已支持；暂无左臂实机拟合']
 card((x,785,x+690,955),f'{label} 球心参考点',lines)
card((800,330,1600,510),'H_ref：公共旋转中心零位参考系',[
 'F_fixed：p = [−13.496, 0, 498.98] mm', '零位旋转 R=I（名义假设，待独立核实）','只使用 MJCF 中心位置；不使用其头链'],known)
arrow((1200,510),(1200,550))
card((800,550,1600,710),'H0：左右转头（Yaw）',[
 '纯旋转 Rz(q_yaw)，平移为 0', '反馈 head[0]；控制器限位 [−90°, +90°]'])
arrow((1200,710),(1200,750))
card((800,750,1600,910),'H1：上下抬头（Pitch）',[
 '纯旋转 Ry(−q_pitch)，平移为 0','反馈 head[1]；限位 [−25°, +25°]；H2无自由度'])
arrow((1200,910),(1200,950))
card((800,950,1600,1110),'校正左眼 C_L → 固定双目组',[
 'X = T_H1_CL：拟合 6DoF', 'T_CR_CL：R=I，t=[−59.538355, 0, 0] mm'],fit)
card((55,1000,745,1110),'数据坐标约定',['数据保留 M_R 参考；拟合残差转换到躯干 S'],computed)
card((1655,1000,2345,1110),'躯干输出',['P_S = F_fixed · Rz(qy) · Ry(−qp) · X · P_CL'],computed)
card((55,1160,1150,1735),'双臂共用的 MDH 数值表（各用对应 q7）',[],known)
text(85,1230,'段         a(mm)        α(°)        d(mm)        θ偏置(°)')
for i,row in enumerate(assets.rows('R.A1',7)):
 label=f'J{i}' if i<7 else '法兰固定'
 text(85,1280+i*46,f'{label:<10} {row[0]:>6g}           {row[1]:>5g}         {row[2]:>7g}          {row[3]:>5g}')
text(85,1666,'变换约定：Rx(α) · Tx(a) · Rz(θ+q) · Tz(d)',small)
K=assets.K[0];base=np.linalg.norm(assets.stereo['right_from_left_T_m'])*1000
card((1200,1160,2345,1450),'双目固定标定参数（本轮不重新优化）',[
 '校正分辨率：1280 × 992；左 CSI0 / 右 CSI1',
 f'左右校正 K 相同：fx=fy={K[0,0]:.6f} px',
 f'cx={K[0,2]:.6f}，cy={K[1,2]:.6f} px；校正 D=0',
 f'基线长度 {base:.6f} mm；相机右眼位姿用 inv(T_CR_CL)',
 '内外参源：37对有效标定图；立体 RMS=0.1183 px'])
card((1200,1490,2345,1735),'参与本次手眼拟合的数据',[
 '原始969条 → 有效884条；311个有效球位组；仅右臂',
 '头姿 [yaw,pitch]：[-10,-25] / [0,-20] / [10,-25]',
 '输入：相机球心、头角、q7、法兰球心偏置；单会话',
 '仅拟合 X 共6参数；中心、轴向、双目、球偏置固定'],fit)
card((55,1780,2345,2005),'算法及验证边界',[
 '先将参考球心反变换到 H1：Kabsch/SVD 初始化 X → 6参数 Huber 优化；坐标分量阈值 3 mm。',
 '训练 P95=13.598 mm；轨迹留出 P95=8.086 mm。不是亚毫米验收，也不是左臂或连续运动精度。',
 '资料：luna_new.ini / torso_mounts.json / luna_rectified_stereo.json / ball_point_head_20261018_v1.jsonl',
 '固定头部安装读取 head_mount_v2.json，绑定模型哈希；不从旧12参数模型冻结 A。'],computed)
im.save(out/'luna_v2_stereo_dual_arm_parameters.png')
print(out/'luna_v2_stereo_dual_arm_parameters.png')
