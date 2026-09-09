
%% 8极12槽 SPM 电机四分之一结构图（含中间开口+有厚度齿唇的槽口）
% 保存文件名：pmsm_8p12s_quarter_gap.m
% 说明：参数化绘图，仅作示意，单位 mm / 度（deg）

clear; clc; close all;
fprintf('Script pwd = %s\n', pwd);
addpath('C:\femm42\mfiles');   % 内含 femm.m 和一堆 mi_*/mo_* 封装函数
savepath;                                     % 可选：下次自动生效
openfemm;
newdocument(0);  % 磁场
mi_probdef(0,'millimeters','planar',1e-8,1,36);

%% ---------- 基本参数 ----------
Q       = 12;                 % 槽数
P       = 8;                  % 极数（机械极数）
thetaQ  = 360/Q;              % 槽距（机械角）
thetaP  = 360/P;              % 极距（机械角）

R_bore     = 31.5;              % 定子内圆（气隙外圆）
airgap     = 0.7;             % 气隙
R_r_out    = R_bore - airgap; % 转子外圆（磁体外圆）
R_gap1 = R_bore-airgap/3;
R_gap2= R_r_out+airgap/3;
R_r_core   = R_bore-airgap;              % 转子铁心外半径
R_shaft    = R_r_core-15;

t_mag      = 10;               % 永磁体厚度（径向）
beta_m     = 1;            % 磁体弧比（相对极距 0~1）

lipH=2.5;
slot_depth   = 15+lipH;
t_yoke     = 5;              % 定子轭厚
R_sy_out   = R_bore+t_yoke+slot_depth;             % 定子外半径

quarter_span = [0, 90];       % 画 0~90°的四分之一
deg = @(x) x*pi/180;

% %% ---------- 槽口"中间开口 + 齿唇厚度"参数 ----------
% % 中间开口角（度）：可用弧长换算，或直接给角度
% theta_gap    = 1 * (slot_opening / R_bore) * 180/pi;   % 开口缝角宽（越小越像半闭口）
% theta_bottom = 1.20 * (slot_opening / R_bore) * 180/pi;   % 槽底角宽（常比缝大）
% 
% lip_thick    =1;            % 齿唇"可见厚度"（mm）
% lip_span     = 3+3;   % 每侧齿唇沿圆周角宽（度）%现在不争为+3，为了防止齿唇和槽中间有小缝隙
% 
% % 如果想完全开口：theta_gap ~= theta_bottom，lip_thick=0
% % 如果想更半闭口：减小 theta_gap，或增大 lip_span/lip_thick

% ---------- 槽几何（局部参数，均为 mm 或 度） ----------
mouthW   = 8.7;         % 槽口宽度（矩形口的切向宽，mm）
mouthH   = 2;         % 槽口矩形的径向高度（mm）
bottomW  = 15.5;        % 槽底宽度（主槽体底边切向宽，mm）
bottomH  = 12.645+lipH;
edgeFillet = 0;         % 先不做圆角（0=无）

depth2 = 0.5; %槽六边形向外突出的程度,代替槽口高度
slot_depth2   = slot_depth+depth2; % 槽深（示意）
lipW=1.8; %槽口宽度

% ---------- 线圈/绝缘（局部参数） ----------
ins     = 1;          % 槽内绝缘/气隙到线圈的"退让"（四周统一退 ins，mm）
splitGap= 1;          % 中央分裂缝（两线圈之间的净间隙，mm）
coilShrinkTop   = 0;  % 线圈相对主槽在"上口"额外再缩一点（形似上窄）
coilShrinkBottom= 0;  % 线圈相对主槽在"底边"额外再缩一点（形似下窄）

% 假设转子外圆 R_r_out、转子铁心外半径 R_r_core 已有
% 典型参数（按需要改）
phi_m      = 40;     % 每块磁体切向跨度(度) —— 越大越长
t_mag      = 2;    % 磁体厚度(mm) = Rout - Rin
t_bridge   = 2.5;    % 表面桥厚(mm)：磁体到转子外圆之间的铁桥
% 设定磁体的内外半径（放在外圆之下、留桥）
Rout = R_r_out - t_bridge;     % 磁体外半径（靠近气隙侧）
Rin  = Rout - t_mag;           % 磁体内半径（靠近铁心侧）

bodyC = [0.90 0.55 0.55];      % 磁体主体颜色（淡红）
capC  = [1 1 1];               % 端盖颜色（白色，符合你的要求）
edgeC = [0 0 0];               % 黑色描边                  % 参考半径（用 R_in 很方便）

maxsegdeg = 1; %其实没用了，但是保留参数

R_bottom   = R_bore + slot_depth;
theta_open = 1;
tooth_tip  = 0.5;                   % 比如齿顶往里缩 0.5 mm
group_slot = 1;                     % 定子槽组号

phases = 3;   % 相数
slots  = 12;  % 槽数
Nc     = 100; % 导体数（假定：每相的导体数）
Is_amp = 3.5; % 定子电流幅值（相电流峰值）
kf     = 0.6; % 槽填充系数
Tc     = 30;  % 线圈温度

%% ---------- 预计算角度（槽中心、磁体中心） ----------
slot_centers = (0:Q-1)*thetaQ;
slot_centers_quarter = slot_centers(slot_centers >= quarter_span(1) & slot_centers <= quarter_span(2));

pole_pitch = thetaP;
mag_span   = beta_m * pole_pitch;
mag_centers = (thetaP/2) : thetaP : 360;  % 22.5°, 67.5° ...
mag_centers_quarter = mag_centers(mag_centers>quarter_span(1) & mag_centers<quarter_span(2));

%% ---------- 绘图 ----------
% figure('Color','w'); hold on; axis equal off;

% 背景扇形（淡灰，非必须）
% draw_ring_sector(R_shaft, R_sy_out, quarter_span(1), quarter_span(2), 1);

% 1) 定子轭
draw_ring_sector(R_sy_out, R_sy_out+20, quarter_span(1), quarter_span(2), 2);
[xs, ys] = pol2cart(45*pi/180, R_sy_out+10);

mi_getmaterial('Air');    % 或你自己定义的铁心材料

mi_addblocklabel(xs, ys);
mi_selectlabel(xs, ys);
mi_setblockprop('Air', 1, 0, '', 1, 0, 0);
mi_clearselected;
% draw_ring_sector(R_bore, R_sy_out, quarter_span(1), quarter_span(2), 2);

[xs, ys] = pol2cart(45*pi/180, (R_bore+R_sy_out)/2);

mi_getmaterial('Pure Iron');    % 或你自己定义的铁心材料

mi_addblocklabel(xs, ys);
mi_selectlabel(xs, ys);
mi_setblockprop('Pure Iron', 1, 0, '', 1, 0, 0);
mi_clearselected;

% 2) 齿区铺底
% draw_ring_sector(R_bore, R_sy_in, quarter_span(1), quarter_span(2), 1);

% 3) （把内圆弧先画在底层，避免"封口视觉"）
% draw_arc(R_bore, quarter_span(1), quarter_span(2), maxsegdeg, 1);

% 4) 在齿区"挖槽 + 补两侧齿唇"
group_bore = 11;
th_q0 = quarter_span(1);
th_q1 = quarter_span(2);

prev_p2_ang = [];    % 记录上一槽 p2 的角度
first = true;

mi_addcircprop('A', 0, 1);    % A 相电路，初始电流设为 0
mi_addcircprop('B', 0, 1);    % B 相
mi_addcircprop('C', 0, 1);    % C 相

for k = 1:numel(slot_centers_quarter)
    th_center = slot_centers_quarter(k);

    slot = build_slot_polys(mouthW, mouthH, bottomW, bottomH, slot_depth, depth2, lipW, 0);

    % 1) 画槽体（齿根开口）
    patch_uv(slot.bodyU, slot.bodyV, R_bore, th_center, [1 1 1], 'none', quarter_span, false);

    % 2) 计算这一槽在齿根圆上的交点角度
    [u1,v1] = deal(slot.p1(1), slot.p1(2));   % 左交点（局部）
    [u2,v2] = deal(slot.p2(1), slot.p2(2));   % 右交点（局部）

    [x1,y1,th1] = slot_uv2xy_angle(u1, v1, R_bore, th_center);
    [x2,y2,th2] = slot_uv2xy_angle(u2, v2, R_bore, th_center);

    % 3) 先画上一段齿根弧：上一槽 p2 -> 本槽 p1
    if first
        % 对第一槽：从 quarter_span(1) -> 第一槽的 p1
        draw_arc_Rbore(R_bore, th_q0, th1, 5, 2);
        first = false;
    else
        % 从上一槽的 p2 -> 本槽的 p1
        draw_arc_Rbore(R_bore, prev_p2_ang, th1, 5, 2);
    end

    % 更新"上一槽"的 p2 角度
    prev_p2_ang = th2;

r1 = hypot(x1,y1);
r2 = hypot(x2,y2);
disp([r1 r2 R_bore]);

% [coilL, coilR] = build_coils_in_slot(mouthW, mouthH, bottomW, bottomH, slot_depth, depth2, ins, splitGap, coilShrinkTop, coilShrinkBottom);
% 
% patch_uv(coilL.U, coilL.V, R_bore, th_center, [0.95 0.55 0.25], 'none',quarter_span,true);
% patch_uv(coilR.U, coilR.V, R_bore, th_center, [0.95 0.55 0.25], 'none',quarter_span,true);
d_edge = [1; 1; 0.5; 1]; % 对应每条边的偏移距离
P_coil = offset_polygon_inward(slot.U2, d_edge);

slot.coilU = P_coil(:,1);
slot.coilV = P_coil(:,2);

patch_uv(slot.coilU, slot.coilV, R_bore, th_center, [0.95 0.55 0.25], 'none',quarter_span,true);

d_edge = [-1; -1; -0.5; -1]; % 对应每条边的偏移距离
P_coil = offset_polygon_inward(slot.U3, d_edge);

slot.coilU = P_coil(:,1);
slot.coilV = P_coil(:,2);

patch_uv(slot.coilU, slot.coilV, R_bore, th_center, [0.95 0.55 0.25], 'none',quarter_span,true);

    u_coil = bottomW/4;                                        % 槽中心
    v_coil = (mouthH + slot_depth2)/2; % 线圈厚度中点（你根据自己变量名改）

    [xc, yc] = slot_uv2xy(u_coil, v_coil, R_bore, th_center);

    if xc>0
    mi_getmaterial('Copper');     % FEMM 自带 "Copper"

    mi_addblocklabel(xc, yc);
    mi_selectlabel(xc, yc);
    if(k==1)
    mi_setblockprop('Copper', 1, 0, 'B', 1, 0, -100);
    end
    if(k==2)
    mi_setblockprop('Copper', 1, 0, 'C', 1, 0, -100);
    end
    if(k==3)
    mi_setblockprop('Copper', 1, 0, 'A', 1, 0, -100);
    end
    mi_clearselected;
    end
        u_coil = -bottomW/4;                                        % 槽中心
    v_coil = (mouthH + slot_depth2)/2; % 线圈厚度中点（你根据自己变量名改）

    [xc, yc] = slot_uv2xy(u_coil, v_coil, R_bore, th_center);

    if yc>0
    mi_getmaterial('Copper');     % FEMM 自带 "Copper"

    mi_addblocklabel(xc, yc);
    mi_selectlabel(xc, yc);
    if(k==2)
    mi_setblockprop('Copper', 1, 0, 'B', 1, 0, 100);
    end
    if(k==3)
    mi_setblockprop('Copper', 1, 0, 'C', 1, 0, 100);
    end
    if(k==4)
    mi_setblockprop('Copper', 1, 0, 'A', 1, 0, 100);
    end
    mi_clearselected;
    end
end

% 4) 循环结束后，再补最后一段：最后一槽的 p2 -> quarter_span(2)
if ~isempty(prev_p2_ang)
    draw_arc_Rbore(R_bore, prev_p2_ang, th_q1, 5, 2);
end
 mi_selectsegment(0.24, 31.5);
 mi_deleteselected;


% 5) 气隙
draw_ring_sector(R_r_out, R_sy_out, quarter_span(1), quarter_span(2), 2);
draw_arc(R_r_out, quarter_span(1), quarter_span(2), maxsegdeg, 1);

% 创建名为 'Air gap' 的 Periodic Air Gap 边界，初始内外角都为 0°
mi_addboundprop('Air gap', ...
                0, 0, 0, 0, ...   % A0, A1, A2, phi  对气隙不用，设 0
                0, 0, ...         % mu, sigma       对气隙不用，设 0
                0, 0, ...         % c0, c1          对气隙不用，设 0
                6, ...            % BdryFormat = 6  => Periodic Air Gap
                0, 0);            % ia, oa          内角、外角（度）

[xs, ys] = pol2cart(45*pi/180, (R_bore+R_gap1)/2);

mi_getmaterial('Air');    % 或你自己定义的铁心材料

mi_addblocklabel(xs, ys);
mi_selectlabel(xs, ys);
mi_setblockprop('Air', 1, 0, '', 1, 0, 0);
mi_clearselected;

draw_arc(R_gap1, quarter_span(1), quarter_span(2), maxsegdeg, 2);

[xs, ys] = pol2cart(45*pi/180, R_gap1);
mi_selectarcsegment(xs,  ys);
mi_setarcsegmentprop(0, 'Air gap', 0, 0);
mi_clearselected;

draw_arc(R_gap2, quarter_span(1), quarter_span(2), maxsegdeg, 1);

[xs, ys] = pol2cart(45*pi/180, R_gap2);
mi_selectarcsegment(xs,  ys);
mi_setarcsegmentprop(0, 'Air gap', 0, 0);
mi_clearselected;

[xs, ys] = pol2cart(45*pi/180, (R_gap2+R_gap1)/2);

mi_addblocklabel(xs, ys);
mi_selectlabel(xs, ys);
mi_setblockprop('<No Mesh>', 1, 0, '', 1, 0, 0);
mi_clearselected;

[xs, ys] = pol2cart(45*pi/180, (R_gap2+R_r_out)/2);

mi_getmaterial('Air');    % 或你自己定义的铁心材料

mi_addblocklabel(xs, ys);
mi_selectlabel(xs, ys);
mi_setblockprop('Air', 1, 0, '', 1, 0, 0);
mi_clearselected;

% 6) 转子（轴+铁心）
draw_ring_sector(0, R_shaft, quarter_span(1), quarter_span(2), 1);
draw_ring_sector(R_shaft, R_r_out, quarter_span(1), quarter_span(2), 1);
[xs, ys] = pol2cart(45*pi/180, (R_shaft+R_r_out)/2);

mi_getmaterial('Pure Iron');    % 或你自己定义的铁心材料

mi_addblocklabel(xs, ys);
mi_selectlabel(xs, ys);
mi_setblockprop('Pure Iron', 1, 0, '', 0, 1, 0);
mi_clearselected;

[xs, ys] = pol2cart(45*pi/180, (R_shaft)/2);

mi_getmaterial('Air');    % 或你自己定义的铁心材料

mi_addblocklabel(xs, ys);
mi_selectlabel(xs, ys);
mi_setblockprop('Air', 1, 0, '', 0, 1, 0);
mi_clearselected;
%draw_ring_sector(R_shaft, R_r_core, quarter_span(1), quarter_span(2), 1);
% draw_arc(R_r_out, quarter_span(1), quarter_span(2), maxsegdeg, 1);   % 转子外圆

% 7) 永磁体（SPM）
i=1;
for c = mag_centers_quarter    % 你已有：处在 0–90° 里的磁体中心角
mi_draw_ipm_capsule_xy(Rin, Rout, c, phi_m, quarter_span, 1,((-1)^i));
i=i+1;
end

% 8) 边界线与注释（可选）

mi_addboundprop('apbc1', 0,0,0, 0, 0, 0, 0, 0, 4);
[xb1, yb1] = pol2cart(0*pi/180, R_sy_out+10);
[xb2, yb2] = pol2cart(90*pi/180, R_sy_out+10);
    mi_selectsegment(xb1, yb1);
    mi_setsegmentprop('apbc1', maxsegdeg, 1, 0, 0);
    mi_selectsegment(xb2, yb2);
    mi_setsegmentprop('apbc1', maxsegdeg, 1, 0, 0);
    mi_clearselected;

mi_addboundprop('apbc2', 0,0,0, 0, 0, 0, 0, 0, 4);
[xb1, yb1] = pol2cart(0*pi/180, (R_bore+slot_depth+R_sy_out)/2);
[xb2, yb2] = pol2cart(90*pi/180, (R_bore+slot_depth+R_sy_out)/2);
    mi_selectsegment(xb1, yb1);
    mi_setsegmentprop('apbc2', maxsegdeg, 1, 0, 0);
    mi_selectsegment(xb2, yb2);
    mi_setsegmentprop('apbc2', maxsegdeg, 1, 0, 0);
    mi_clearselected;


mi_addboundprop('apbc3', 0,0,0, 0, 0, 0, 0, 0, 4);
[xb1, yb1] = pol2cart(0*pi/180, (R_bore+R_sy_out)/2);
[xb2, yb2] = pol2cart(90*pi/180, (R_bore+R_sy_out)/2);
    mi_selectsegment(xb1, yb1);
    mi_setsegmentprop('apbc3', maxsegdeg, 1, 0, 0);
    mi_selectsegment(xb2, yb2);
    mi_setsegmentprop('apbc3', maxsegdeg, 1, 0, 0);
    mi_clearselected;

mi_addboundprop('apbc4', 0,0,0, 0, 0, 0, 0, 0, 4);
[xb1, yb1] = pol2cart(0*pi/180, (R_bore+R_gap1)/2);
[xb2, yb2] = pol2cart(90*pi/180, (R_bore+R_gap1)/2);
    mi_selectsegment(xb1, yb1);
    mi_setsegmentprop('apbc4', maxsegdeg, 1, 0, 0);
    mi_selectsegment(xb2, yb2);
    mi_setsegmentprop('apbc4', maxsegdeg, 1, 0, 0);
    mi_clearselected;

% mi_addboundprop('apbc5', 0,0,0, 0, 0, 0, 0, 0, 4);
% [xb1, yb1] = pol2cart(0*pi/180, (R_gap2+R_gap1)/2);
% [xb2, yb2] = pol2cart(90*pi/180, (R_gap2+R_gap1)/2);
%     mi_selectsegment(xb1, yb1);
%     mi_setsegmentprop('apbc5', maxsegdeg, 1, 0, 0);
%     mi_selectsegment(xb2, yb2);
%     mi_setsegmentprop('apbc5', maxsegdeg, 1, 0, 0);
%     mi_clearselected;

mi_addboundprop('apbc6', 0,0,0, 0, 0, 0, 0, 0, 4);
[xb1, yb1] = pol2cart(0*pi/180, (R_gap2+R_r_core)/2);
[xb2, yb2] = pol2cart(90*pi/180, (R_gap2+R_r_core)/2);
    mi_selectsegment(xb1, yb1);
    mi_setsegmentprop('apbc6', maxsegdeg, 1, 0, 0);
    mi_selectsegment(xb2, yb2);
    mi_setsegmentprop('apbc6', maxsegdeg, 1, 0, 0);
    mi_clearselected;

mi_addboundprop('apbc7', 0,0,0, 0, 0, 0, 0, 0, 4);
[xb1, yb1] = pol2cart(0*pi/180, (R_shaft+R_r_out)/2);
[xb2, yb2] = pol2cart(90*pi/180, (R_shaft+R_r_out)/2);
    mi_selectsegment(xb1, yb1);
    mi_setsegmentprop('apbc7', maxsegdeg, 1, 0, 1);
    mi_selectsegment(xb2, yb2);
    mi_setsegmentprop('apbc7', maxsegdeg, 1, 0, 1);
    mi_clearselected;

mi_addboundprop('apbc8', 0,0,0, 0, 0, 0, 0, 0, 4);
[xb1, yb1] = pol2cart(0*pi/180, (R_shaft)/2);
[xb2, yb2] = pol2cart(90*pi/180, (R_shaft)/2);
    mi_selectsegment(xb1, yb1);
    mi_setsegmentprop('apbc8', maxsegdeg, 1, 0, 1);
    mi_selectsegment(xb2, yb2);
    mi_setsegmentprop('apbc8', maxsegdeg, 1, 0, 1);
    mi_clearselected;

    
%% ==================== 工具函数区 ====================

function [x,y] = pol2cart_deg(r, th_deg)
    x = r.*cosd(th_deg);  y = r.*sind(th_deg);
end

function draw_arc(R, th1, th2, maxsegdeg, group)
    if nargin<4, maxsegdeg = 5; end
    if nargin<5, group      = 0; end

    [x1,y1] = pol2cart_deg(R, th1);
    [x2,y2] = pol2cart_deg(R, th2);

    dth  = th2 - th1;                         % 扫角，可以正也可以负
    nseg = 1; % 分段数必须是正数，至少 1 段 %就1段就可以了

    mi_addnode(x1,y1);
    mi_addnode(x2,y2);
    mi_addarc(x1,y1, x2,y2, dth, nseg);

    mi_selectarcsegment( (x1+x2)/2, (y1+y2)/2 );
    mi_setarcsegmentprop(maxsegdeg, '', 0, group);
    mi_clearselected;
end

function draw_ring_sector(Rin, Rout, th1, th2, group)
    if nargin<5, group = 0; end

    % 外弧 + 内弧
    draw_arc(Rout, th1, th2, 5, group);
    draw_arc(Rin,  th1, th2, 5, group);  % 注意反向

    % 两条径向直线
    [x1o,y1o] = pol2cart_deg(Rout, th1);
    [x1i,y1i] = pol2cart_deg(Rin,  th1);
    [x2o,y2o] = pol2cart_deg(Rout, th2);
    [x2i,y2i] = pol2cart_deg(Rin,  th2);

    mi_addnode(x1o,y1o); mi_addnode(x1i,y1i);
    mi_addnode(x2o,y2o); mi_addnode(x2i,y2i);

    mi_addsegment(x1o,y1o,x1i,y1i);
    mi_addsegment(x2o,y2o,x2i,y2i);

    mi_selectsegment((x1o+x1i)/2,(y1o+y1i)/2);
    mi_selectsegment((x2o+x2i)/2,(y2o+y2i)/2);
    mi_setsegmentprop('', 0, 0, 0, group);
    mi_clearselected;
end

function draw_radial_line(th, R)
    [x,y] = pol2cart_deg(R, th);  plot([0 x],[0 y],'k:');
end

function patch_uv(U, V, Rbore, th_center_deg, faceColor, edgeColor, qspan,close)
% 把局部(u,v)顶点 → 极坐标(r,th) → 在扇形θ∈[qspan(1),qspan(2)]内裁剪 → patch
% u:切向(mm)  v:径向(mm)  r = Rbore+v,  th = th_center + u/r(度)
    % ---- 1) 映射到极坐标 ----
    r  = Rbore + V(:);
    th = th_center_deg + (U(:) ./ r) * 180/pi;   % 弧长→角度(度)

    thmin = qspan(1); thmax = qspan(2);

    % 若全部在范围内就直接画，省去裁剪开销
    if all(th >= thmin & th <= thmax)
        [x,y] = pol2cart_deg(r, th);
   % 如果最后一个点已经和第一个点重复，就去掉最后一个，避免 0 长度线段
if x(1)==x(end) && y(1)==y(end)
    x(end) = [];
    y(end) = [];
end

n = numel(x);

% 1) 加所有节点
for k = 1:n
    mi_addnode(x(k), y(k));
end

% 2) 相邻点之间加线段，并把最后一个点和第一个点闭合
for k = 1:n-1
    mi_addsegment(x(k), y(k), x(k+1), y(k+1));
end
if close
mi_addsegment(x(n), y(n), x(1), y(1));    % 闭合多边形
end
        return;s
    end

    % ---- 2) 封闭多边形（首尾相连）----
    if r(1)~=r(end) || th(1)~=th(end)
        r  = [r;  r(1)];
        th = [th; th(1)];
    end

    % ---- 3) 两步 Sutherland–Hodgman 角度裁剪（就地实现，不调用外部函数）----
    % 半平面1：保留 th >= thmin
    r1=[]; th1=[];
    for i=1:numel(r)-1
        rA=r(i);   rB=r(i+1);
        aA=th(i);  aB=th(i+1);
        inA = (aA >= thmin); inB = (aB >= thmin);
        if inA && inB
            r1=[r1; rA]; th1=[th1; aA];
        elseif inA && ~inB
            t  = (thmin - aA)/(aB - aA);
            rI = rA + t*(rB - rA);
            r1=[r1; rA; rI]; th1=[th1; aA; thmin];
        elseif ~inA && inB
            t  = (thmin - aA)/(aB - aA);
            rI = rA + t*(rB - rA);
            r1=[r1; rI]; th1=[th1; thmin];
        end
    end
    if isempty(r1), return; end
    r1=[r1; r1(1)]; th1=[th1; th1(1)];

    % 半平面2：保留 th <= thmax
    r2=[]; th2=[];
    for i=1:numel(r1)-1
        rA=r1(i);   rB=r1(i+1);
        aA=th1(i);  aB=th1(i+1);
        inA = (aA <= thmax); inB = (aB <= thmax);
        if inA && inB
            r2=[r2; rA]; th2=[th2; aA];
        elseif inA && ~inB
            t  = (thmax - aA)/(aB - aA);
            rI = rA + t*(rB - rA);
            r2=[r2; rA; rI]; th2=[th2; aA; thmax];
        elseif ~inA && inB
            t  = (thmax - aA)/(aB - aA);
            rI = rA + t*(rB - rA);
            r2=[r2; rI]; th2=[th2; thmax];
        end
    end
    if numel(r2)<3, return; end

    % % 移除闭合重复点
    % r2(end)=[]; th2(end)=[];
% ---- 4) 在 FEMM 里连线 ----
[x,y] = pol2cart_deg(r2, th2);

% 如果最后一个点已经和第一个点重复，就去掉最后一个，避免 0 长度线段
if x(1)==x(end) && y(1)==y(end)
    x(end) = [];
    y(end) = [];
end

n = numel(x);

% 1) 加所有节点
for k = 1:n
    mi_addnode(x(k), y(k));
end

% 2) 相邻点之间加线段，并把最后一个点和第一个点闭合
for k = 1:n-1
    mi_addsegment(x(k), y(k), x(k+1), y(k+1));
end
if close
mi_addsegment(x(n), y(n), x(1), y(1));    % 闭合多边形
end
end

function draw_arc_Rbore(R, th1, th2, maxsegdeg, group)
    if nargin < 4, maxsegdeg = 5; end
    if nargin < 5, group = 0;      end

    [x1,y1] = pol2cart_deg(R, th1);
    [x2,y2] = pol2cart_deg(R, th2);

    % mi_addnode(x1,y1);
    % mi_addnode(x2,y2);

    mi_addarc(x1,y1, x2,y2, th2-th1, ceil((th2-th1)/maxsegdeg));

    % 设置弧段属性（组号等，可选）
    % xm = (x1 + x2)/2;
    % ym = (y1 + y2)/2;
    % mi_selectarcsegment(xm,ym);
    % mi_setarcsegmentprop(maxsegdeg,'',0,group);
    % mi_clearselected;
end

function S = build_slot_polys(mouthW, mouthH, bottomW, bottomH, slotDepth,Depth2,lipW, edgeFillet)
% 生成"矩形槽口 + 对称六边形主槽体"的多边形顶点（局部坐标 u/v）
% 顶点顺序：按顺时针列出，patch 会自动闭合
    %#ok<INUSD>
    w1 = mouthW;        % 主槽上口宽（与槽口矩形相接）
    w2 = bottomW;       % 主槽底宽
    v0 = 0;             % 定义槽口矩形从 v=0 到 v=mouthH
    v1 = mouthH;        % 主槽从 v=mouthH 开始
    v2 = bottomH;     % 主槽底
    v22=slotDepth;
    v12= mouthH+Depth2;

    % 槽体六边形（上边是 v1 处的 w1，下边是 v2 处的 w2）
    U = [ ...
         lipW/2, 0;
         lipW/2,(v1+v12)/2;
         w1/2,  v1;   % 上右    
         w2/2,  v2;   % 下右
         0,  v22;
        -w2/2,  v2;   % 下左    
        -w1/2,  v1;   % 上左
        -lipW/2,(v1+v12)/2;        
        -lipW/2, 0;
    ];


    % 合并为一个顺时针多边形：矩形顶→右→下到主槽→左→回到矩形顶
    body = [U(1,:);U(2,:); U(3,:); U(4,:);U(5,:); U(6,:);U(7,:);U(8,:);U(9,:)]; %#ok<NASGU>

    S.bodyU = body(:,1)';   % 行向量
    S.bodyV = body(:,2)';
    S.p1=[ -lipW/2, 0];
    S.p2=[ lipW/2, 0];

        S.U2 = [
         w1/2,  v1;   % 上右    
         w2/2,  v2;   % 下右
         0,  v22;
        % -w2/2,  v2;   % 下左    
        % -w1/2,  v1;   % 上左
           0,v12;
    ];
                
        S.U3 = [
        -w1/2,  v1;      
        -w2/2,  v2;  
         0,  v22;
         0,v12;
    ];

end

function [x,y] = slot_uv2xy(u, v, R_bore, th_center)
    r   = R_bore + v;            % 齿根圆往外 v
    dth = (u ./ r) * 180/pi;     % 切向位移 u -> 角度偏移（度）
    th  = th_center + dth;       % 槽中心角 th_center

    x = r .* cosd(th);
    y = r .* sind(th);
end

function [x,y,th_deg] = slot_uv2xy_angle(u, v, R_bore, th_center)
    r   = R_bore + v;
    th_deg = th_center + (u ./ r) * 180/pi;
    x = r .* cosd(th_deg);
    y = r .* sind(th_deg);
end

function delete_arc_between_points(R_bore, p1, p2)
% 删除 "半径 = R_bore" 上，经过 p1,p2 的那一小段圆弧
% p1, p2 是全局坐标 [x y]

    % 1) 两点的极角（度）
    th1 = atan2d(p1(2), p1(1));
    th2 = atan2d(p2(2), p2(1));

    % 2) 处理跨 0° 情况（比如 350° 和 10°）
    if abs(th2 - th1) > 180
        if th1 < th2
            th1 = th1 + 360;
        else
            th2 = th2 + 360;
        end
    end

    % 3) 圆弧中间方向的角度
    thm = 0.5 * (th1 + th2);

    % 4) 圆弧上的中点（半径强制 = R_bore）
    xm = R_bore * cosd(thm);
    ym = R_bore * sind(thm);

    % 4) 只选弧，不选直线
    mi_addnode(xm, ym);  % 如需调试可以暂时打开看看点落在哪
    mi_clearselected;
    % mi_selectsegment(xm, ym);
    % mi_deleteselected;
        mi_clearselected;
    mi_selectarcsegment(xm, ym);
    mi_deleteselected;
    % x1 = xL; y1 = yL;
    % x2 = xR; y2 = yR;
    % 
    % % 1) 计算两点对应的极坐标
    % th1 = atan2d(y1, x1);      % 角度
    % th2 = atan2d(y2, x2);
    % r1  = hypot(x1, y1);
    % r2  = hypot(x2, y2);
    % rmid = 0.5*(r1+r2);        % 半径取中间值
    % 
    % % 2) 用中间角度算出弧上的一个点
    % thm = 0.5.*(th1 + th2);     % 弧的"中间角"
    % xm  = rmid.*cosd(thm);
    % ym  = rmid.*sind(thm);
    % 
    % % 3) 选中该弧段并删除
    % mi_selectarcsegment(xm, ym);
    % mi_deleteselected;
end

function [L, R] = build_coils_in_slot(mouthW, mouthH, bottomW, bottomH, slotDepth, Depth2, ins, splitGap, shrinkTop, shrinkBot)
% 在主槽体内部生成两块对称的四边形线圈
                                      
% 思路：在 v=v1..v2 区间内，令线圈的上/下宽度相对主槽再缩一些，并在中间留 splitGap
    v1 = mouthH + ins;           % 线圈起始径向，避开槽口矩形与绝缘
    v2 = bottomH - (bottomH/slotDepth)* ins;        % 线圈底部向上退 ins
    d1 = Depth2 * (mouthW-splitGap)/mouthW
    d2 = Depth2 * (slotDepth-splitGap)/slotDepth
    v22 = slotDepth - ins;
    v11= mouthH + d1 + ins; 

    % 主槽在 v1/v2 的理论"可用总宽"（线性插值）
    wTop   = max(mouthW - ins, 1e-3);
    wBot   = max(bottomW - 2*ins, 1e-3);

    % 在线圈层面再额外收缩（外缘到线圈之间留制造退让）
    wTopCoil = max(wTop - 2*shrinkTop, 1e-3);
    wBotCoil = max(wBot - 2*shrinkBot, 1e-3);

    % 左/右各占： 0.5*(可用宽 - 中缝)
    halfTop  = 0.5*(wTop  - splitGap);
    halfBot  = 0.5*(wBot  - splitGap);
    halfTop  = max(halfTop,  0.2);
    halfBot  = max(halfBot,  0.2);

    % —— 左线圈（顺时针四边形）——
    UL = [ ...
        -splitGap/2 - halfTop, v1;   % 上内
        -splitGap/2,           v11;   % 上中（靠缝）
        -splitGap/2,           v22;   % 下中
        -splitGap/2 - halfBot, v2;   % 下外
    ];
    % —— 右线圈（顺时针四边形，镜像）——
    UR = [ ...
         splitGap/2,           v11;
         splitGap/2 + halfTop, v1;
         splitGap/2 + halfBot, v2;
         splitGap/2,           v22;
    ];

    L.U = UL(:,1)'; L.V = UL(:,2)';
    R.U = UR(:,1)'; R.V = UR(:,2)';
end

function Q = offset_polygon_inward(P, d_edge)
% P      : N×2，多边形顶点，逆时针
% d_edge : N×1，每条边向内偏移的距离，
%          d_edge(i) 对应边 P(i)->P(i+1)

    N = size(P,1);
    Q = zeros(N,2);

    t     = zeros(N,2);
    n_in  = zeros(N,2);

    % 1) 每条边的切向和内法线
    for i = 1:N
        j = mod(i, N) + 1;
        e = P(j,:) - P(i,:);
        L = hypot(e(1), e(2));
        t(i,:) = e / L;
        n_out  = [ t(i,2), -t(i,1)];
        n_in(i,:) = -n_out;
    end

    % 2) 相邻两条偏移直线求交点
    for i = 1:N
        im1 = i-1; if im1==0, im1 = N; end

        % 上一条边偏移 d_edge(im1)
        p1 = P(im1,:) + d_edge(im1)*n_in(im1,:);
        v1 = t(im1,:);

        % 当前边偏移 d_edge(i)
        p2 = P(i,:)   + d_edge(i)*n_in(i,:);
        v2 = t(i,:);

        Q(i,:) = line_intersect(p1, v1, p2, v2);
    end
end

function X = line_intersect(p1, v1, p2, v2)
% 求两条直线 p1 + s*v1 与 p2 + t*v2 的交点 X
% 若两直线几乎平行，则做一个稳健的退化处理

    A = [v1(:), -v2(:)];          % 2x2
    b = (p2 - p1).';

    % 行列式判断是否接近平行
    detA = A(1,1)*A(2,2) - A(1,2)*A(2,1);

    if abs(detA) < 1e-12
        % —— 退化情况：两条线几乎平行 ——
        % 这里我们直接把交点放在"这两条偏移线的中点附近"，
        % 对于槽这种几乎直线的角，相当于把顶点简单平移了一小段，几何上是合理的。

        % 把 p2 投影到 p1 所在直线，再取中点：
        v1n = v1 / norm(v1);
        proj = dot(p2 - p1, v1n);
        p2_on_line1 = p1 + proj * v1n;

        X = 0.5 * (p2_on_line1 + p2);   % 取一个折中的点
    else
        % —— 正常情况：两线有明显夹角 ——
        st = A \ b;                    % 解 [s; t]
        X  = p1 + st(1)*v1;
    end
end

function mi_draw_ipm_capsule_xy(Rin, Rout, th_center_deg, phi_deg, qspan, group,i)
% 在 FEMM 中画一块"胶囊形"内嵌永磁体（只画轮廓线，不设材料）
% Rin / Rout       : 磁体内外半径
% th_center_deg    : 磁体中心角（度）
% phi_deg          : 磁体在圆周方向的弧角（度）
% qspan            : [thmin thmax] 当前扇形可视范围（度），超出部分自动裁掉
% group            : 这块磁体轮廓段所属的 group 号（例如 30）

    if nargin < 6
        group = 0;
    end

    thmin = qspan(1);
    thmax = qspan(2);

    % ---- 1) 角度裁剪 ----
    th1 = th_center_deg - phi_deg/2;   % 左端角
    th2 = th_center_deg + phi_deg/2;   % 右端角

    th1 = max(th1, thmin);
    th2 = min(th2, thmax);

    if th2 <= th1
        return;   % 完全在可视范围外
    end

    % ---- 2) 基本几何量 ----
    Rc   = 0.5*(Rin + Rout);      % 磁体中心半径
    rcap = 0.5*(Rout - Rin);      % 端盖圆半径（也等于磁体"厚度"的一半）
    sArc = (th2 - th1) * pi/180 * Rc;   % 磁体弧长（沿切向）

    if sArc < 1e-6
        return;
    end

    % 磁体局部坐标系：n = 径向方向，t = 切向方向
    thc = (th1 + th2)/2;              % 磁体中心角（裁剪后的）
    n = [cosd(thc);  sind(thc)];      % 径向单位向量
    t = [-sind(thc); cosd(thc)];      % 切向单位向量（逆时针）

    % 磁体中心
    C  = Rc * n;

    % 两个半圆端盖的圆心（沿切向方向错开）
    C1 = C - 0.5*sArc * t;   % 左端盖圆心
    C2 = C + 0.5*sArc * t;   % 右端盖圆心

% ---- 3) 生成边界点序列（按顺时针或逆时针都可以）----
Ncap = 24;   % 半圆端盖上的离散点数

% 左端半圆：从"左内" -> "左外"
betaL = linspace(-180, 0, Ncap);      % 180°(内) -> 0°(外)
csL   = cosd(betaL(:));              % Ncap×1
ssL   = sind(betaL(:));              % Ncap×1

% n, t 都是 2×1 列向量
nx = n(1); ny = n(2);
tx = t(1); ty = t(2);

% 左半圆各点的偏移量：沿 n 方向 + 沿 t 方向
offL = rcap * csL;   % N×1
offT = rcap * ssL;   % N×1

PLx = C1(1) + offL*nx + offT*tx;  % N×1
PLy = C1(2) + offL*ny + offT*ty;  % N×1
PL  = [PLx, PLy];                 % N×2

% 右端半圆：从"右外" -> "右内"
betaR = linspace(0, 180, Ncap);   % 0°(外) -> 180°(内)
csR   = cosd(betaR(:));
ssR   = sind(betaR(:));

offL = rcap * csR;
offT = rcap * ssR;

PRx = C2(1) + offL*nx + offT*tx;
PRy = C2(2) + offL*ny + offT*ty;
PR  = [PRx, PRy];


    % 现在构造一圈边界点：
    % 1) 左端半圆：左内 -> 左外  (PL(1,:) -> PL(end,:))
    % 2) 外侧直线：左外 -> 右外  (PL(end,:) -> PR(1,:))
    % 3) 右端半圆：右外 -> 右内  (PR(1,:) -> PR(end,:))
    % 4) 内侧直线：右内 -> 左内  (PR(end,:) -> PL(1,:))
    B = [PL;                     % Ncap 行
         PR(1,:);                % +1 行: 外侧直线终点
         PR(2:end,:);            % Ncap-1 行: 右端半圆剩余点
         PL(1,:)];               % +1 行: 内侧直线终点 = 起点，方便闭合

    % ---- 4) 在 FEMM 中画节点和线段 ----
    nB = size(B,1);

    % 所有边界点
    for k = 1:nB
        mi_addnode(B(k,1), B(k,2));
    end

    % 相邻点之间连线
    for k = 1:nB-1
        mi_addsegment(B(k,1), B(k,2), B(k+1,1), B(k+1,2));
            % 2. 选中刚画的这条线段（用中点坐标选）
     x1 = B(k,1);   y1 = B(k,2);
    x2 = B(k+1,1); y2 = B(k+1,2);
    xm = (x1 + x2)/2;
    ym = (y1 + y2)/2;
    mi_selectsegment(xm, ym);
    mi_setsegmentprop('', 0, 0, '0', group);
    mi_clearselected;
    end
    P_L_in  = PL(1,:);       % 左内
P_L_out = PL(end,:);     % 左外
P_R_out = PR(1,:);       % 右外
P_R_in  = PR(end,:);     % 右内
% 外侧边
mi_addsegment(P_L_out(1), P_L_out(2), P_L_in(1), P_L_in(2));
    x1 = P_L_out(1);   y1 = P_L_out(2);
    x2 = P_L_in(1); y2 = P_L_in(2);
    xm = (x1 + x2)/2;
    ym = (y1 + y2)/2;
    mi_selectsegment(xm, ym);
    mi_setsegmentprop('', 0, 0, '0', group);
    mi_clearselected;

% 内侧边
mi_addsegment(P_R_in(1), P_R_in(2), P_R_out(1), P_R_out(2));
    x1 = P_R_in(1);   y1 = P_R_in(2);
    x2 = P_R_out(1); y2 = P_R_out(2);
    xm = (x1 + x2)/2;
    ym = (y1 + y2)/2;
    mi_selectsegment(xm, ym);
    mi_setsegmentprop('', 0, 0, '0', group);
    mi_clearselected;
    % 闭合最后一条边（理论上上面已经闭了，这里再保险一次也没问题）
    mi_addsegment(B(end,1), B(end,2), B(1,1), B(1,2));
    x1 = B(end,1);   y1 = B(end,2);
    x2 = B(1,1); y2 = B(1,2);
    xm = (x1 + x2)/2;
    ym = (y1 + y2)/2;
    mi_selectsegment(xm, ym);
    mi_setsegmentprop('', 0, 0, '0', group);
    mi_clearselected;

    % % ---- 5) 给这些线段设置 group 号（可选）----
    %     % 用一个小包围框把刚画出来的线段选中，再设 group
    %     xmin = min(B(:,1)); xmax = max(B(:,1));
    %     ymin = min(B(:,2)); ymax = max(B(:,2));
    %     dx = (xmax - xmin)*0.1 + 1e-3;
    %     dy = (ymax - ymin)*0.1 + 1e-3;
    % 
    %     mi_clearselected;
    %     mi_selectrectangle(xmin-dx, ymin-dy, xmax+dx, ymax+dy, 4);  % 4=只选边界
    %     mi_setsegmentprop('', 0, 0, '0', group);
    %     mi_clearselected;
    % 1. 取一个在磁体内部的点（基本就是磁体中心）
Rc = 0.5*(Rin + Rout);
[xm, ym] = pol2cart(th_center_deg*pi/180, Rc);

% 2. 在 FEMM 中添加材料（只需要做一次，可以放在更前面）
mi_getmaterial('N40');    % 或者你自己的材料名

% 3. 放 block label 并设置材料
mi_addblocklabel(xm, ym);
mi_selectlabel(xm, ym);
if i > 0 
MagDir = th_center_deg;  
else
    MagDir = th_center_deg-180;  
end
mi_setblockprop('N40', 1, 0, '', MagDir, group, 0);
mi_clearselected;


% 2. 在 FEMM 中添加材料（只需要做一次，可以放在更前面）
mi_getmaterial('Air');    % 或者你自己的材料名

[xm, ym] = pol2cart(th_center_deg*pi/180 - 0.5*(th2 - th1) * pi/180, Rout);

% 3. 放 block label 并设置材料
mi_addblocklabel(xm, ym);
mi_selectlabel(xm, ym);
 
mi_setblockprop('Air', 1, 0, '', 0,group, 0);
mi_clearselected;

[xm, ym] = pol2cart(th_center_deg*pi/180 + 0.5*(th2 - th1) * pi/180, Rout);

% 3. 放 block label 并设置材料
mi_addblocklabel(xm, ym);
mi_selectlabel(xm, ym);
 
mi_setblockprop('Air', 1, 0, '', 0,group, 0);
mi_clearselected;

end

function [theta_deg, T] = torque_cogging_scan(inp)

    rotor_group = 1;      % 你把转子所有 block 都设成 group=1 了
    innerIndex  = 10;     % ia = Inner Angle, Deg 在 mi_addboundprop 里的索引是 10

    dtheta    = 1;        % 每步 1°
    maxAngle  = 90;       % 扫 0~90°
    theta_deg = 0:dtheta:maxAngle;
    nSteps    = numel(theta_deg);
    T         = zeros(nSteps,1);

delta_e_deg = 90;      % 电角度电流相位（90° ≈ 只在 q 轴上，表贴机型大致最大转矩）
delta_e = delta_e_deg*pi/180;

    % 先保存一次。
    mi_saveas('temp_airgap_scan.fem');

    % --- 2) 扫描角度 ---
    for k = 1:nSteps
        th = theta_deg(k);          % 机械角，单位：deg

        % 2.1 只改 Air gap 边界的 Inner Angle = 转子角度
        mi_modifyboundprop('Air gap', innerIndex, th);
        % Outer Angle(11) 一直保持 0，不用动

            % ===== 新增：根据当前转子位置设置三相电流 =====
    theta_e = 4 * th * pi/180 + delta_e;  % 电角度（rad）

    Ia = inp * cos(theta_e);
    Ib = inp * cos(theta_e - 2*pi/3);
    Ic = inp * cos(theta_e - 4*pi/3);

    % 写入三相电路电流（propID=0 表示电流）
    mi_modifycircprop('A', 1, Ia);
    mi_modifycircprop('B', 1, Ib);
    mi_modifycircprop('C', 1, Ic);

        % 2.2 求解
        mi_analyze(false);
        mi_loadsolution;
        mi_zoomnatural;           % 自动缩放到合适大小

        % 2.3 在转子组上做转矩积分
        mo_groupselectblock(rotor_group);
        T(k) = mo_blockintegral(22);    % 22 = Torque
        mo_clearblock;
        mo_close;
    end

    % --- 3) 简单画图 ---
    figure;
    plot(theta_deg, T, '-o');
    xlabel('Mechanical angle / deg');
    ylabel('Electromagnetic torque / Nm');
    grid on;
    title('Loaded torque vs. rotor position (Air-gap BC)');

    % closefemm;   % 需要的话自己决定什么时候关
end

% [theta_deg, T] = torque_cogging_scan(Is_amp)
% 
%     T_avg    = mean(T);
%     T_max    = max(T);
%     T_min    = min(T);
%     T_ripple = (T_max - T_min)/T_avg;

