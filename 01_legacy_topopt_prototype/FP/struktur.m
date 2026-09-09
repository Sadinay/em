%% 8极12槽 SPM 电机四分之一结构图（含中间开口+有厚度齿唇的槽口）
% 保存文件名：pmsm_8p12s_quarter_gap.m
% 说明：参数化绘图，仅作示意，单位 mm / 度（deg）

clear; clc; close all;
addpath('C:\femm42\mfiles');   % 内含 femm.m 和一堆 mi_*/mo_* 封装函数
savepath;                                     % 可选：下次自动生效


%% ---------- 基本参数 ----------
Q       = 12;                 % 槽数
P       = 8;                  % 极数（机械极数）
thetaQ  = 360/Q;              % 槽距（机械角）
thetaP  = 360/P;              % 极距（机械角）

R_sy_out   = 100;             % 定子外半径
t_yoke     = 12;              % 定子轭厚
R_sy_in    = R_sy_out - t_yoke;  % 定子内半径（槽底近似半径）

R_bore     = 70;              % 定子内圆（气隙外圆）
airgap     = 3;             % 气隙
R_r_out    = R_bore - airgap; % 转子外圆（磁体外圆）
R_gap1 = R_bore-1;
R_gap2= R_r_out+1;
R_shaft    = 15;              % 轴半径
R_r_core   = 60;              % 转子铁心外半径

t_mag      = 10;               % 永磁体厚度（径向）
beta_m     = 1;            % 磁体弧比（相对极距 0~1）

slot_depth   = R_sy_in - R_bore;
depth2 = 1; %槽六边形向外突出的程度
slot_depth2   = slot_depth+depth2; % 槽深（示意）
slot_opening = 8;                  % "传统口宽弧长(mm)"（仅用于给出参考角度）

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
mouthW   = 8;         % 槽口宽度（矩形口的切向宽，mm）
mouthH   = 2;         % 槽口矩形的径向高度（mm）
bottomW  = 16;        % 槽底宽度（主槽体底边切向宽，mm）
edgeFillet = 0;         % 先不做圆角（0=无）

lipW=1; %槽口宽度
lipH=1;

% ---------- 线圈/绝缘（局部参数） ----------
ins     = 1;          % 槽内绝缘/气隙到线圈的"退让"（四周统一退 ins，mm）
splitGap= 1;          % 中央分裂缝（两线圈之间的净间隙，mm）
coilShrinkTop   = 0;  % 线圈相对主槽在"上口"额外再缩一点（形似上窄）
coilShrinkBottom= 0;  % 线圈相对主槽在"底边"额外再缩一点（形似下窄）

% 假设转子外圆 R_r_out、转子铁心外半径 R_r_core 已有
% 典型参数（按需要改）
phi_m      = 22;     % 每块磁体切向跨度(度) —— 越大越长
t_mag      = 6;    % 磁体厚度(mm) = Rout - Rin
t_bridge   = 3;    % 表面桥厚(mm)：磁体到转子外圆之间的铁桥
% 设定磁体的内外半径（放在外圆之下、留桥）
Rout = R_r_out - t_bridge;     % 磁体外半径（靠近气隙侧）
Rin  = Rout - t_mag;           % 磁体内半径（靠近铁心侧）

bodyC = [0.90 0.55 0.55];      % 磁体主体颜色（淡红）
capC  = [1 1 1];               % 端盖颜色（白色，符合你的要求）
edgeC = [0 0 0];               % 黑色描边                  % 参考半径（用 R_in 很方便）

%% ---------- 预计算角度（槽中心、磁体中心） ----------
slot_centers = (0:Q-1)*thetaQ;
slot_centers_quarter = slot_centers(slot_centers >= quarter_span(1) & slot_centers <= quarter_span(2));

pole_pitch = thetaP;
mag_span   = beta_m * pole_pitch;
mag_centers = (thetaP/2) : thetaP : 360;  % 22.5°, 67.5° ...
mag_centers_quarter = mag_centers(mag_centers>quarter_span(1) & mag_centers<quarter_span(2));

%% ---------- 绘图 ----------
figure('Color','w'); hold on; axis equal off;

% 背景扇形（淡灰，非必须）
draw_ring_sector(R_shaft, R_sy_out, quarter_span(1), quarter_span(2), [0.98 0.98 0.98], 'none');

% 1) 定子轭
stator_color = [0.85 0.90 0.95];
draw_ring_sector(R_sy_in, R_sy_out, quarter_span(1), quarter_span(2), [0.80 0.85 0.90], 0.3*[1 1 1]);

% 2) 齿区铺底
draw_ring_sector(R_bore, R_sy_in, quarter_span(1), quarter_span(2), stator_color, 'none');

% 3) （把内圆弧先画在底层，避免"封口视觉"）
% draw_arc(R_bore, quarter_span(1), quarter_span(2), 'none', 0.1);

% 4) 在齿区"挖槽 + 补两侧齿唇"
for th = slot_centers_quarter

slot = build_slot_polys(mouthW, mouthH, bottomW, slot_depth, depth2, 0);
patch_uv(slot.bodyU, slot.bodyV, R_bore, th, [1 1 1], 'none',quarter_span);

lip= build_lips(lipW,mouthH,depth2);
patch_uv(lip.lipU, lip.lipV, R_bore, th, [1 1 1], 'none',quarter_span);

[coilL, coilR] = build_coils_in_slot(mouthW, mouthH, bottomW, slot_depth, depth2, ins, splitGap, coilShrinkTop, coilShrinkBottom);
                                     
patch_uv(coilL.U, coilL.V, R_bore, th, [0.95 0.55 0.25], 'none',quarter_span);
patch_uv(coilR.U, coilR.V, R_bore, th, [0.95 0.55 0.25], 'none',quarter_span);

end

% 5) 气隙
draw_ring_sector(R_r_out, R_bore, quarter_span(1), quarter_span(2), [1 1 1], 'none');

draw_ring_sector(R_gap2, R_gap1, quarter_span(1), quarter_span(2), [1 1 1], 0.5*[1 1 1]);

% 6) 转子（轴+铁心）
draw_ring_sector(0, R_shaft, quarter_span(1), quarter_span(2), 0.85*[0.85 0.85 0.85], 'none');
% draw_ring_sector(R_shaft, R_r_core, quarter_span(1), quarter_span(2), [0.92 0.92 0.92], 0.5*[1 1 1]);
draw_arc(R_r_out, quarter_span(1), quarter_span(2), 'k', 1.0);   % 转子外圆

% 7) 永磁体（SPM）
for c = mag_centers_quarter    % 你已有：处在 0–90° 里的磁体中心角
    draw_ipm_capsule_xy(Rin, Rout, c, phi_m, bodyC, edgeC, capC, quarter_span);
end

% 8) 边界线与注释（可选）
draw_radial_line(quarter_span(1), R_sy_out);
draw_radial_line(quarter_span(2), R_sy_out);
% title('8极12槽 SPM 电机四分之一剖面（中间开口槽口 + 齿唇厚度）');

xlim([0 R_sy_out*1.02]); ylim([0 R_sy_out*1.02]);

%% ==================== 工具函数区 ====================

function [x,y] = pol2cart_deg(r, th_deg)
    x = r.*cosd(th_deg);  y = r.*sind(th_deg);
end

function draw_arc(R, th1, th2, ec, lw)
    th = linspace(th1, th2, 400);
    [x,y] = pol2cart_deg(R, th);
    plot(x,y,'Color',ec,'LineWidth',lw);
end

function draw_ring_sector(Rin, Rout, th1, th2, fc, ec)
    th = linspace(th1, th2, 220);
    [xo, yo] = pol2cart_deg(Rout, th);
    [xi, yi] = pol2cart_deg(Rin,  fliplr(th));
    patch([xo,xi],[yo,yi], fc, 'EdgeColor', ec);
end

function draw_radial_line(th, R)
    [x,y] = pol2cart_deg(R, th);  plot([0 x],[0 y],'k:');
end

function patch_uv(U, V, Rbore, th_center_deg, faceColor, edgeColor, qspan)
% 把局部(u,v)顶点 → 极坐标(r,th) → 在扇形θ∈[qspan(1),qspan(2)]内裁剪 → patch
% u:切向(mm)  v:径向(mm)  r = Rbore+v,  th = th_center + u/r(度)

    % ---- 1) 映射到极坐标 ----
    r  = Rbore + V(:);
    th = th_center_deg + (U(:) ./ r) * 180/pi;   % 弧长→角度(度)

    thmin = qspan(1); thmax = qspan(2);

    % 若全部在范围内就直接画，省去裁剪开销
    if all(th >= thmin & th <= thmax)
        [x,y] = pol2cart_deg(r, th);
        patch(x, y, faceColor, 'EdgeColor', edgeColor, 'LineWidth', 0.9);
        return;
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

    % 移除闭合重复点
    r2(end)=[]; th2(end)=[];

    % ---- 4) 画出来 ----
    [x,y] = pol2cart_deg(r2, th2);
    patch(x, y, faceColor, 'EdgeColor', edgeColor, 'LineWidth', 0.9);
end


function S = build_slot_polys(mouthW, mouthH, bottomW, slotDepth,Depth2, edgeFillet)
% 生成"矩形槽口 + 对称六边形主槽体"的多边形顶点（局部坐标 u/v）
% 顶点顺序：按顺时针列出，patch 会自动闭合
    %#ok<INUSD>
    w1 = mouthW;        % 主槽上口宽（与槽口矩形相接）
    w2 = bottomW;       % 主槽底宽
    v0 = 0;             % 定义槽口矩形从 v=0 到 v=mouthH
    v1 = mouthH;        % 主槽从 v=mouthH 开始
    v2 = slotDepth;     % 主槽底
    v22=slotDepth+Depth2;
    v12= mouthH+Depth2;

    % 槽体六边形（上边是 v1 处的 w1，下边是 v2 处的 w2）
    U = [ ...
        -w1/2,  v1;   % 上左
         w1/2,  v1;   % 上右
         w2/2,  v2;   % 下右
        -w2/2,  v2;   % 下左
        0,  v22;
        0, v12
    ];
    % 槽口矩形（拼在上方）
    UR = [ ...
        -mouthW/2, v0;
         mouthW/2, v0;
         mouthW/2, v1;
        -mouthW/2, v1;
    ];

    % 合并为一个顺时针多边形：矩形顶→右→下到主槽→左→回到矩形顶
    body = [U(2,:); U(3,:); U(5,:);U(4,:); U(1,:);U(6,:)]; %#ok<NASGU>

    S.bodyU = body(:,1)';   % 行向量
    S.bodyV = body(:,2)';
end

function L=build_lips(lipW,mouthH,depth2)

    UR = [ ...
        -lipW/2, 0;
         lipW/2, 0;
         lipW/2, mouthH+depth2;
        -lipW/2, mouthH+depth2;
    ];

        lip = [UR(1,:); UR(2,:); UR(3,:);UR(4,:)]; %#ok<NASGU>

    L.lipU = lip(:,1)';   % 行向量
    L.lipV = lip(:,2)';

end

function [L, R] = build_coils_in_slot(mouthW, mouthH, bottomW, slotDepth, Depth2, ins, splitGap, shrinkTop, shrinkBot)
% 在主槽体内部生成两块对称的四边形线圈
                                      
% 思路：在 v=v1..v2 区间内，令线圈的上/下宽度相对主槽再缩一些，并在中间留 splitGap
    v1 = mouthH + ins;           % 线圈起始径向，避开槽口矩形与绝缘
    v2 = slotDepth - ins;        % 线圈底部向上退 ins
    d1 = Depth2 * (mouthW-splitGap)/mouthW
    d2 = Depth2 * (slotDepth-splitGap)/slotDepth
    v22 = slotDepth + d2 - ins;
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

function draw_ipm_capsule_xy(Rin, Rout, th_center, phi_deg, bodyColor, edgeColor, capColor, qspan)
% 在全局 x-y 中精确绘制"内嵌胶囊形"永磁体（矩形+两端半圆）
% - Rin/Rout : 磁体内/外半径
% - th_center: 磁体中心角(度)
% - phi_deg  : 磁体切向角宽(度)
% - bodyColor: 主体颜色（如 [0.90 0.55 0.55]）
% - edgeColor: 描边颜色（如 [0 0 0]）
% - capColor : 端盖颜色（白色就用 [1 1 1]）
% - qspan    : 可视扇区 [thmin thmax]（如 [0 90]），超出的部分自动不画

    % ---- 基本量 ----
    th1 = th_center - phi_deg/2;
    th2 = th_center + phi_deg/2;
    th1e = max(th1, qspan(1));          % 角度裁剪后左/右端角
    th2e = min(th2, qspan(2));
    if th2e <= th1e, return; end


    % ---------- 主体（不含端盖）：环形扇形条 ----------


    Rc   = 0.5*(Rin + Rout);            % 端盖圆心半径（矩形中心也在该半径）
    rcap = 0.5*(Rout - Rin);            % 半圆半径 = 厚度/2
    sArc = (th2e - th1e) * pi/180 * Rc; % 端盖圆心间的切向距离（矩形长度）
    if sArc < 1e-6, return; end

    % ---- 中心方向（裁剪后）与切向/法向基 ----
    thc = 0.5*(th1e + th2e);            % 有效中心角
    n   = [cosd(thc);  sind(thc)];      % 径向法向
    t   = [-sind(thc); cosd(thc)];      % 切向
    C   = Rc * n;                       % 矩形中心

    % ---- 两端端盖圆心（在切向方向±sArc/2）----
    C1 = C - (sArc/2)*t;                % 左端圆心
    C2 = C + (sArc/2)*t;                % 右端圆心

        % ---- 两端半圆端盖（方向已修正：左-1，右+1）----
    draw_cap(th1e, C1, -1);     % 左端盖：外→内
    draw_cap(th2e, C2, +1);     % 右端盖：内→外


    % ---- 矩形主体四个角：按顺时针（左上→右上→右下→左下）----
    % 角点 = 端盖圆心 ± rcap * n
    P = [ (C1 + rcap*n).';   % 左上
          (C2 + rcap*n).';   % 右上
          (C2 - rcap*n).';   % 右下
          (C1 - rcap*n).'];  % 左下

    patch(P(:,1), P(:,2), bodyColor, 'EdgeColor', edgeColor, 'LineWidth', 1.0);

    function draw_cap(th_edge, Cc, sgn)
        % 半圆基向量由 th_edge 决定（与局部径向一致）
        Ncap = 64;
        a    = linspace(0, 360, Ncap); 
        n_e  = [cosd(th_edge);  sind(th_edge)];
        t_e  = [-sind(th_edge); cosd(th_edge)];
        cs   = cosd(a);  ss = sind(a);
        Pcap = Cc.' + (rcap*cs).'*n_e.' + (rcap*ss).'*t_e.';   % Ncap×2

        % 端盖颜色可与主体不同（如白色）
        patch(Pcap(:,1), Pcap(:,2), capColor, 'EdgeColor', edgeColor, 'LineWidth', 1.0);
    end
end

