function femm_draw_design_grid_inset(domain, theta_offset_deg, groupInset, inset_r_ratio, inset_th_ratio)
%FEMM_DRAW_DESIGN_GRID_INSET 在每个格子内部画一圈"内缩边界"（双层格子）
%
% inset_r_ratio  : 径向内缩比例（相对每个格子自身 dr = ratio*(r2-r1)）
% inset_th_ratio : 角向内缩比例（相对每个格子自身 dth = ratio*(th2-th1)）
%
% 建议：0.15~0.30，必须 <0.5，否则内核会翻转/消失

if nargin < 3 || isempty(groupInset)
    groupInset = 21;
end
if nargin < 4 || isempty(inset_r_ratio)
    inset_r_ratio = 0.20;
end
if nargin < 5 || isempty(inset_th_ratio)
    inset_th_ratio = 0.20;
end

r_edges     = domain.r_edges(:).';
theta_edges = (domain.theta_edges(:).' + theta_offset_deg);

nr = numel(r_edges) - 1;
nt = numel(theta_edges) - 1;

% 为安全起见：把比例夹到 (0,0.45)
inset_r_ratio  = max(0.0, min(0.45, inset_r_ratio));
inset_th_ratio = max(0.0, min(0.45, inset_th_ratio));

for it = 1:nt
    th1 = theta_edges(it);
    th2 = theta_edges(it+1);
    dth_cell = th2 - th1;

    % 每个格子角向内缩量（度）
    dth_in = inset_th_ratio * dth_cell;
    if dth_in*2 >= dth_cell
        dth_in = 0.45 * dth_cell;
    end

    th1i = th1 + dth_in;
    th2i = th2 - dth_in;
    dth_i = th2i - th1i;

    for ir = 1:nr
        r1 = r_edges(ir);
        r2 = r_edges(ir+1);
        dr_cell = r2 - r1;

        % 每个格子径向内缩量
        dr_in = inset_r_ratio * dr_cell;
        if dr_in*2 >= dr_cell
            dr_in = 0.45 * dr_cell;
        end

        r1i = r1 + dr_in;
        r2i = r2 - dr_in;

        if r2i <= r1i
            continue; % 太小就跳过
        end

        % ---- 4个角点（内核边界）----
        x11 = r1i*cosd(th1i);  y11 = r1i*sind(th1i);
        x12 = r2i*cosd(th1i);  y12 = r2i*sind(th1i);
        x21 = r2i*cosd(th2i);  y21 = r2i*sind(th2i);
        x22 = r1i*cosd(th2i);  y22 = r1i*sind(th2i);

        % 加节点（重复无所谓）
        mi_addnode(x11,y11); mi_addnode(x12,y12);
        mi_addnode(x21,y21); mi_addnode(x22,y22);

% ---- 两条"内缩径向边" ----
mi_addsegment(x11,y11, x12,y12);
mi_selectsegment((x11+x12)/2, (y11+y12)/2);
mi_setsegmentprop('', 0, 0, 0, groupInset);   % 这里最后一个才是 group
mi_clearselected();

mi_addsegment(x22,y22, x21,y21);
mi_selectsegment((x22+x21)/2, (y22+y21)/2);
mi_setsegmentprop('', 0, 0, 0, groupInset);
mi_clearselected();

% ---- 两条"内缩圆弧边" ----
mi_addarc(x11,y11, x22,y22, dth_i, 15);

% 选弧段：用弧中点坐标（最稳）
thm = (th1i + th2i)/2;
xm  = r1i*cosd(thm);
ym  = r1i*sind(thm);

mi_selectarcsegment(xm, ym);
mi_setarcsegmentprop(15, '', 0, groupInset);   % maxsegdeg=1只是示例，你可改大
mi_clearselected();


mi_addarc(x12,y12, x21,y21, dth_i, 15);

thm = (th1i + th2i)/2;
xm  = r2i*cosd(thm);
ym  = r2i*sind(thm);

mi_selectarcsegment(xm, ym);
mi_setarcsegmentprop(15, '', 0, groupInset);
mi_clearselected();
    end
end

end
