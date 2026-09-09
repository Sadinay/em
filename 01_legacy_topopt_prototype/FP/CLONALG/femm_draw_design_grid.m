function femm_draw_design_grid(domain, theta_offset_deg, groupId)
%FEMM_DRAW_DESIGN_GRID 在 FEMM 中画出设计域的极坐标网格
%
%  domain.r_edges      - 径向边界 (1×(nr+1))
%  domain.theta_edges  - 角向边界 (1×(nt+1)) [0~theta_span_deg]
%  theta_offset_deg    - 该扇区在 FEMM 里起始角度（例如 0, 15, 30...）
%  groupId             - 分组号（方便以后整体隐藏/删除），可随便给一个整数

if nargin < 3
    groupId = 20;  % 默认把网格线放到 group 20
end

r_edges     = domain.r_edges;
theta_edges = domain.theta_edges + theta_offset_deg;  % 平移到 FEMM 全局角度里

nr = numel(r_edges)     - 1;   % 径向格子数
nt = numel(theta_edges) - 1;   % 角向格子数

% ---- 1) 先在所有交点处加节点 ----
for it = 1:(nt+1)             % 角向索引，对应 theta_edges(it)
    th = theta_edges(it);
    c  = cosd(th);
    s  = sind(th);
    for ir = 1:(nr+1)         % 径向索引，对应 r_edges(ir)
        r = r_edges(ir);
        x = r * c;
        y = r * s;
        mi_addnode(x, y);     %#ok<*MINV>  % 同坐标重复调用没关系，FEMM 会合并
    end
end

% ---- 2) 画径向线（每条从 r_inner 到 r_outer）----
for it = 1:(nt+1)
    th = theta_edges(it);
    c  = cosd(th);
    s  = sind(th);

    for ir = 1:nr
        r1 = r_edges(ir);
        r2 = r_edges(ir+1);

        x1 = r1 * c;  y1 = r1 * s;
        x2 = r2 * c;  y2 = r2 * s;

        mi_addsegment(x1, y1, x2, y2);

        % 关键：选中刚画的线段（用中点坐标）
        mi_selectsegment((x1+x2)/2, (y1+y2)/2);

        % setsegmentprop(boundary, automesh, meshsize, hide, group)
        mi_setsegmentprop('', 0, 0, 0, groupId);

        mi_clearselected();
    end
end

% 这里我们用 mi_addarc，以 r_edges(ir) 为半径，在相邻两条射线之间画弧
% ---- 3) 画圆弧线（每条对应一个 r_edge）----
maxSegDegArc = 10;   % ✅ 优化阶段建议 10~20，想更粗就再加

% for ir = 1:(nr+1)
%     r = r_edges(ir);
% 
%     for it = 1:nt
%         th1 = theta_edges(it);
%         th2 = theta_edges(it+1);
% 
%         x1 = r * cosd(th1);  y1 = r * sind(th1);
%         x2 = r * cosd(th2);  y2 = r * sind(th2);
% 
%         dth = th2 - th1;
% 
%         % 先画弧：最后一个参数是 maxsegdeg（别用 1！）
%         mi_addarc(x1, y1, x2, y2, dth, maxSegDegArc);
% 
%         % 选中弧段：用弧中点坐标最稳
%         thm = (th1 + th2)/2;
%         xm  = r * cosd(thm);
%         ym  = r * sind(thm);
%         mi_selectarcsegment(xm, ym);
% 
%         % setarcsegmentprop(maxsegdeg, boundary, hide, group)
%         mi_setarcsegmentprop(maxSegDegArc, '', 0, groupId);
% 
%         mi_clearselected();
%     end
% end


end
