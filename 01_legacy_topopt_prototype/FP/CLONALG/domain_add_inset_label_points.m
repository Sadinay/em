function domain = domain_add_inset_label_points(domain, theta_offset_deg, inset_r_ratio, inset_th_ratio)
% 计算每个格子的 core/ring label 点（全网格，nt×nr）

r_edges     = domain.r_edges(:).';
theta_edges = (domain.theta_edges(:).' + theta_offset_deg);

nr = numel(r_edges)-1;
nt = numel(theta_edges)-1;

x_core = zeros(nt,nr); y_core = zeros(nt,nr);
x_ring = zeros(nt,nr); y_ring = zeros(nt,nr);

for it = 1:nt
    th1 = theta_edges(it); th2 = theta_edges(it+1);
    dth = th2-th1;
    dth_in = min(0.45*dth, inset_th_ratio*dth);
    th1i = th1 + dth_in; th2i = th2 - dth_in;
    thm  = 0.5*(th1i+th2i); % 内核中心角（基本就是格子中心角）

    for ir = 1:nr
        r1 = r_edges(ir); r2 = r_edges(ir+1);
        dr = r2-r1;
        dr_in = min(0.45*dr, inset_r_ratio*dr);
        r1i = r1 + dr_in; r2i = r2 - dr_in;

        r_core = 0.5*(r1i+r2i);
        r_ring = 0.5*(r2i+r2);   % 取靠外的一点，一定落在外环带里

        x_core(it,ir) = r_core*cosd(thm);
        y_core(it,ir) = r_core*sind(thm);

        x_ring(it,ir) = r_ring*cosd(thm);
        y_ring(it,ir) = r_ring*sind(thm);
    end
end

domain.x_core_grid = x_core;
domain.y_core_grid = y_core;
domain.x_ring_grid = x_ring;
domain.y_ring_grid = y_ring;
end