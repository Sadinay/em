function domain = prepare_design_domain(domain, theta_offset_deg, inset_r_ratio, inset_th_ratio)
%PREPARE_DESIGN_DOMAIN
% 在已有 domain 上补充：
%   design_idx, Nd
%   x_c,y_c            : 原格子中心（可继续用于 debug）
%   x_core,y_core      : 内核(core)区域的 label 点（落在内缩小格子里）
%   x_ring,y_ring      : 外环(ring)区域的 label 点（落在外环带里）
%
% inset_r_ratio / inset_th_ratio：必须与画内缩边界时用的一致

    if nargin < 2 || isempty(theta_offset_deg)
        theta_offset_deg = 0;
    end
    if nargin < 3 || isempty(inset_r_ratio)
        inset_r_ratio = 0.20;
    end
    if nargin < 4 || isempty(inset_th_ratio)
        inset_th_ratio = 0.20;
    end

    mask = domain.design_mask;          % nt×nr
    design_idx = find(mask);           % 线性索引
    Nd = numel(design_idx);

    r_edges     = domain.r_edges(:).';
    theta_edges = domain.theta_edges(:).';   % 0..15
    nr = numel(r_edges)-1;
    nt = numel(theta_edges)-1;

    % 先算每个格子的中心（用于 debug）
    r_cent_1d     = r_edges(1:end-1)     + diff(r_edges)/2;
    theta_cent_1d = theta_edges(1:end-1) + diff(theta_edges)/2;

    [theta_centers, r_centers] = meshgrid(theta_cent_1d, r_cent_1d);
    theta_centers = theta_centers';   % nt×nr
    r_centers     = r_centers';

    r_c  = r_centers(design_idx);
    th_c = theta_centers(design_idx) + theta_offset_deg;
    x_c = r_c .* cosd(th_c);
    y_c = r_c .* sind(th_c);

    % --------- 关键：计算 core/ring 的 label 点（按每个格子的边界） ----------
    x_core = zeros(Nd,1); y_core = zeros(Nd,1);
    x_ring = zeros(Nd,1); y_ring = zeros(Nd,1);

    % 为了从 design_idx 反推 (it,ir)，用 ind2sub
    for k = 1:Nd
        [it, ir] = ind2sub([nt, nr], design_idx(k));

        r1 = r_edges(ir);
        r2 = r_edges(ir+1);
        th1 = theta_edges(it)   + theta_offset_deg;
        th2 = theta_edges(it+1) + theta_offset_deg;

        dr  = r2 - r1;
        dth = th2 - th1;

        % 内缩量（不要超过 0.45）
        dr_in  = min(0.45*dr,  inset_r_ratio  * dr);
        dth_in = min(0.45*dth, inset_th_ratio * dth);

        r1i = r1 + dr_in;
        r2i = r2 - dr_in;
        th1i = th1 + dth_in;
        th2i = th2 - dth_in;

        % 内核中心点（一定在内缩小格子里）
        r_core = 0.5*(r1i + r2i);
        th_mid = 0.5*(th1i + th2i);

        % 外环点：取"内核外边界 r2i 与外边界 r2 的中间"
        % 这样肯定落在 ring 区域里
        r_ring = 0.5*(r2i + r2);

        x_core(k) = r_core*cosd(th_mid);
        y_core(k) = r_core*sind(th_mid);

        x_ring(k) = r_ring*cosd(th_mid);
        y_ring(k) = r_ring*sind(th_mid);
    end

    % 写回 domain
    domain.design_idx = design_idx;
    domain.Nd         = Nd;

    domain.x_c  = x_c(:);
    domain.y_c  = y_c(:);

    domain.x_core = x_core;
    domain.y_core = y_core;
    domain.x_ring = x_ring;
    domain.y_ring = y_ring;
end
