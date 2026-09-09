function visualize_design_cells(domain, theta_offset_deg)
    % 用小 patch 把设计域/冻结域拼出来，蓝=设计，灰=冻结

    if nargin < 2
        theta_offset_deg = 0;
    end

    r_edges     = domain.r_edges;
    theta_edges = domain.theta_edges;
    M           = domain.design_mask;  % nt×nr

    [nt, nr] = size(M);

    figure; hold on; axis equal;
    colormap([0.8 0.8 0.8; 0.2 0.4 1]); % 1=灰,2=蓝
    caxis([1 2]);

    for it = 1:nt
        for ir = 1:nr
            r1  = r_edges(ir);
            r2  = r_edges(ir+1);
            th1 = theta_edges(it)   + theta_offset_deg;
            th2 = theta_edges(it+1) + theta_offset_deg;

            x = [r1*cosd(th1), r2*cosd(th1), ...
                 r2*cosd(th2), r1*cosd(th2)];
            y = [r1*sind(th1), r2*sind(th1), ...
                 r2*sind(th2), r1*sind(th2)];

            c = 1 + double(M(it,ir));  % 冻结=1, 设计=2
            patch(x, y, c, 'EdgeColor','none');
        end
    end

    title('设计单元拼图（蓝=design, 灰=冻结）');
    xlabel('x (mm)'); ylabel('y (mm)');
end