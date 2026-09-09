function visualize_design_domain(domain)
    r_edges    = domain.r_edges;
    theta_edges = domain.theta_edges;
    [theta_c, r_c] = meshgrid( ...
        theta_edges(1:end-1) + diff(theta_edges)/2, ...
        r_edges(1:end-1) + diff(r_edges)/2);
    theta_c = theta_c'; r_c = r_c';

    % 极坐标 -> 笛卡尔坐标（只为画图）
    x = r_c .* cosd(theta_c);
    y = r_c .* sind(theta_c);

    figure; hold on; axis equal;
    % 画设计域
    mask = domain.design_mask;
    scatter(x(mask), y(mask), 10, 'filled');  % 设计域
    % 画槽
    mask_slot = domain.slot_mask;
    scatter(x(mask_slot), y(mask_slot), 10, 'r'); % 槽区
    title('设计域(蓝) + 槽区(红)');
    xlabel('x (mm)'); ylabel('y (mm)');
end