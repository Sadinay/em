function femm_debug_design_domain(domain, cfg, theta_offset_deg)

    mi_addmaterial('true', 1, 1, 0, 0, 0, 0);
    mi_addmaterial('TestFrozen', 1, 1, 0, 0, 0, 0);
    mi_addmaterial('false',   1, 1, 0, 0, 0, 0);  % 槽区调试用

    mat_design  = 'true';
    mat_frozen  = 'TestFrozen';
    mat_slot    = 'false';

    group_design = 100;
    group_frozen = 101;
    group_slot   = 102;

    r_edges     = domain.r_edges;
    theta_edges = domain.theta_edges;

    r_cent_1d     = r_edges(1:end-1)    + diff(r_edges)/2;
    theta_cent_1d = theta_edges(1:end-1) + diff(theta_edges)/2;

    [theta_c, r_c] = meshgrid(theta_cent_1d, r_cent_1d);
    theta_c = theta_c';  r_c = r_c';

    theta_global = theta_offset_deg + theta_c;
    x = r_c .* cosd(theta_global);
    y = r_c .* sind(theta_global);

    design_mask = domain.design_mask;
    slot_mask   = domain.slot_mask;
    [nt, nr] = size(design_mask);

    for it = 1:nt
        for ir = 1:nr
            xx = x(it,ir);
            yy = y(it,ir);

            if slot_mask(it,ir)
                % 槽区：画成 TestSlot，方便肉眼确认
                % mi_addblocklabel(xx, yy);
                % mi_selectlabel(xx, yy);
                % mi_setblockprop(mat_slot, 1, 0, '', 0, group_slot, 1);
                % mi_clearselected;

            elseif design_mask(it,ir)
                % 设计域
                mi_addblocklabel(xx, yy);
                mi_selectlabel(xx, yy);
                mi_setblockprop(mat_design, 1, 0, '', 0, group_design, 1);
                mi_clearselected;

            else
                % 非槽且非设计域 = 冻结铁心
                mi_addblocklabel(xx, yy);
                mi_selectlabel(xx, yy);
                mi_setblockprop(mat_frozen, 1, 0, '', 0, group_frozen, 1);
                mi_clearselected;
            end
        end
    end

    mi_zoomnatural;
end
