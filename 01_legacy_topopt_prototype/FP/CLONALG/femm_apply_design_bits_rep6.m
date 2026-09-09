function femm_apply_design_bits_rep6(gene, domain, phase_id_sector, mats, circNames, groupIdCore, groupIdRing, turns_per_circ)

Nd = domain.Nd;

% 3进制材料：0 Air, 1 Iron, 2 Copper
mat_code = decode_material_bits(gene, Nd);  % Nd×1

% 删掉旧 label（只删 label，不动网格线）
mi_selectgroup(groupIdCore); mi_deleteselected();
mi_selectgroup(groupIdRing); mi_deleteselected();
mi_clearselected();

% 基域（0~15°）两套点
x0c = domain.x_core(:); y0c = domain.y_core(:);
x0r = domain.x_ring(:); y0r = domain.y_ring(:);

r0c  = hypot(x0c,y0c);   th0c = atan2d(y0c,x0c);
r0r  = hypot(x0r,y0r);   th0r = atan2d(y0r,x0r);

% 用 core 的角度去定义中线和偏移（ring 用同一个偏移）
c0 = mean(th0c);
d  = th0c - c0;

theta_edges = domain.theta_edges;
th_span = theta_edges(end) - theta_edges(1);   % 15°
nSector = numel(phase_id_sector);              % 6

for s = 1:nSector
    pid = phase_id_sector{s};  % Nd×1

    Cs = c0 + (s-1)*th_span;

    if mod(s,2)==1
        th_c = Cs + d;      % core 角度
    else
        th_c = Cs - d;      % 镜像
    end

    % ring 角度跟 core 一样（同一条射线上）
    th_r = th_c;

    % 坐标
    xc = r0c .* cosd(th_c);  yc = r0c .* sind(th_c);
    xr = r0r .* cosd(th_r);  yr = r0r .* sind(th_r);

    for k = 1:Nd
        code = mat_code(k);

        % ========= core 区域材料（由基因决定） =========
        if code == 0
            matCore = mats.air;   circCore = '';  Nturns = 0;
        elseif code == 1
            matCore = mats.iron;  circCore = '';  Nturns = 0;
        else
            matCore = mats.copper;
            phaseId = pid(k);
            if phaseId==0
                circCore = ''; Nturns = 0;
            else
                circCore = circNames{phaseId};
                Nturns   = turns_per_circ(phaseId);
            end
        end

        % ========= ring 区域材料（隔离带逻辑） =========
        % 如果 core 是铜，则 ring 强制为空气；否则 ring 跟 core（或你也可以固定 ring=iron）
        if code == 2
            matRing = mats.air;
        else
            matRing = matCore;   % 跟随
        end
        circRing = '';  % ring 永远不挂电路

        % ---- 放 core label ----
        mi_addblocklabel(xc(k), yc(k));
        mi_selectlabel(xc(k), yc(k));
        mi_setblockprop(matCore, 1, 0, circCore, 0, groupIdCore, Nturns);
        mi_clearselected();

        % ---- 放 ring label ----
        mi_addblocklabel(xr(k), yr(k));
        mi_selectlabel(xr(k), yr(k));
        mi_setblockprop(matRing, 1, 0, circRing, 0, groupIdRing, 0);
        mi_clearselected();
    end
end
end
