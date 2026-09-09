function femm_apply_design_bits_rep6_inset_merge_cu( ...
    gene, domain, ctx, phase_id_sector, mats, circNames, ...
    groupIdCore, groupIdRing, groupIdCuGeom, turns_per_circ)

Nd = domain.Nd;

% 3进制材料：0 Air, 1 Iron, 2 Copper
mat_code = decode_material_bits(gene, Nd);  % Nd×1

% ===== 1) 删旧的 label（只删label）=====
mi_selectgroup(groupIdCore); mi_deleteselected();
mi_selectgroup(groupIdRing); mi_deleteselected();
mi_clearselected();

% ===== 2) 删旧的动态铜几何（本版本不再生成） =====
mi_selectgroup(groupIdCuGeom);
mi_deleteselected();
mi_clearselected();

% ===== 基域（0~15°）两套点 =====
x0c = domain.x_core(:); y0c = domain.y_core(:);
x0r = domain.x_ring(:); y0r = domain.y_ring(:);

r0c  = hypot(x0c,y0c);   th0c = atan2d(y0c,x0c);
r0r  = hypot(x0r,y0r);

% 用 core 的角度去定义中线和偏移（ring 用同一个偏移）
c0 = mean(th0c);
d  = th0c - c0;

theta_edges = domain.theta_edges(:);
th_span = theta_edges(end) - theta_edges(1);   % 15°
nSector = numel(phase_id_sector);              % 6

% 网格索引：k -> (ir,it)
nr = ctx.cfg.nr; nt = ctx.cfg.nt;
assert(Nd == nr*nt, 'Nd != nr*nt');

r_edges = domain.r_edges(:);

% ===== 建立 k -> (ir,it) 映射 =====
th0c_wrap = wrap_to_span(th0c, theta_edges(1), theta_edges(end));

epsr  = 1e-9;
epsth = 1e-9;
r0c2  = r0c;
r0c2(r0c2 >= r_edges(end)) = r_edges(end) - epsr;
th0c2 = th0c_wrap;
th0c2(th0c2 >= theta_edges(end)) = theta_edges(end) - epsth;

ir_of_k = discretize(r0c2,  r_edges);      % Nd×1, 1..nr
it_of_k = discretize(th0c2, theta_edges);  % Nd×1, 1..nt

if any(isnan(ir_of_k)) || any(isnan(it_of_k))
    error('k->(ir,it) mapping has NaN. Check r_edges/theta_edges span vs x_core centers.');
end

% 空气/铁机制保持不变；新增：铜也按相邻关系并区
keyMerge = build_merge_keys_air_iron_copper(mat_code);
keep_big_label = keep_one_label_per_region(nr, nt, keyMerge, ir_of_k, it_of_k);
comp_size = component_size_per_cell(nr, nt, keyMerge, ir_of_k, it_of_k);

for s = 1:nSector
    pid = phase_id_sector{s};  % Nd×1

    Cs = c0 + (s-1)*th_span;
    isMirror = mod(s,2)==0;

    if ~isMirror
        th_c = Cs + d;
    else
        th_c = Cs - d;
    end
    th_r = th_c;

    % label 坐标
    xc = r0c .* cosd(th_c);  yc = r0c .* sind(th_c);
    xr = r0r .* cosd(th_r);  yr = r0r .* sind(th_r);

    % ===== 删除相同材料（空气/铁/铜）之间的内部边界 =====
    delete_internal_boundaries_sector(r_edges, theta_edges, c0, Cs, isMirror, nr, nt, keyMerge, ir_of_k, it_of_k);
    remove_isolated_nodes_sector(r_edges, theta_edges, c0, Cs, isMirror, nr, nt, keyMerge, ir_of_k, it_of_k);

    % ===== 每个连通区域只保留一个 label =====
    for k = 1:Nd
        if ~keep_big_label(k)
            continue;
        end

        code = mat_code(k);
        if code == 0
            matName = mats.air;
            circName = '';
            turns = 0;
            xlbl = xr(k); ylbl = yr(k);
            grp = groupIdRing;
        elseif code == 1
            matName = mats.iron;
            circName = '';
            turns = 0;
            xlbl = xr(k); ylbl = yr(k);
            grp = groupIdRing;
        else
            matName = mats.copper;
            phaseId = pid(k);
            if phaseId == 0
                circName = '';
                turns = 0;
            else
                circName = circNames{phaseId};
                turns = comp_size(k) * turns_per_circ(phaseId);
            end
            xlbl = xc(k); ylbl = yc(k);
            grp = groupIdCore;
        end

        mi_addblocklabel(xlbl, ylbl);
        mi_selectlabel(xlbl, ylbl);
        mi_setblockprop(matName, 1, 0, circName, 0, grp, turns);
        mi_clearselected();
    end
end

end

function thw = wrap_to_span(th, th_start, th_end)
thw = th;
span = th_end - th_start;
thw = thw - th_start;
thw = mod(thw, 360);
thw(thw >= span & thw < (360-span)) = thw(thw >= span & thw < (360-span)) - 360;
thw = thw + th_start;
end

% ======================================================================
% 生成可合并 key：空气/铁保持原机制 + 铜也合并
% ======================================================================
function keyMerge = build_merge_keys_air_iron_copper(mat_code)
Nd = numel(mat_code);
keyMerge = zeros(Nd,1);
for k = 1:Nd
    if mat_code(k) == 0
        keyMerge(k) = 100; % air
    elseif mat_code(k) == 1
        keyMerge(k) = 200; % iron
    else
        keyMerge(k) = 300; % copper
    end
end
end

% ======================================================================
% 返回每个 cell 所在连通区的大小（同 key 的4邻域连通）
% ======================================================================
function comp_size = component_size_per_cell(nr, nt, key, ir_of_k, it_of_k)
keyGrid = nan(nr, nt);
kGrid = nan(nr, nt);
for k = 1:numel(key)
    keyGrid(ir_of_k(k), it_of_k(k)) = key(k);
    kGrid(ir_of_k(k), it_of_k(k)) = k;
end

comp_size = ones(numel(key),1);
visited = false(nr, nt);

for i = 1:nr
    for j = 1:nt
        if visited(i,j) || isnan(keyGrid(i,j))
            continue;
        end

        targetKey = keyGrid(i,j);
        stack = [i, j];
        visited(i,j) = true;
        comp_cells = zeros(0,2);

        while ~isempty(stack)
            ci = stack(end,1);
            cj = stack(end,2);
            stack(end,:) = [];
            comp_cells = [comp_cells; ci, cj]; %#ok<AGROW>

            neighbors = [ci-1, cj; ci+1, cj; ci, cj-1; ci, cj+1];
            for n = 1:4
                ni = neighbors(n,1);
                nj = neighbors(n,2);
                if ni < 1 || ni > nr || nj < 1 || nj > nt
                    continue;
                end
                if visited(ni,nj)
                    continue;
                end
                if keyGrid(ni,nj) == targetKey
                    visited(ni,nj) = true;
                    stack = [stack; ni, nj]; %#ok<AGROW>
                end
            end
        end

        sz = size(comp_cells,1);
        for t = 1:sz
            kk = kGrid(comp_cells(t,1), comp_cells(t,2));
            if ~isnan(kk)
                comp_size(kk) = sz;
            end
        end
    end
end
end

% ======================================================================
% 每个连通区域只保留一个 label
% ======================================================================
function keep_label = keep_one_label_per_region(nr, nt, key, ir_of_k, it_of_k)
keyGrid = nan(nr, nt);
kGrid = nan(nr, nt);
for k = 1:numel(key)
    keyGrid(ir_of_k(k), it_of_k(k)) = key(k);
    kGrid(ir_of_k(k), it_of_k(k)) = k;
end

keep_label = false(numel(key), 1);
visited = false(nr, nt);

for i = 1:nr
    for j = 1:nt
        if visited(i,j)
            continue;
        end
        if isnan(keyGrid(i,j))
            visited(i,j) = true;
            continue;
        end

        targetKey = keyGrid(i,j);
        stack = [i, j];
        visited(i,j) = true;
        rep_k = kGrid(i,j);

        while ~isempty(stack)
            ci = stack(end,1);
            cj = stack(end,2);
            stack(end,:) = [];

            neighbors = [ci-1, cj; ci+1, cj; ci, cj-1; ci, cj+1];
            for n = 1:4
                ni = neighbors(n,1);
                nj = neighbors(n,2);
                if ni < 1 || ni > nr || nj < 1 || nj > nt
                    continue;
                end
                if visited(ni,nj)
                    continue;
                end
                if keyGrid(ni,nj) == targetKey
                    visited(ni,nj) = true;
                    stack = [stack; ni, nj]; %#ok<AGROW>
                end
            end
        end

        if ~isnan(rep_k)
            keep_label(rep_k) = true;
        end
    end
end
end

% ======================================================================
% 删本扇区内部边界：相邻 cell key 相同则删边界
% ======================================================================
function delete_internal_boundaries_sector(r_edges, theta_edges, c0, Cs, isMirror, nr, nt, key, ir_of_k, it_of_k)

keyGrid = nan(nr, nt);
for k = 1:numel(key)
    keyGrid(ir_of_k(k), it_of_k(k)) = key(k);
end

% A) radial 邻居：删圆弧边界
for j = 1:nt
    th_mid_base = 0.5*(theta_edges(j) + theta_edges(j+1));
    th_mid = map_theta(th_mid_base, c0, Cs, isMirror);

    for i = 1:(nr-1)
        if keyGrid(i,j) == keyGrid(i+1,j)
            r = r_edges(i+1);
            x = r*cosd(th_mid); y = r*sind(th_mid);

            try
                mi_clearselected();
                mi_selectarcsegment(x,y);
                mi_deleteselected();
                mi_clearselected();
            catch
                try
                    mi_clearselected();
                    mi_selectsegment(x,y);
                    mi_deleteselected();
                    mi_clearselected();
                catch
                    mi_clearselected();
                end
            end
        end
    end
end

% B) tangential 邻居：删径向边界（不删扇区左右边界）
for j = 1:(nt-1)
    th_base = theta_edges(j+1);
    th = map_theta(th_base, c0, Cs, isMirror);

    for i = 1:nr
        if keyGrid(i,j) == keyGrid(i,j+1)
            rmid = 0.5*(r_edges(i) + r_edges(i+1));
            x = rmid*cosd(th); y = rmid*sind(th);
            try
                mi_clearselected();
                mi_selectsegment(x,y);
                mi_deleteselected();
                mi_clearselected();
            catch
                mi_clearselected();
            end
        end
    end
end
end

% ======================================================================
% 删除并区后孤立节点
% ======================================================================
function remove_isolated_nodes_sector(r_edges, theta_edges, c0, Cs, isMirror, nr, nt, key, ir_of_k, it_of_k)

keyGrid = nan(nr, nt);
for k = 1:numel(key)
    keyGrid(ir_of_k(k), it_of_k(k)) = key(k);
end

for i = 2:nr
    r = r_edges(i);
    for j = 2:nt
        k11 = keyGrid(i-1, j-1);
        k21 = keyGrid(i,   j-1);
        k12 = keyGrid(i-1, j);
        k22 = keyGrid(i,   j);

        if isnan(k11) || isnan(k21) || isnan(k12) || isnan(k22)
            continue;
        end
        if ~(k11 == k21 && k11 == k12 && k11 == k22)
            continue;
        end

        th_base = theta_edges(j);
        th = map_theta(th_base, c0, Cs, isMirror);
        x = r*cosd(th); y = r*sind(th);

        try
            mi_clearselected();
            mi_selectnode(x, y);
            mi_deleteselected();
            mi_clearselected();
        catch
            mi_clearselected();
        end
    end
end
end

function th_out = map_theta(th_in, c0, Cs, isMirror)
d = th_in - c0;
if ~isMirror
    th_out = Cs + d;
else
    th_out = Cs - d;
end
end
