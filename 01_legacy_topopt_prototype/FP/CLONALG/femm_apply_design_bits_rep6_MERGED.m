function femm_apply_design_bits_rep6_MERGED(gene, domain, phase_id_sector, mats, circNames, groupIdCore, groupIdRing, turns_per_circ)

Nd = domain.Nd;

% 3进制材料：0 Air, 1 Iron, 2 Copper
mat_code = decode_material_bits(gene, Nd);  % Nd×1

% 删掉旧 label（只删 label，不动网格线）
mi_selectgroup(groupIdCore); mi_deleteselected();
mi_selectgroup(groupIdRing); mi_deleteselected();
mi_clearselected();

% ========== 需要 domain.r_edges / domain.theta_edges 来定位内部边界 ==========
assert(isfield(domain,'theta_edges'), 'domain.theta_edges missing');
assert(isfield(domain,'r_edges') || isfield(domain,'r_edges_core'), 'domain.r_edges (or r_edges_core) missing');

theta_edges = domain.theta_edges(:);
[r_edges_core, r_edges_ring] = get_redges_core_ring(domain);

nr = numel(r_edges_core) - 1;
nt = numel(theta_edges)  - 1;
assert(Nd == nr*nt, 'Nd != (numel(r_edges)-1)*(numel(theta_edges)-1), check domain indexing.');

% 基域（0~15°）两套点
x0c = domain.x_core(:); y0c = domain.y_core(:);
x0r = domain.x_ring(:); y0r = domain.y_ring(:);

r0c  = hypot(x0c,y0c);   th0c = atan2d(y0c,x0c);
r0r  = hypot(x0r,y0r);   th0r = atan2d(y0r,x0r);

% 用 core 的角度去定义中线和偏移（ring 用同一个偏移）
c0 = mean(th0c);
d  = th0c - c0;

th_span = theta_edges(end) - theta_edges(1);   % 15°
nSector = numel(phase_id_sector);              % 6

for s = 1:nSector
    pid = phase_id_sector{s};  % Nd×1
    Cs  = c0 + (s-1)*th_span;

    isOdd = mod(s,2)==1;
    if isOdd
        th_c = Cs + d;      % core 角度
    else
        th_c = Cs - d;      % 镜像
    end
    th_r = th_c;

    % 坐标（每个cell中心点）
    xc = r0c .* cosd(th_c);  yc = r0c .* sind(th_c);
    xr = r0r .* cosd(th_r);  yr = r0r .* sind(th_r);

    % ===== 1) 为本扇区计算"块属性key"（相同key才合并）=====
    % core: material + (circuit, turns)
    [keyC, matNameC, circC, turnsC] = build_core_keys(mat_code, pid, mats, circNames, turns_per_circ);

    % ring: core=铜=>air, 其他跟随但不挂电路
    [keyR, matNameR] = build_ring_keys(mat_code, matNameC, mats);

    % ===== 2) 真正删除"内部边界线"（只删本扇区内部，不动扇区边界）=====
    delete_internal_boundaries_sector(r_edges_core, theta_edges, c0, Cs, isOdd, nr, nt, keyC);
    delete_internal_boundaries_sector(r_edges_ring, theta_edges, c0, Cs, isOdd, nr, nt, keyR);

    % ===== 3) 连通域合并后：每个连通域只放一个 label，并且铜区 turns 进行累加 =====
    % --- core ---
    [rootsC, groupsC] = connected_components(keyC, nr, nt);
    for g = 1:numel(groupsC)
        idx = groupsC{g};
        rep = idx(1);  % 代表点用第一个cell中心，保证一定在域内

        matCore  = matNameC{rep};
        circCore = circC{rep};

        % turns：同一连通域内累加（对铜区很关键，否则安匝数会被你合并掉）
        Nturns_total = sum(turnsC(idx));

        mi_addblocklabel(xc(rep), yc(rep));
        mi_selectlabel(xc(rep), yc(rep));
        mi_setblockprop(matCore, 1, 0, circCore, 0, groupIdCore, Nturns_total);
        mi_clearselected();
    end

    % --- ring ---
    [rootsR, groupsR] = connected_components(keyR, nr, nt);
    for g = 1:numel(groupsR)
        idx = groupsR{g};
        rep = idx(1);

        matRing = matNameR{rep};
        mi_addblocklabel(xr(rep), yr(rep));
        mi_selectlabel(xr(rep), yr(rep));
        mi_setblockprop(matRing, 1, 0, '', 0, groupIdRing, 0);
        mi_clearselected();
    end
end

end

% ======================================================================
% 子函数：core key（材料 + 电路 + 匝数）
% ======================================================================
function [keyC, matNameC, circC, turnsC] = build_core_keys(mat_code, pid, mats, circNames, turns_per_circ)

Nd = numel(mat_code);
keyC     = zeros(Nd,1);
turnsC   = zeros(Nd,1);
matNameC = cell(Nd,1);
circC    = cell(Nd,1);

for k = 1:Nd
    code = mat_code(k);

    if code == 0
        matNameC{k} = mats.air;
        circC{k}    = '';
        turnsC(k)   = 0;
        keyC(k)     = 100;   % air
    elseif code == 1
        matNameC{k} = mats.iron;
        circC{k}    = '';
        turnsC(k)   = 0;
        keyC(k)     = 200;   % iron
    else
        matNameC{k} = mats.copper;
        ph = pid(k);
        if ph == 0
            circC{k}  = '';
            turnsC(k) = 0;
            keyC(k)   = 300; % copper but no circuit (单独一类，避免和带电路铜合并)
        else
            circC{k}  = circNames{ph};
            turnsC(k) = turns_per_circ(ph);

            % key 必须把电路+匝数编码进去（相同才允许合并）
            % turns 可能为负，所以做一个偏移保证唯一性
            keyC(k)   = 1000000 + ph*10000 + (turnsC(k) + 5000);
        end
    end
end
end

% ======================================================================
% 子函数：ring key（铜=>空气，其它跟随，但不挂电路/匝数）
% ======================================================================
function [keyR, matNameR] = build_ring_keys(mat_code, matNameC, mats)

Nd = numel(mat_code);
keyR     = zeros(Nd,1);
matNameR = cell(Nd,1);

for k = 1:Nd
    if mat_code(k) == 2
        matNameR{k} = mats.air;
        keyR(k)     = 100; % air
    else
        % 跟随 core 的材料，但 ring 永远不挂电路
        matNameR{k} = matNameC{k};
        if strcmp(matNameR{k}, mats.air)
            keyR(k) = 100;
        else
            keyR(k) = 200; % iron（理论上这里不会出现 copper，因为 core=copper 时 mat_code=2 已处理）
        end
    end
end
end

% ======================================================================
% 删除本扇区内部边界线：相邻cell key 相同则删边界
% - 圆弧边界：r = r_edges(i+1)，用 mi_selectarcsegment 优先，失败再试 segment
% - 径向边界：theta = theta_edges(j+1)，用 mi_selectsegment
% - 不动扇区左右边界，所以切向只遍历 j=1..nt-1
% ======================================================================
function delete_internal_boundaries_sector(r_edges, theta_edges, c0, Cs, isOdd, nr, nt, key)

% A) radial 邻居之间：删圆弧边界
for j = 1:nt
    th_mid_base = 0.5*(theta_edges(j) + theta_edges(j+1));
    th_mid = map_theta(th_mid_base, c0, Cs, isOdd);

    for i = 1:(nr-1)
        k1 = (j-1)*nr + i;
        k2 = k1 + 1;
        if key(k1) == key(k2)
            r = r_edges(i+1);
            x = r*cosd(th_mid); y = r*sind(th_mid);

            % 先当作 arc 删
            try
                mi_clearselected();
                mi_selectarcsegment(x,y);
                mi_deleteselected();
                mi_clearselected();
            catch
                % 再当作 segment 删（有人用折线拟合弧）
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

% B) tangential 邻居之间：删径向线段边界（只删内部，不删扇区边界）
for j = 1:(nt-1)
    th_base = theta_edges(j+1);
    th = map_theta(th_base, c0, Cs, isOdd);

    for i = 1:nr
        k1 = (j-1)*nr + i;
        k2 = j*nr + i;
        if key(k1) == key(k2)
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
% 连通域（4邻域）分组：相同key且相邻则 union
% 返回：roots(每个cell的root)，groups{g}=该连通域包含的cell索引
% ======================================================================
function [roots, groups] = connected_components(key, nr, nt)

Nd = numel(key);
parent = (1:Nd)';

    function r = ffind(a)
        r = a;
        while parent(r) ~= r
            parent(r) = parent(parent(r));
            r = parent(r);
        end
    end

    function funion(a,b)
        ra = ffind(a);
        rb = ffind(b);
        if ra ~= rb
            parent(rb) = ra;
        end
    end

% union radial neighbors
for j = 1:nt
    base = (j-1)*nr;
    for i = 1:(nr-1)
        k1 = base + i;
        k2 = k1 + 1;
        if key(k1) == key(k2)
            funion(k1,k2);
        end
    end
end

% union tangential neighbors
for j = 1:(nt-1)
    base1 = (j-1)*nr;
    base2 = j*nr;
    for i = 1:nr
        k1 = base1 + i;
        k2 = base2 + i;
        if key(k1) == key(k2)
            funion(k1,k2);
        end
    end
end

roots = zeros(Nd,1);
for k = 1:Nd
    roots(k) = ffind(k);
end

% 分组（roots 值在 1..Nd 内，可用 accumarray）
tmp = accumarray(roots, (1:Nd)', [Nd,1], @(x){x}, {[]});
groups = tmp(~cellfun(@isempty,tmp));
end

% ======================================================================
% theta 映射：基域角度 -> 第 s 扇区角度（与你原镜像规则一致）
% ======================================================================
function th_out = map_theta(th_in, c0, Cs, isOdd)
d = th_in - c0;
if isOdd
    th_out = Cs + d;
else
    th_out = Cs - d;
end
end

% ======================================================================
% core/ring r_edges
% ======================================================================
function [r_edges_core, r_edges_ring] = get_redges_core_ring(domain)

if isfield(domain,'r_edges_core')
    r_edges_core = domain.r_edges_core(:);
elseif isfield(domain,'r_edges')
    r_edges_core = domain.r_edges(:);
else
    error('domain missing r_edges (or r_edges_core)');
end

if isfield(domain,'r_edges_ring')
    r_edges_ring = domain.r_edges_ring(:);
elseif isfield(domain,'r_edges')
    r_edges_ring = domain.r_edges(:);
else
    r_edges_ring = r_edges_core;
end

end
