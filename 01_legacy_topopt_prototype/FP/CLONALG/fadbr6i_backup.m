function femm_apply_design_bits_rep6_inset_b( ...
    gene, domain, ctx, phase_id_sector, mats, circNames, ...
    groupIdCore, groupIdRing, groupIdCuGeom, turns_per_circ)

Nd = domain.Nd;

% 3进制材料：0 Air, 1 Iron, 2 Copper
mat_code = decode_material_bits(gene, Nd);  % Nd×1

% ===== 1) 删旧的 label（只删label）=====
mi_selectgroup(groupIdCore); mi_deleteselected();
mi_selectgroup(groupIdRing); mi_deleteselected();
mi_clearselected();

% ===== 2) 删旧的"动态小格子几何"（nodes+segments）=====
% 关键：每次迭代先清空上一代加进去的小格子，否则会叠加
mi_selectgroup(groupIdCuGeom);
mi_deleteselected();
mi_clearselected();

% ===== 基域（0~15°）两套点 =====
x0c = domain.x_core(:); y0c = domain.y_core(:);
x0r = domain.x_ring(:); y0r = domain.y_ring(:);

r0c  = hypot(x0c,y0c);   th0c = atan2d(y0c,x0c);
r0r  = hypot(x0r,y0r);   th0r = atan2d(y0r,x0r);

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
sh_r  = ctx.inset_r_ratio;   % e.g. 0.20
sh_th = ctx.inset_th_ratio;  % e.g. 0.20

% ===== 预先建立 k -> (ir,it) 的正确映射（由中心点落在哪个edge区间决定）=====
th0c_wrap = wrap_to_span(th0c, theta_edges(1), theta_edges(end));

% 处理刚好落在最右边界上的点，避免 discretize 返回 NaN
epsr  = 1e-9;
epsth = 1e-9;
r0c2  = r0c;  r0c2(r0c2 >= r_edges(end))       = r_edges(end)       - epsr;
th0c2 = th0c_wrap; th0c2(th0c2 >= theta_edges(end)) = theta_edges(end) - epsth;

ir_of_k = discretize(r0c2,  r_edges);      % Nd×1, 值域 1..nr
it_of_k = discretize(th0c2, theta_edges);  % Nd×1, 值域 1..nt

% 安全检查
if any(isnan(ir_of_k)) || any(isnan(it_of_k))
    error('k->(ir,it) mapping has NaN. Check r_edges/theta_edges span vs x_core centers.');
end

for s = 1:nSector
    pid = phase_id_sector{s};  % Nd×1

    Cs = c0 + (s-1)*th_span;
    isMirror = mod(s,2)==0;

    % core/ring label 的角度
    if ~isMirror
        th_c = Cs + d;
    else
        th_c = Cs - d;
    end
    th_r = th_c;

    % label 坐标
    xc = r0c .* cosd(th_c);  yc = r0c .* sind(th_c);
    xr = r0r .* cosd(th_r);  yr = r0r .* sind(th_r);

        % ===== 删除相同材料(仅空气/铁)之间的内部边界线 =====
    keyMerge = build_merge_keys(mat_code);
    delete_internal_boundaries_sector(r_edges, theta_edges, c0, Cs, isMirror, nr, nt, keyMerge, ir_of_k, it_of_k);

    for k = 1:Nd
        code = mat_code(k);

        % ========== 非铜：只保留"大格子一个材料"，不生成小格子 ==========
        if code == 0
            matBig = mats.air;  circBig = '';  Nturns = 0;
        elseif code == 1
            matBig = mats.iron; circBig = '';  Nturns = 0;
        else
            % ========== 铜：大格子外层强制为空气，小格子内部为铜 ==========
            matBig = mats.air;  circBig = '';  Nturns = 0;
        end

        % ---- 放"大格子"label（用 ring 点）----
        mi_addblocklabel(xr(k), yr(k));
        mi_selectlabel(xr(k), yr(k));
        mi_setblockprop(matBig, 1, 0, circBig, 0, groupIdRing, 0);
        mi_clearselected();

        % ---- 如果是铜：生成小格子边界 + 放铜label（用 core 点）----
        if code == 2
            phaseId = pid(k);
            if phaseId == 0
                circCu = ''; NturnsCu = 0;
            else
                circCu = circNames{phaseId};
                NturnsCu = turns_per_circ(phaseId);
            end

            % 1) 动态生成"小格子"边界（在该大格子内部收缩一圈）
add_inner_cell_boundary(k, s, isMirror, Cs, c0, ...
    nr, nt, r_edges, theta_edges, sh_r, sh_th, groupIdCuGeom, ir_of_k, it_of_k);

            % 2) 放铜label（小格子内部）
            mi_addblocklabel(xc(k), yc(k));
            mi_selectlabel(xc(k), yc(k));
            mi_setblockprop(mats.copper, 1, 0, circCu, 0, groupIdCore, NturnsCu);
            mi_clearselected();
        end
    end
end

end

% ============================================================
% 在第 k 个大格子里生成一个"收缩后的小格子边界"
% ============================================================
function add_inner_cell_boundary(k, s, isMirror, Cs, c0, ...
    nr, nt, r_edges, theta_edges, sh_r, sh_th, groupIdCuGeom, ir_of_k, it_of_k)

% 用"真实中心点落bin"的映射，而不是 ind2sub 猜顺序
ir = ir_of_k(k);
it = it_of_k(k);

% 大格子极坐标边界（基域）
r1  = r_edges(ir);
r2  = r_edges(ir+1);
th1 = theta_edges(it);
th2 = theta_edges(it+1);

% 收缩得到小格子边界
dr  = sh_r  * (r2 - r1);
dth = sh_th * (th2 - th1);
ri1 = r1 + dr;  ri2 = r2 - dr;
ti1 = th1 + dth; ti2 = th2 - dth;

% 映射到第 s 个扇区（按你原来的"以 c0 为镜像轴"的方式）
mapTh = @(th_base) (~isMirror) * (Cs + (th_base - c0)) + ...
                   ( isMirror) * (Cs - (th_base - c0));

T1 = mapTh(ti1);
T2 = mapTh(ti2);

% 四个角点（按极坐标->直角坐标）
p = zeros(4,2);
p(1,:) = [ri1*cosd(T1), ri1*sind(T1)];
p(2,:) = [ri2*cosd(T1), ri2*sind(T1)];
p(3,:) = [ri2*cosd(T2), ri2*sind(T2)];
p(4,:) = [ri1*cosd(T2), ri1*sind(T2)];

% 依次加 node（并把 node 放入 groupIdCuGeom）
for i = 1:4
    mi_addnode(p(i,1), p(i,2));
    mi_selectnode(p(i,1), p(i,2));
    mi_setnodeprop('', groupIdCuGeom);
    mi_clearselected();
end

% 加 4 条 segment 形成闭合
for i = 1:4
    j = i + 1; if j==5, j=1; end
    mi_addsegment(p(i,1), p(i,2), p(j,1), p(j,2));

    % 选中该段的中点设置 segment group
    mx = 0.5*(p(i,1)+p(j,1));
    my = 0.5*(p(i,2)+p(j,2));
    mi_selectsegment(mx, my);
    % mi_setsegmentprop(propname, elementsize, automesh, hide, group)
    mi_setsegmentprop('', 0, 1, 0, groupIdCuGeom);
    mi_clearselected();
end

end

function thw = wrap_to_span(th, th_start, th_end)
% 把角度 th（deg）包裹到 [th_start, th_end) 区间（跨度一般是 15°）
thw = th;
span = th_end - th_start;

% 先移到附近
thw = thw - th_start;
thw = mod(thw, 360);      % [0,360)
% 找到最接近 [0,span) 的那个等价角（因为 span 很小）
thw(thw >= span & thw < (360-span)) = thw(thw >= span & thw < (360-span)) - 360;
% 回到原起点
thw = thw + th_start;

% 最终确保在 [start, start+360) 内
end

% ======================================================================
% 生成"可合并"的 key：空气/铁允许合并，铜不合并（给唯一key）
% ======================================================================
function keyMerge = build_merge_keys(mat_code)
Nd = numel(mat_code);
keyMerge = zeros(Nd,1);
for k = 1:Nd
    if mat_code(k) == 0
        keyMerge(k) = 100; % air
    elseif mat_code(k) == 1
        keyMerge(k) = 200; % iron
    else
        keyMerge(k) = 1000 + k; % copper: unique key (no merge)
    end
end
end

% ======================================================================
% 删除本扇区内部边界线：相邻cell key 相同则删边界
% - 圆弧边界：r = r_edges(i+1)
% - 径向边界：theta = theta_edges(j+1)
% - 不动扇区左右边界，所以切向只遍历 j=1..nt-1
% ======================================================================
function delete_internal_boundaries_sector(r_edges, theta_edges, c0, Cs, isMirror, nr, nt, key, ir_of_k, it_of_k)

keyGrid = nan(nr, nt);
for k = 1:numel(key)
    keyGrid(ir_of_k(k), it_of_k(k)) = key(k);
end

% A) radial 邻居之间：删圆弧边界
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

% B) tangential 邻居之间：删径向线段边界（只删内部，不删扇区边界）
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
% theta 映射：基域角度 -> 第 s 扇区角度（与你原镜像规则一致）
% ======================================================================
function th_out = map_theta(th_in, c0, Cs, isMirror)
d = th_in - c0;
if ~isMirror
    th_out = Cs + d;
else
    th_out = Cs - d;
end
end


%%
function femm_apply_design_bits_rep6_inset( ...
    gene, domain, ctx, phase_id_sector, mats, circNames, ...
    groupIdCore, groupIdRing, groupIdCuGeom, turns_per_circ)

Nd = domain.Nd;

% 3进制材料：0 Air, 1 Iron, 2 Copper
mat_code = decode_material_bits(gene, Nd);  % Nd×1

% ===== 1) 删旧的 label（只删label）=====
mi_selectgroup(groupIdCore); mi_deleteselected();
mi_selectgroup(groupIdRing); mi_deleteselected();
mi_clearselected();

% ===== 2) 删旧的"动态小格子几何"（nodes+segments）=====
% 关键：每次迭代先清空上一代加进去的小格子，否则会叠加
mi_selectgroup(groupIdCuGeom);
mi_deleteselected();
mi_clearselected();

% ===== 基域（0~15°）两套点 =====
x0c = domain.x_core(:); y0c = domain.y_core(:);
x0r = domain.x_ring(:); y0r = domain.y_ring(:);

r0c  = hypot(x0c,y0c);   th0c = atan2d(y0c,x0c);
r0r  = hypot(x0r,y0r);   th0r = atan2d(y0r,x0r);

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
sh_r  = ctx.inset_r_ratio;   % e.g. 0.20
sh_th = ctx.inset_th_ratio;  % e.g. 0.20

% ===== 预先建立 k -> (ir,it) 的正确映射（由中心点落在哪个edge区间决定）=====
th0c_wrap = wrap_to_span(th0c, theta_edges(1), theta_edges(end));

% 处理刚好落在最右边界上的点，避免 discretize 返回 NaN
epsr  = 1e-9;
epsth = 1e-9;
r0c2  = r0c;  r0c2(r0c2 >= r_edges(end))       = r_edges(end)       - epsr;
th0c2 = th0c_wrap; th0c2(th0c2 >= theta_edges(end)) = theta_edges(end) - epsth;

ir_of_k = discretize(r0c2,  r_edges);      % Nd×1, 值域 1..nr
it_of_k = discretize(th0c2, theta_edges);  % Nd×1, 值域 1..nt

% 安全检查
if any(isnan(ir_of_k)) || any(isnan(it_of_k))
    error('k->(ir,it) mapping has NaN. Check r_edges/theta_edges span vs x_core centers.');
end

for s = 1:nSector
    pid = phase_id_sector{s};  % Nd×1

    Cs = c0 + (s-1)*th_span;
    isMirror = mod(s,2)==0;

    % core/ring label 的角度
    if ~isMirror
        th_c = Cs + d;
    else
        th_c = Cs - d;
    end
    th_r = th_c;

    % label 坐标
    xc = r0c .* cosd(th_c);  yc = r0c .* sind(th_c);
    xr = r0r .* cosd(th_r);  yr = r0r .* sind(th_r);

    % ===== 删除相同材料(空气/铁)之间的内部边界线 =====
    keyMergeRing = build_merge_keys_ring(mat_code);
    delete_internal_boundaries_sector(r_edges, theta_edges, c0, Cs, isMirror, nr, nt, keyMergeRing, ir_of_k, it_of_k);

    % ===== 合并后的 ring 区域只放一个材料标签（避免重复label）=====
    keyGridRing = build_key_grid(keyMergeRing, nr, nt, ir_of_k, it_of_k);
    kGrid = build_k_grid(nr, nt, ir_of_k, it_of_k);
    groupsRing = connected_components_grid(keyGridRing, nr, nt);
    for g = 1:numel(groupsRing)
        rep = groupsRing{g}(1);
        [ir, it] = ind2sub([nr, nt], rep);
        k = kGrid(ir, it);
        if isnan(k)
            continue;
        end
        if mat_code(k) == 1
            matRing = mats.iron;
        else
                 matRing = mats.air;
        end

        mi_addblocklabel(xr(rep), yr(rep));
        mi_selectlabel(xr(rep), yr(rep));
        mi_setblockprop(matRing, 1, 0, '', 0, groupIdRing, 0);
        mi_clearselected();
    end

    for k = 1:Nd
        code = mat_code(k);

        % ---- 如果是铜：生成小格子边界 + 放铜label（用 core 点）----
        if code == 2
            phaseId = pid(k);
            if phaseId == 0
                circCu = ''; NturnsCu = 0;
            else
                circCu = circNames{phaseId};
                NturnsCu = turns_per_circ(phaseId);
            end

            % 1) 动态生成"小格子"边界（在该大格子内部收缩一圈）
            add_inner_cell_boundary(k, s, isMirror, Cs, c0, ...
                nr, nt, r_edges, theta_edges, sh_r, sh_th, groupIdCuGeom, ir_of_k, it_of_k);

            % 2) 放铜label（小格子内部）
            mi_addblocklabel(xc(k), yc(k));
            mi_selectlabel(xc(k), yc(k));
            mi_setblockprop(mats.copper, 1, 0, circCu, 0, groupIdCore, NturnsCu);
            mi_clearselected();
        end
    end
end

end

% ============================================================
% 在第 k 个大格子里生成一个"收缩后的小格子边界"
% ============================================================
function add_inner_cell_boundary(k, s, isMirror, Cs, c0, ...
    nr, nt, r_edges, theta_edges, sh_r, sh_th, groupIdCuGeom, ir_of_k, it_of_k)

% 用"真实中心点落bin"的映射，而不是 ind2sub 猜顺序
ir = ir_of_k(k);
it = it_of_k(k);

% 大格子极坐标边界（基域）
r1  = r_edges(ir);
r2  = r_edges(ir+1);
th1 = theta_edges(it);
th2 = theta_edges(it+1);

% 收缩得到小格子边界
dr  = sh_r  * (r2 - r1);
dth = sh_th * (th2 - th1);
ri1 = r1 + dr;  ri2 = r2 - dr;
ti1 = th1 + dth; ti2 = th2 - dth;

% 映射到第 s 个扇区（按你原来的"以 c0 为镜像轴"的方式）
mapTh = @(th_base) (~isMirror) * (Cs + (th_base - c0)) + ...
    ( isMirror) * (Cs - (th_base - c0));

T1 = mapTh(ti1);
T2 = mapTh(ti2);

% 四个角点（按极坐标->直角坐标）
p = zeros(4,2);
p(1,:) = [ri1*cosd(T1), ri1*sind(T1)];
p(2,:) = [ri2*cosd(T1), ri2*sind(T1)];
p(3,:) = [ri2*cosd(T2), ri2*sind(T2)];
p(4,:) = [ri1*cosd(T2), ri1*sind(T2)];

% 依次加 node（并把 node 放入 groupIdCuGeom）
for i = 1:4
    mi_addnode(p(i,1), p(i,2));
    mi_selectnode(p(i,1), p(i,2));
    mi_setnodeprop('', groupIdCuGeom);
    mi_clearselected();
end

% 加 4 条 segment 形成闭合
for i = 1:4
    j = i + 1; if j==5, j=1; end
    mi_addsegment(p(i,1), p(i,2), p(j,1), p(j,2));

    % 选中该段的中点设置 segment group
    mx = 0.5*(p(i,1)+p(j,1));
    my = 0.5*(p(i,2)+p(j,2));
    mi_selectsegment(mx, my);
    % mi_setsegmentprop(propname, elementsize, automesh, hide, group)
    mi_setsegmentprop('', 0, 1, 0, groupIdCuGeom);
    mi_clearselected();
end

end

function thw = wrap_to_span(th, th_start, th_end)
% 把角度 th（deg）包裹到 [th_start, th_end) 区间（跨度一般是 15°）
thw = th;
span = th_end - th_start;

% 先移到附近
thw = thw - th_start;
thw = mod(thw, 360);      % [0,360)
% 找到最接近 [0,span) 的那个等价角（因为 span 很小）
thw(thw >= span & thw < (360-span)) = thw(thw >= span & thw < (360-span)) - 360;
% 回到原起点
thw = thw + th_start;

% 最终确保在 [start, start+360) 内
end

% ======================================================================
% ring 合并 key：空气/铁合并，铜也视为空气（外圈空气一起合并）
% ======================================================================
function keyMerge = build_merge_keys_ring(mat_code)
Nd = numel(mat_code);
keyMerge = zeros(Nd,1);
for k = 1:Nd
    if mat_code(k) == 1
        keyMerge(k) = 200; % iron
    else
        keyMerge(k) = 100; % air (include copper cells' outer air)
    end
end
end

% ======================================================================
% 删除本扇区内部边界线：相邻cell key 相同则删边界
% - 圆弧边界：r = r_edges(i+1)
% - 径向边界：theta = theta_edges(j+1)
% - 不动扇区左右边界，所以切向只遍历 j=1..nt-1
% ======================================================================
function delete_internal_boundaries_sector(r_edges, theta_edges, c0, Cs, isMirror, nr, nt, key, ir_of_k, it_of_k)

keyGrid = build_key_grid(key, nr, nt, ir_of_k, it_of_k);

% A) radial 邻居之间：删圆弧边界
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

% B) tangential 邻居之间：删径向线段边界（只删内部，不删扇区边界）
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
% key 向 (ir,it) 网格映射
% ======================================================================
function keyGrid = build_key_grid(key, nr, nt, ir_of_k, it_of_k)
keyGrid = nan(nr, nt);
for k = 1:numel(key)
    keyGrid(ir_of_k(k), it_of_k(k)) = key(k);
end
end

% ======================================================================
% k 向 (ir,it) 网格映射
% ======================================================================
function kGrid = build_k_grid(nr, nt, ir_of_k, it_of_k)
kGrid = nan(nr, nt);
for k = 1:numel(ir_of_k)
    kGrid(ir_of_k(k), it_of_k(k)) = k;
end
end

% ======================================================================
% 连通域（4邻域）分组：相同key且相邻则 union
% 返回：groups{g}=该连通域包含的cell索引（线性k索引）
% ======================================================================
function groups = connected_components_grid(keyGrid, nr, nt)
Nd = nr * nt;
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
        if keyGrid(i,j) == keyGrid(i+1,j)
            funion(base + i, base + i + 1);
        end
    end
end

% union tangential neighbors
for j = 1:(nt-1)
    base1 = (j-1)*nr;
    base2 = j*nr;
    for i = 1:nr
        if keyGrid(i,j) == keyGrid(i,j+1)
            funion(base1 + i, base2 + i);
        end
    end
end

roots = zeros(Nd,1);
for k = 1:Nd
    roots(k) = ffind(k);
end

tmp = accumarray(roots, (1:Nd)', [Nd,1], @(x){x}, {[]});
groups = tmp(~cellfun(@isempty,tmp));
end

% ======================================================================
% theta 映射：基域角度 -> 第 s 扇区角度（与你原镜像规则一致）
% ======================================================================
function th_out = map_theta(th_in, c0, Cs, isMirror)
d = th_in - c0;
if ~isMirror
    th_out = Cs + d;
else
    th_out = Cs - d;
end
end

