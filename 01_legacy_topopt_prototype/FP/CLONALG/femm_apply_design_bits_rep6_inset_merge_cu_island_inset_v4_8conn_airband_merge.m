function femm_apply_design_bits_rep6_inset_merge_cu_island_inset_v4_8conn_airband_merge( ...
    gene, domain, ctx, phase_id_sector, mats, circNames, ...
    groupIdCore, groupIdRing, groupIdCuGeom, turns_per_circ)

% v3_8conn 目标（在 v2 基础上）：
% 1) 铜岛外边界做小幅内缩（默认 0.2）形成空气带
% 2) 每个铜格子保留独立铜标签/匝数（不再按铜连通岛合并铜标签）
% 3) 背景空气/铁标签不落在铜格子上，避免铜岛内部出现空气标签
% 4) 引入 8 邻接检测；并修正边覆盖判定，避免"正邻已是铜"时两格之间仍生成空气带

Nd = domain.Nd;
mat_code = decode_material_bits(gene, Nd);  % 0 air, 1 iron, 2 copper

alpha = 0.2;
if isfield(ctx, 'cu_island_inset_ratio')
    alpha = ctx.cu_island_inset_ratio;
elseif isfield(ctx, 'cu_inset_ratio')
    alpha = ctx.cu_inset_ratio;
elseif isfield(ctx, 'inset_r_ratio')
    alpha = ctx.inset_r_ratio;
end
alpha = max(0, min(alpha, 0.45));

% ===== 1) 删旧 label =====
mi_selectgroup(groupIdCore); mi_deleteselected();
mi_selectgroup(groupIdRing); mi_deleteselected();
mi_clearselected();

% ===== 2) 删旧动态铜几何 =====
mi_selectgroup(groupIdCuGeom);
mi_deleteselected();
mi_clearselected();

% ===== 基域点 =====
x0c = domain.x_core(:); y0c = domain.y_core(:);
x0r = domain.x_ring(:); y0r = domain.y_ring(:);
r0c = hypot(x0c,y0c); th0c = atan2d(y0c,x0c);
r0r = hypot(x0r,y0r);

c0 = mean(th0c);
d  = th0c - c0;

theta_edges = domain.theta_edges(:);
th_span = theta_edges(end) - theta_edges(1);
nSector = numel(phase_id_sector);

nr = ctx.cfg.nr; nt = ctx.cfg.nt;
assert(Nd == nr*nt, 'Nd != nr*nt');
r_edges = domain.r_edges(:);

% ===== k -> (ir,it) =====
th0c_wrap = wrap_to_span(th0c, theta_edges(1), theta_edges(end));
epsr = 1e-9; epsth = 1e-9;
r0c2 = r0c; r0c2(r0c2 >= r_edges(end)) = r_edges(end) - epsr;
th0c2 = th0c_wrap; th0c2(th0c2 >= theta_edges(end)) = theta_edges(end) - epsth;
ir_of_k = discretize(r0c2, r_edges);
it_of_k = discretize(th0c2, theta_edges);
if any(isnan(ir_of_k)) || any(isnan(it_of_k))
    error('k->(ir,it) mapping has NaN.');
end

% ===== 背景网格（空气/铁） =====
% 注意：铜格子在背景并区中被 mask 掉（key=NaN），避免 air label 掉进铜岛内部
key_bg = build_merge_keys_air_iron_mask_copper(mat_code);
keep_bg_label = keep_one_label_per_region(nr, nt, key_bg, ir_of_k, it_of_k);

% 铜布尔网格（用于判断外边界）
is_copper_grid = false(nr, nt);
for k = 1:Nd
    if mat_code(k) == 2
        is_copper_grid(ir_of_k(k), it_of_k(k)) = true;
    end
end
[copper_island_id, num_copper_islands] = build_copper_island_id_8conn(is_copper_grid);

for s = 1:nSector
    pid = phase_id_sector{s};

    Cs = c0 + (s-1)*th_span;
    isMirror = mod(s,2)==0;

    if ~isMirror
        th_c = Cs + d;
    else
        th_c = Cs - d;
    end
    th_r = th_c;

    xc = r0c .* cosd(th_c); yc = r0c .* sind(th_c);
    xr = r0r .* cosd(th_r); yr = r0r .* sind(th_r);

    % 仅对背景（空气/铁）删内部边；铜区因 key=NaN 不会参与
    delete_internal_boundaries_sector(r_edges, theta_edges, c0, Cs, isMirror, nr, nt, key_bg, ir_of_k, it_of_k);
    remove_isolated_nodes_sector(r_edges, theta_edges, c0, Cs, isMirror, nr, nt, key_bg, ir_of_k, it_of_k);
    % 合并空气带：只删除铜岛内侧已有分界线，不新增任何连线
    delete_internal_boundaries_in_copper_airband_sector( ...
        r_edges, theta_edges, c0, Cs, isMirror, nr, nt, is_copper_grid, copper_island_id);
    keep_cu_air_island_label = false(num_copper_islands, 1);
    reset_cu_geom_edge_cache();

    for k = 1:Nd
        code = mat_code(k);

        % ===== 背景 label：
        % 非铜格子按并区保留一个；铜格子仅在确有空气带时补空气标签 =====
        if code == 2
            ir = ir_of_k(k); it = it_of_k(k);
            island_id = copper_island_id(ir, it);
            if island_id > 0 && (~keep_cu_air_island_label(island_id)) && ...
                    copper_cell_has_air_band(k, nr, nt, ir_of_k, it_of_k, is_copper_grid)
                [xAirList, yAirList] = air_band_label_points_for_copper_cell( ...
                    k, c0, Cs, isMirror, nr, nt, r_edges, theta_edges, alpha, alpha, ...
                    ir_of_k, it_of_k, is_copper_grid);
                if ~isempty(xAirList)
                    mi_addblocklabel(xAirList(1), yAirList(1));
                    mi_selectlabel(xAirList(1), yAirList(1));
                    mi_setblockprop(mats.air, 1, 0, '', 0, groupIdRing, 0);
                    mi_clearselected();
                    keep_cu_air_island_label(island_id) = true;
                end
            end
        elseif keep_bg_label(k)
            if code == 1
                matName = mats.iron;
            else
                matName = mats.air;
            end
            mi_addblocklabel(xr(k), yr(k));
            mi_selectlabel(xr(k), yr(k));
            mi_setblockprop(matName, 1, 0, '', 0, groupIdRing, 0);
            mi_clearselected();
        end

        if code ~= 2
            continue;
        end

        % ===== 铜几何：只对铜岛外边界内缩；内部共享边不缩 =====
        add_copper_cell_boundary_island_inset(k, s, isMirror, Cs, c0, ...
            nr, nt, r_edges, theta_edges, alpha, alpha, groupIdCuGeom, ...
            ir_of_k, it_of_k, is_copper_grid);

        % ===== 铜 label：每个铜格子都保留独立标签/匝数 =====
        phaseId = pid(k);
        if phaseId == 0
            circCu = ''; turns = 0;
        else
            circCu = circNames{phaseId};
            turns = turns_per_circ(phaseId);
        end

        mi_addblocklabel(xc(k), yc(k));
        mi_selectlabel(xc(k), yc(k));
        mi_setblockprop(mats.copper, 1, 0, circCu, 0, groupIdCore, turns);
        mi_clearselected();
    end
end

end

function [island_id, num_islands] = build_copper_island_id_8conn(is_copper_grid)
[nr, nt] = size(is_copper_grid);
island_id = zeros(nr, nt);
num_islands = 0;

for ir = 1:nr
    for it = 1:nt
        if ~is_copper_grid(ir, it) || island_id(ir, it) ~= 0
            continue;
        end

        num_islands = num_islands + 1;
        stack = [ir, it];
        island_id(ir, it) = num_islands;

        while ~isempty(stack)
            ci = stack(end, 1); cj = stack(end, 2);
            stack(end, :) = [];

            nb = [ci-1,cj; ci+1,cj; ci,cj-1; ci,cj+1; ...
                  ci-1,cj-1; ci-1,cj+1; ci+1,cj-1; ci+1,cj+1];
            for n = 1:size(nb,1)
                ni = nb(n,1); nj = nb(n,2);
                if ni<1 || ni>nr || nj<1 || nj>nt
                    continue;
                end
                if ~is_copper_grid(ni, nj) || island_id(ni, nj) ~= 0
                    continue;
                end
                island_id(ni, nj) = num_islands;
                stack = [stack; ni, nj]; %#ok<AGROW>
            end
        end
    end
end
end


function tf = copper_cell_has_air_band(k, nr, nt, ir_of_k, it_of_k, is_copper_grid)
ir = ir_of_k(k);
it = it_of_k(k);
n8 = get_copper_neighbors8(ir, it, nr, nt, is_copper_grid);
sides = side_covered_by_8conn(n8);
side_air = ~(sides.has_in && sides.has_out && sides.has_lft && sides.has_rgt);
% 角空气带：两条边都被铜覆盖，但该角对角格为空气（或越界）
corner_air = (sides.has_in  && sides.has_lft && ~n8.ul) || ...
             (sides.has_in  && sides.has_rgt && ~n8.ur) || ...
             (sides.has_out && sides.has_lft && ~n8.dl) || ...
             (sides.has_out && sides.has_rgt && ~n8.dr);
tf = side_air || corner_air;
end

function delete_internal_boundaries_in_copper_airband_sector( ...
    r_edges, theta_edges, c0, Cs, isMirror, nr, nt, is_copper_grid, copper_island_id)
for j = 1:nt
    th_mid_base = 0.5*(theta_edges(j) + theta_edges(j+1));
    th_mid = map_theta(th_mid_base, c0, Cs, isMirror);
    for i = 1:(nr-1)
        if ~(is_copper_grid(i,j) && is_copper_grid(i+1,j))
            continue;
        end
        if copper_island_id(i,j) ~= copper_island_id(i+1,j)
            continue;
        end
        r = r_edges(i+1); x = r*cosd(th_mid); y = r*sind(th_mid);
        try
            mi_clearselected(); mi_selectarcsegment(x,y); mi_deleteselected(); mi_clearselected();
        catch
            try
                mi_clearselected(); mi_selectsegment(x,y); mi_deleteselected(); mi_clearselected();
            catch
                mi_clearselected();
            end
        end
    end
end

for j = 1:(nt-1)
    th_base = theta_edges(j+1);
    th = map_theta(th_base, c0, Cs, isMirror);
    for i = 1:nr
        if ~(is_copper_grid(i,j) && is_copper_grid(i,j+1))
            continue;
        end
        if copper_island_id(i,j) ~= copper_island_id(i,j+1)
            continue;
        end
        rmid = 0.5*(r_edges(i) + r_edges(i+1));
        x = rmid*cosd(th); y = rmid*sind(th);
        try
            mi_clearselected(); mi_selectsegment(x,y); mi_deleteselected(); mi_clearselected();
        catch
            mi_clearselected();
        end
    end
end
end


function [xAirList, yAirList] = air_band_label_points_for_copper_cell( ...
    k, c0, Cs, isMirror, nr, nt, r_edges, theta_edges, sh_r, sh_th, ...
    ir_of_k, it_of_k, is_copper_grid)

ir = ir_of_k(k);
it = it_of_k(k);

r1 = r_edges(ir);   r2 = r_edges(ir+1);
th1 = theta_edges(it); th2 = theta_edges(it+1);
dr  = sh_r  * (r2 - r1);
dth = sh_th * (th2 - th1);

n8 = get_copper_neighbors8(ir, it, nr, nt, is_copper_grid);
sides = side_covered_by_8conn(n8);
has_in  = sides.has_in;
has_out = sides.has_out;
has_lft = sides.has_lft;
has_rgt = sides.has_rgt;
need_ul = has_in  && has_lft && ~n8.ul;
need_ur = has_in  && has_rgt && ~n8.ur;
need_dl = has_out && has_lft && ~n8.dl;
need_dr = has_out && has_rgt && ~n8.dr;

ri1 = r1 + (~has_in ) * dr;
ri2 = r2 - (~has_out) * dr;
ti1 = th1 + (~has_lft) * dth;
ti2 = th2 - (~has_rgt) * dth;

rad_mid = 0.5*(ti1 + ti2);
tan_mid = 0.5*(ri1 + ri2);

rLab = []; tLab = [];

% 仅在存在"边空气带"时放边带标签；
% 角空气带(need_*)由后续角标签单独处理，避免 corner-only 情况误放一个侧边标签到铜区。
has_side_air = (~has_in) || (~has_out) || (~has_lft) || (~has_rgt);
if has_side_air
    % 若仅两侧且为对边空气带（内外 或 左右），两条空气带彼此不连通，需要两个标签
    if (~has_in && ~has_out && has_lft && has_rgt)
        rLab = [0.5*(r1 + ri1), 0.5*(ri2 + r2)];
        tLab = [rad_mid,          rad_mid];
    elseif (~has_lft && ~has_rgt && has_in && has_out)
        rLab = [tan_mid,          tan_mid];
        tLab = [0.5*(th1 + ti1),  0.5*(ti2 + th2)];
    else
        % 其他情形空气带连通，1 个标签即可
        if ~has_out
            rLab = 0.5*(ri2 + r2); tLab = rad_mid;
        elseif ~has_in
            rLab = 0.5*(r1 + ri1); tLab = rad_mid;
        elseif ~has_rgt
            rLab = tan_mid;        tLab = 0.5*(ti2 + th2);
        else
            % ~has_lft
            rLab = tan_mid;        tLab = 0.5*(th1 + ti1);
        end
    end
end

% 角空气带（仅角暴露时）补角标签，避免"内部格子漏掉角空气区"
corner_label_frac = 0.25;  % <0.5，确保标签落在角空气带内部而非斜分界线上
if need_ul
    rLab(end+1) = r1 + corner_label_frac*dr; %#ok<AGROW>
    tLab(end+1) = th1 + corner_label_frac*dth; %#ok<AGROW>
end
if need_ur
    rLab(end+1) = r1 + corner_label_frac*dr; %#ok<AGROW>
    tLab(end+1) = th2 - corner_label_frac*dth; %#ok<AGROW>
end
if need_dl
    rLab(end+1) = r2 - corner_label_frac*dr; %#ok<AGROW>
    tLab(end+1) = th1 + corner_label_frac*dth; %#ok<AGROW>
end
if need_dr
    rLab(end+1) = r2 - corner_label_frac*dr; %#ok<AGROW>
    tLab(end+1) = th2 - corner_label_frac*dth; %#ok<AGROW>
end

xAirList = zeros(size(rLab));
yAirList = zeros(size(rLab));
for ia = 1:numel(rLab)
    T = map_theta(tLab(ia), c0, Cs, isMirror);
    xAirList(ia) = rLab(ia)*cosd(T);
    yAirList(ia) = rLab(ia)*sind(T);
end
end

function add_copper_cell_boundary_island_inset(k, s, isMirror, Cs, c0, ...
    nr, nt, r_edges, theta_edges, sh_r, sh_th, groupIdCuGeom, ...
    ir_of_k, it_of_k, is_copper_grid)

ir = ir_of_k(k);
it = it_of_k(k);

r1 = r_edges(ir);   r2 = r_edges(ir+1);
th1 = theta_edges(it); th2 = theta_edges(it+1);

dr  = sh_r  * (r2 - r1);
dth = sh_th * (th2 - th1);

n8 = get_copper_neighbors8(ir, it, nr, nt, is_copper_grid);
sides = side_covered_by_8conn(n8);
has_in  = sides.has_in;
has_out = sides.has_out;
has_lft = sides.has_lft;
has_rgt = sides.has_rgt;
need_ul = has_in  && has_lft && ~n8.ul;
need_ur = has_in  && has_rgt && ~n8.ur;
need_dl = has_out && has_lft && ~n8.dl;
need_dr = has_out && has_rgt && ~n8.dr;

ri1 = r1 + (~has_in ) * dr;
ri2 = r2 - (~has_out) * dr;
ti1 = th1 + (~has_lft) * dth;
ti2 = th2 - (~has_rgt) * dth;

if ri2 <= ri1
    rm = 0.5*(r1+r2);
    ri1 = rm - 0.1*(r2-r1);
    ri2 = rm + 0.1*(r2-r1);
end
if ti2 <= ti1
    tm = 0.5*(th1+th2);
    ti1 = tm - 0.1*(th2-th1);
    ti2 = tm + 0.1*(th2-th1);
end

if need_ul || need_ur || need_dl || need_dr
    % 角空气带改为"方形缺口"（而非三角斜切）
    cr_ul = need_ul * dr; ct_ul = need_ul * dth;
    cr_ur = need_ur * dr; ct_ur = need_ur * dth;
    cr_dl = need_dl * dr; ct_dl = need_dl * dth;
    cr_dr = need_dr * dr; ct_dr = need_dr * dth;

    pr = [ri1 + cr_ul, ri2 - cr_dl];
    pt = [ti1,         ti1];
    if need_dl
        pr = [pr, ri2 - cr_dl, ri2];
        pt = [pt, ti1 + ct_dl, ti1 + ct_dl];
    else
        pr = [pr, ri2];
        pt = [pt, ti1];
    end

    pr = [pr, ri2];
    pt = [pt, ti2 - ct_dr];
    if need_dr
        pr = [pr, ri2 - cr_dr, ri2 - cr_dr];
        pt = [pt, ti2 - ct_dr, ti2];
    else
        pr = [pr, ri2];
        pt = [pt, ti2];
    end

    pr = [pr, ri1 + cr_ur];
    pt = [pt, ti2];
    if need_ur
        pr = [pr, ri1 + cr_ur, ri1];
        pt = [pt, ti2 - ct_ur, ti2 - ct_ur];
    else
        pr = [pr, ri1];
        pt = [pt, ti2];
    end

    pr = [pr, ri1];
    pt = [pt, ti1 + ct_ul];
    if need_ul
        pr = [pr, ri1 + cr_ul];
        pt = [pt, ti1 + ct_ul];
    else
        pr = [pr, ri1];
        pt = [pt, ti1];
    end

    use = true(size(pr));
    for ii = 2:numel(pr)
        if abs(pr(ii)-pr(ii-1)) < 1e-12 && abs(pt(ii)-pt(ii-1)) < 1e-12
            use(ii) = false;
        end
    end
    if abs(pr(1)-pr(end)) < 1e-12 && abs(pt(1)-pt(end)) < 1e-12
        use(end) = false;
    end
    pr = pr(use); pt = pt(use);

    nP = numel(pr);
    px = zeros(1,nP); py = zeros(1,nP);
    for iP = 1:nP
        T = map_theta(pt(iP), c0, Cs, isMirror);
        px(iP) = pr(iP)*cosd(T);
        py(iP) = pr(iP)*sind(T);
        mi_addnode(px(iP), py(iP));
        mi_selectnode(px(iP), py(iP));
        mi_setnodeprop('', groupIdCuGeom);
        mi_clearselected();
    end

    for iP = 1:nP
        jP = mod(iP, nP) + 1;
        add_segment_once(px(iP), py(iP), px(jP), py(jP), groupIdCuGeom);
    end
    return;
end

mapTh = @(th_base) (~isMirror) * (Cs + (th_base - c0)) + ...
                   ( isMirror) * (Cs - (th_base - c0));
T1 = mapTh(ti1);
T2 = mapTh(ti2);

p = zeros(4,2);
p(1,:) = [ri1*cosd(T1), ri1*sind(T1)];
p(2,:) = [ri2*cosd(T1), ri2*sind(T1)];
p(3,:) = [ri2*cosd(T2), ri2*sind(T2)];
p(4,:) = [ri1*cosd(T2), ri1*sind(T2)];

for i = 1:4
    mi_addnode(p(i,1), p(i,2));
    mi_selectnode(p(i,1), p(i,2));
    mi_setnodeprop('', groupIdCuGeom);
    mi_clearselected();
end

% 保留铜格子之间边界，但去重避免重复连线
add_segment_once(p(1,1), p(1,2), p(2,1), p(2,2), groupIdCuGeom);
add_segment_once(p(4,1), p(4,2), p(3,1), p(3,2), groupIdCuGeom);

% 周向边（常半径）用圆弧，且按扇区镜像方向选端点，避免后续扇区弧段丢失
angSpan = ti2 - ti1;           % 基域内一定为正
angMidBase = 0.5*(ti1 + ti2);
angMid = mapTh(angMidBase);

if ~isMirror
    add_arc_once(p(2,1), p(2,2), p(3,1), p(3,2), angSpan, 1, ...
        ri2*cosd(angMid), ri2*sind(angMid), groupIdCuGeom);
    add_arc_once(p(1,1), p(1,2), p(4,1), p(4,2), angSpan, 1, ...
        ri1*cosd(angMid), ri1*sind(angMid), groupIdCuGeom);
else
    add_arc_once(p(3,1), p(3,2), p(2,1), p(2,2), angSpan, 1, ...
        ri2*cosd(angMid), ri2*sind(angMid), groupIdCuGeom);
    add_arc_once(p(4,1), p(4,2), p(1,1), p(1,2), angSpan, 1, ...
        ri1*cosd(angMid), ri1*sind(angMid), groupIdCuGeom);
end

end

function reset_cu_geom_edge_cache()
global cu_geom_seg_cache cu_geom_arc_cache
cu_geom_seg_cache = [];
cu_geom_arc_cache = [];
cu_geom_seg_cache = containers.Map('KeyType', 'char', 'ValueType', 'logical');
cu_geom_arc_cache = containers.Map('KeyType', 'char', 'ValueType', 'logical');
end

function add_segment_once(x1, y1, x2, y2, groupId)
global cu_geom_seg_cache
if isempty(cu_geom_seg_cache) || ~isa(cu_geom_seg_cache, 'containers.Map')
    cu_geom_seg_cache = containers.Map('KeyType', 'char', 'ValueType', 'logical');
end
k = make_segment_key(x1, y1, x2, y2);
if isKey(cu_geom_seg_cache, k)
    return;
end
mi_addsegment(x1, y1, x2, y2);
mx = 0.5*(x1 + x2); my = 0.5*(y1 + y2);
mi_selectsegment(mx, my);
mi_setsegmentprop('', 0, 1, 0, groupId);
mi_clearselected();
cu_geom_seg_cache(k) = true;
end

function add_arc_once(x1, y1, x2, y2, angleDeg, maxSegDeg, selx, sely, groupId)
global cu_geom_arc_cache
if isempty(cu_geom_arc_cache) || ~isa(cu_geom_arc_cache, 'containers.Map')
    cu_geom_arc_cache = containers.Map('KeyType', 'char', 'ValueType', 'logical');
end
k = make_arc_key(x1, y1, x2, y2, angleDeg);
if isKey(cu_geom_arc_cache, k)
    return;
end
mi_addarc(x1, y1, x2, y2, angleDeg, maxSegDeg);
mi_selectarcsegment(selx, sely);
mi_setarcsegmentprop(maxSegDeg, '', 0, groupId);
mi_clearselected();
cu_geom_arc_cache(k) = true;
end

function k = make_segment_key(x1, y1, x2, y2)
p1 = [round(x1, 12), round(y1, 12)];
p2 = [round(x2, 12), round(y2, 12)];
if (p1(1) > p2(1)) || (p1(1) == p2(1) && p1(2) > p2(2))
    t = p1; p1 = p2; p2 = t;
end
k = sprintf('S:%.12g,%.12g|%.12g,%.12g', p1(1), p1(2), p2(1), p2(2));
end

function k = make_arc_key(x1, y1, x2, y2, angleDeg)
p1 = [round(x1, 12), round(y1, 12)];
p2 = [round(x2, 12), round(y2, 12)];
if (p1(1) > p2(1)) || (p1(1) == p2(1) && p1(2) > p2(2))
    t = p1; p1 = p2; p2 = t;
end
k = sprintf('A:%.12g,%.12g|%.12g,%.12g|%.12g', ...
    p1(1), p1(2), p2(1), p2(2), round(angleDeg, 12));
end

function thw = wrap_to_span(th, th_start, th_end)
span = th_end - th_start;
thw = mod(th - th_start, 360);
thw(thw >= span & thw < (360-span)) = thw(thw >= span & thw < (360-span)) - 360;
thw = thw + th_start;
end

function keyMerge = build_merge_keys_air_iron_mask_copper(mat_code)
Nd = numel(mat_code);
keyMerge = nan(Nd,1); % copper 留 NaN（背景并区时忽略）
for k = 1:Nd
    if mat_code(k) == 1
        keyMerge(k) = 200; % iron
    elseif mat_code(k) == 0
        keyMerge(k) = 100; % air
    else
        keyMerge(k) = nan; % copper -> masked out
    end
end
end

function keep_label = keep_one_label_per_region(nr, nt, key, ir_of_k, it_of_k)
keyGrid = nan(nr, nt); kGrid = nan(nr, nt);
for k = 1:numel(key)
    keyGrid(ir_of_k(k), it_of_k(k)) = key(k);
    kGrid(ir_of_k(k), it_of_k(k)) = k;
end

keep_label = false(numel(key),1);
visited = false(nr, nt);
for i = 1:nr
    for j = 1:nt
        if visited(i,j), continue; end
        if isnan(keyGrid(i,j)), visited(i,j) = true; continue; end

        target = keyGrid(i,j);
        stack = [i,j]; visited(i,j) = true;
        rep_k = kGrid(i,j);

        while ~isempty(stack)
            ci = stack(end,1); cj = stack(end,2); stack(end,:) = [];
            nb = [ci-1,cj; ci+1,cj; ci,cj-1; ci,cj+1];
            for n = 1:4
                ni = nb(n,1); nj = nb(n,2);
                if ni<1 || ni>nr || nj<1 || nj>nt || visited(ni,nj)
                    continue;
                end
                if keyGrid(ni,nj) == target
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

function delete_internal_boundaries_sector(r_edges, theta_edges, c0, Cs, isMirror, nr, nt, key, ir_of_k, it_of_k)
keyGrid = nan(nr, nt);
for k = 1:numel(key)
    keyGrid(ir_of_k(k), it_of_k(k)) = key(k);
end

for j = 1:nt
    th_mid_base = 0.5*(theta_edges(j) + theta_edges(j+1));
    th_mid = map_theta(th_mid_base, c0, Cs, isMirror);
    for i = 1:(nr-1)
        if isnan(keyGrid(i,j)) || isnan(keyGrid(i+1,j))
            continue;
        end
        if keyGrid(i,j) == keyGrid(i+1,j)
            r = r_edges(i+1); x = r*cosd(th_mid); y = r*sind(th_mid);
            try
                mi_clearselected(); mi_selectarcsegment(x,y); mi_deleteselected(); mi_clearselected();
            catch
                try
                    mi_clearselected(); mi_selectsegment(x,y); mi_deleteselected(); mi_clearselected();
                catch
                    mi_clearselected();
                end
            end
        end
    end
end

for j = 1:(nt-1)
    th_base = theta_edges(j+1);
    th = map_theta(th_base, c0, Cs, isMirror);
    for i = 1:nr
        if isnan(keyGrid(i,j)) || isnan(keyGrid(i,j+1))
            continue;
        end
        if keyGrid(i,j) == keyGrid(i,j+1)
            rmid = 0.5*(r_edges(i) + r_edges(i+1));
            x = rmid*cosd(th); y = rmid*sind(th);
            try
                mi_clearselected(); mi_selectsegment(x,y); mi_deleteselected(); mi_clearselected();
            catch
                mi_clearselected();
            end
        end
    end
end
end

function remove_isolated_nodes_sector(r_edges, theta_edges, c0, Cs, isMirror, nr, nt, key, ir_of_k, it_of_k)
keyGrid = nan(nr, nt);
for k = 1:numel(key)
    keyGrid(ir_of_k(k), it_of_k(k)) = key(k);
end

for i = 2:nr
    r = r_edges(i);
    for j = 2:nt
        k11 = keyGrid(i-1, j-1); k21 = keyGrid(i,   j-1);
        k12 = keyGrid(i-1, j);   k22 = keyGrid(i,   j);

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
            mi_clearselected(); mi_selectnode(x,y); mi_deleteselected(); mi_clearselected();
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

function n8 = get_copper_neighbors8(ir, it, nr, nt, is_copper_grid)
n8.in  = (ir > 1 ) && is_copper_grid(ir-1, it);
n8.out = (ir < nr) && is_copper_grid(ir+1, it);
n8.lft = (it > 1 ) && is_copper_grid(ir, it-1);
n8.rgt = (it < nt) && is_copper_grid(ir, it+1);

n8.ul = (ir > 1  && it > 1 ) && is_copper_grid(ir-1, it-1);
n8.ur = (ir > 1  && it < nt) && is_copper_grid(ir-1, it+1);
n8.dl = (ir < nr && it > 1 ) && is_copper_grid(ir+1, it-1);
n8.dr = (ir < nr && it < nt) && is_copper_grid(ir+1, it+1);
end

function sides = side_covered_by_8conn(n8)
% 把 8 邻接"折算"到每一条边：
% 为避免"正邻已是铜但仍在两格之间生成空气带"，边覆盖只由该侧正邻决定。
% 8 邻接信息仍用于整体邻接检测（get_copper_neighbors8），但不会把对角缺失
% 误判成该边需要开空气带。
sides.has_in  = n8.in;
sides.has_out = n8.out;
sides.has_lft = n8.lft;
sides.has_rgt = n8.rgt;
end
