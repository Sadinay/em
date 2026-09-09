function domain = stator_design_domain_blank(cfg)
%STATOR_DESIGN_DOMAIN_BLANK
%   在一个空白的定子扇区内生成离散设计域（不考虑槽口几何）。
%
%  必填 cfg 字段：
%    nr, nt           - 径向 / 切向 单元数
%    r_inner, r_outer - 设计域内半径范围（例如：气隙半径 ~ 轭外圆）
%    theta_span_deg   - 扇区角度（例如 15°）
%
%  可选：
%    yoke_buffer_deg  - 在扇区两侧冻结的角度缓冲（度），用于周期边界；默认 0
%
%  输出 domain：
%    r_edges, theta_edges - 网格边界（长度 nr+1 / nt+1）
%    design_mask          - nt×nr，true = 设计单元，false = 冻结
%    slot_mask            - nt×nr，全 false（保留字段方便旧代码复用）
%    notes                - 一些文字说明

%% ---- 配置检查 ----
required_fields = {'nr','nt','r_inner','r_outer','theta_span_deg'};
for k = 1:numel(required_fields)
    f = required_fields{k};
    if ~isfield(cfg, f)
        error('缺少必填字段 cfg.%s', f);
    end
end

if ~isfield(cfg, 'yoke_buffer_deg'); cfg.yoke_buffer_deg = 0; end

mustPos = @(v,name) assert(isnumeric(v)&&isscalar(v)&&v>0, ...
    '字段 %s 必须为正数标量，当前为 %s', name, mat2str(v));
mustNonNeg = @(v,name) assert(isnumeric(v)&&isscalar(v)&&v>=0, ...
    '字段 %s 必须为非负标量，当前为 %s', name, mat2str(v));

mustPos(cfg.nr,            'nr');
mustPos(cfg.nt,            'nt');
mustPos(cfg.r_inner,       'r_inner');
mustPos(cfg.r_outer,       'r_outer');
mustPos(cfg.theta_span_deg,'theta_span_deg');
mustNonNeg(cfg.yoke_buffer_deg, 'yoke_buffer_deg');

assert(cfg.r_outer > cfg.r_inner, 'r_outer 必须大于 r_inner');

%% ---- 构建极坐标网格边界 ----
r_edges     = linspace(cfg.r_inner, cfg.r_outer, cfg.nr + 1);
theta_edges = linspace(0, cfg.theta_span_deg, cfg.nt + 1);   % [0,15°]

% 网格中心（nt×nr）
r_cent_1d     = r_edges(1:end-1)    + diff(r_edges)/2;
theta_cent_1d = theta_edges(1:end-1)+ diff(theta_edges)/2;
[theta_centers, r_centers] = meshgrid(theta_cent_1d, r_cent_1d);
theta_centers = theta_centers';
r_centers     = r_centers'; %#ok<NASGU> % 目前没用，保留以便以后扩展

%% ---- 设计掩膜：整块都可设计，仅扇区两侧留缓冲 ----
design_mask = true(cfg.nt, cfg.nr);     % 全部可设计

if cfg.yoke_buffer_deg > 0
    edge_mask = (theta_centers <= cfg.yoke_buffer_deg) | ...
                (theta_centers >= (cfg.theta_span_deg - cfg.yoke_buffer_deg));
    design_mask(edge_mask) = false;      % 冻结边界缓冲
end

% 不再考虑槽：slot_mask 全 false（兼容旧接口）
slot_mask = false(cfg.nt, cfg.nr);

%% ---- 汇总输出 ----
domain = struct();
domain.r_edges      = r_edges;
domain.theta_edges  = theta_edges;
domain.design_mask  = design_mask;
domain.slot_mask    = slot_mask;
domain.slot_center_deg = NaN;
domain.slot_outline_uv = [];
domain.notes = {
    '空白定子设计域：在 r∈[r_inner,r_outer], θ∈[0,theta_span_deg] 内离散化';
    'design_mask: true = 优化可切换材料；false = 冻结（含两侧缓冲）';
};

end
