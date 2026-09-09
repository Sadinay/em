function domain = stator_design_domain(cfg)

%
%  CFG 必填字段：
%    nr, nt                - 径向/切向单元数（例：8、20）
%    r_inner, r_outer      - 定子内/外半径（mm）
%    theta_span_deg        - 设计扇区总角度（度，例：15）
%    slot_center_deg       - 该扇区内槽中心角（度，通常=theta_span_deg/2）
%    lip_w, lip_h      - 槽口矩形的宽/高（mm）
%    top_w,  top_h         - 槽体上边宽、上边到槽口顶的高（mm）
%    bottom_w, bottom_h    - 槽体底边宽、槽体高度（mm）
%    slot_depth            - 槽总深度（mm，含底部外延）
%    lip_w                 - 齿唇宽（mm，对应槽口两侧微收口）
%
%  CFG 可选字段：
%    yoke_buffer_deg       - 扇区边界附近的冻结半角，利于周期边界（默认 0）
%
%  输出 DOMAIN 字段：
%    r_edges, theta_edges  - 网格边界
%    design_mask           - nt×nr 逻辑矩阵，true = 可由优化器切换铁/空；false = 冻结
%    slot_mask             - nt×nr 逻辑矩阵，true = 槽/线圈冻结区（含槽口+六边形）
%    slot_outline_uv       - 槽截面顶点（局部 u/v 坐标，便于调试/复用）
%    notes                 - 文字提醒
%
%  示例：
%    cfg = struct(
%    domain = stator_design_domain(cfg);
%    visualize_design_domain(domain); % 可视化确认掩膜是否符合预期
%
%  参见 APPLY_DESIGN_MASK、VISUALIZE_DESIGN_DOMAIN。

% 使用较老版本 MATLAB/Octave 时，避免 ARGUMENTS 语法带来的解析错误，改为
% 传统的字段检查与默认值填充。
required_fields = { ...
    'nr','nt','r_inner','r_outer','theta_span_deg','slot_center_deg', ...
    'lip_w','lip_h','top_w','top_h','bottom_w','bottom_h', ...
    'slot_depth'};
for f = required_fields
    if ~isfield(cfg, f{1})
        error('缺少必填字段 cfg.%s', f{1});
    end
end

% 默认值（如已提供则不覆盖）。
if ~isfield(cfg, 'yoke_buffer_deg');  cfg.yoke_buffer_deg  = 0; end

% 数值检查（正数/非负）。
mustBePositive = @(v,name) assert(isnumeric(v) && isscalar(v) && v>0, ...
    '字段 %s 需为正数标量，当前为 %s', name, mat2str(v));
mustBeNonnegative = @(v,name) assert(isnumeric(v) && isscalar(v) && v>=0, ...
    '字段 %s 需为非负标量，当前为 %s', name, mat2str(v));

mustBePositive(cfg.nr, 'nr');
mustBePositive(cfg.nt, 'nt');
mustBePositive(cfg.r_inner, 'r_inner');
mustBePositive(cfg.r_outer, 'r_outer');
mustBePositive(cfg.lip_w, 'lip_w');
mustBePositive(cfg.lip_h, 'lip_h');
mustBePositive(cfg.top_w, 'top_w');
mustBePositive(cfg.top_h, 'top_h');
mustBePositive(cfg.bottom_w, 'bottom_w');
mustBePositive(cfg.bottom_h, 'bottom_h');
mustBePositive(cfg.slot_depth, 'slot_depth');
mustBePositive(cfg.theta_span_deg, 'theta_span_deg');
mustBeNonnegative(cfg.yoke_buffer_deg, 'yoke_buffer_deg');

if cfg.r_outer <= cfg.r_inner
    error('r_outer must be greater than r_inner (got %.2f vs %.2f).', ...
          cfg.r_outer, cfg.r_inner);
end

if cfg.slot_depth <= cfg.lip_h
    error('slot_depth must exceed lip_h (got %.2f vs %.2f).', ...
          cfg.slot_depth, cfg.lip_h);
end

if cfg.slot_depth <= cfg.top_h + cfg.bottom_h
    error('slot_depth 应大于 top_h+bottom_h，当前 %.2f <= %.2f', ...
          cfg.slot_depth, cfg.top_h + cfg.bottom_h);
end

% ---------- 构建网格 ----------
r_edges     = linspace(cfg.r_inner, cfg.r_outer, cfg.nr + 1);
theta_edges = linspace(0, cfg.theta_span_deg, cfg.nt + 1);

r_cent_1d     = r_edges(1:end-1)    + diff(r_edges)/2;
theta_cent_1d = theta_edges(1:end-1) + diff(theta_edges)/2;

[theta_centers, r_centers] = meshgrid(theta_cent_1d, r_cent_1d);
theta_centers = theta_centers';   % nt×nr
r_centers     = r_centers';

% 槽截面
slot_outline = build_slot_outline(cfg);

% 槽中心角：注意这里可能被重定义（半槽模式）
slot_center_deg = cfg.slot_center_deg;
r_mid = cfg.r_inner + cfg.slot_depth/2;
slot_full_angle_deg = 2 * rad2deg(max(abs(slot_outline.u)) / r_mid);

if cfg.theta_span_deg <= slot_full_angle_deg
    slot_center_deg = theta_edges(1); % 扇区左边界，呈现半槽
end

% 用统一的 slot_center_deg 映射中心点 (u_local, v_local)
dtheta_rad = deg2rad(theta_centers - slot_center_deg);
v_local    = r_centers - cfg.r_inner;
u_local    = r_centers .* dtheta_rad;

% ---------- 精细版槽掩膜 ----------
[nt, nr] = size(theta_centers);
slot_mask = false(nt, nr);

for it = 1:nt
    th1 = theta_edges(it);
    th2 = theta_edges(it+1);

    % 这里也要用同一个 slot_center_deg，而不是 cfg.slot_center_deg
    dth1 = deg2rad(th1 - slot_center_deg);
    dth2 = deg2rad(th2 - slot_center_deg);

    for ir = 1:nr
        r1 = r_edges(ir);
        r2 = r_edges(ir+1);

        v1 = r1 - cfg.r_inner;
        v2 = r2 - cfg.r_inner;

        u11 = r1 * dth1; v11 = v1;
        u12 = r2 * dth1; v12 = v2;
        u21 = r2 * dth2; v21 = v2;
        u22 = r1 * dth2; v22 = v1;

        u_corners = [u11, u12, u21, u22];
        v_corners = [v11, v12, v21, v22];

        u0 = u_local(it, ir);
        v0 = v_local(it, ir);

        uu = [u_corners, u0];
        vv = [v_corners, v0];

        in = inpolygon(uu, vv, slot_outline.u, slot_outline.v);
        slot_mask(it, ir) = any(in);
    end
end

% ---------- 用角度窗口裁剪掉"离槽太远"的误伤单元 ----------
half_angle = slot_full_angle_deg / 2;     % 槽半宽（度）
margin_deg = 0.3;                         % 给一点富余，比如 0.3° 可调

% 单元中心相对槽中心的角度偏差（度）
ang_offset = abs(theta_centers - slot_center_deg);

% 对偏差大于半槽角宽+裕度的单元，强制认为"不属于槽"
slot_mask(ang_offset > (half_angle + margin_deg)) = false;

edge_mask   = (theta_centers <= cfg.yoke_buffer_deg) | ...
              (theta_centers >= (cfg.theta_span_deg - cfg.yoke_buffer_deg));

design_mask = ~slot_mask & ~edge_mask;


% 汇总结果。
domain = struct();
domain.r_edges = r_edges;
domain.theta_edges = theta_edges;
domain.design_mask = design_mask;
domain.slot_mask = slot_mask;
domain.slot_outline_uv = slot_outline;
domain.slot_center_deg = cfg.slot_center_deg;
domain.notes = {
    'design_mask: true = 优化器可切换铁/空气，false = 冻结';
    'slot_mask: true = 槽/线圈区保持不变（槽口矩形 + 六边形槽体）';
    'apply_design_mask(bits, domain, base_val) 会强制冻结区域';
        sprintf('sector %.1f° @ center %.2f°, lip %.1f×%.1f mm, top %.1f×%.1f mm, bottom %.1f×%.1f mm', ...
            cfg.theta_span_deg, cfg.slot_center_deg, ...
            cfg.lip_w, cfg.lip_h, cfg.top_w, cfg.top_h, ...
            cfg.bottom_w, cfg.bottom_h)
};

end

function grid = apply_design_mask(bits, domain, base_val)
%APPLY_DESIGN_MASK 将基因比特映射到网格并冻结槽/缓冲区。
%  GRID = APPLY_DESIGN_MASK(BITS, DOMAIN, BASE_VAL) 返回 nt×nr 数组，
%  可编辑单元取自 BITS（重塑后），冻结区域（槽/线圈或边界缓冲）
%  则被强制为 BASE_VAL（例如铁 = 1）。
%
%  BITS 需为长度 nt*nr 的向量，与 DOMAIN 掩膜尺寸一致。

if numel(bits) ~= numel(domain.design_mask)
    error('Bitstring length %d does not match grid size %d.', ...
          numel(bits), numel(domain.design_mask));
end

grid = reshape(bits, size(domain.design_mask));
freeze_mask = ~domain.design_mask; % 包含槽与边界缓冲
grid(freeze_mask) = base_val;
end

function outline = build_slot_outline(cfg)
%BUILD_SLOT_OUTLINE 生成槽口矩形 + 六边形槽体的局部坐标（u/v）。
%  基于 strukturFemm 的槽型：
%    - mouth_w / mouth_h     : 槽口矩形
%    - top_w  / top_h        : 槽体上边宽与槽口顶到上边的高
%    - bottom_w / bottom_h   : 槽体底边宽与槽体高度
%    - slot_depth            : 槽总深度（含 bottom_ext）
%    - bottom_ext (0.5 mm)   : 底部外延
%    - lip_w                 : 齿唇宽，控制槽口两侧向内收一点

%  坐标系：v 从齿根圆向外（mm），u 为切向位移（mm，右正）。

v0 = 0;                        % 齿根处
v_lip = v0 + cfg.lip_h;    % 槽口矩形上边
v_top   = v0 + cfg.top_h; % 槽体上边高度
v_bottom= v0 + cfg.bottom_h;% 槽体底边高度
v_tip   = cfg.slot_depth; % 槽底最外侧

w_top   = cfg.top_w/2;
w_bottom= cfg.bottom_w/2;
w_lip   = cfg.lip_w/2; % 齿唇微收口（v=0 处）

outline_u = [ ...
    w_lip, v0;          
    w_lip, v_lip;   
    w_top, v_top;       
    w_bottom, v_bottom; 
    0, v_tip;           
   -w_bottom, v_bottom; 
   -w_top, v_top;       
   -w_lip, v_lip;   
   -w_lip, v0          
];

% 变为行向量，便于 inpolygon 调用。
outline.u = outline_u(:,1)';
outline.v = outline_u(:,2)';
end