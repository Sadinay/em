clear; clc;

% ---- 0.1 设计域参数（示意，改成你自己的 cfg）----
   cfg = struct('nr',18,'nt',10,'r_inner',31.5,'r_outer',52,...
     'theta_span_deg',15,'slot_center_deg',7.5,...
     'lip_w',1.8,'lip_h',2.5,'top_w',8.7,'top_h',2,...
     'bottom_w',15.5,'bottom_h',12.645+2.5,'slot_depth',17.5,...
     'yoke_buffer_deg',0.1);

domain0 = stator_design_domain_blank(cfg);  % 你现有的"空白定子设计域"函数
theta0_deg = 0;                             % 第一块 15° 就是 0~15°
domain  = prepare_design_domain(domain0, theta0_deg);  % 加上 x_c,y_c,Nd 等

Nd   = domain.Nd;
Lbit = Nd;

% ---- 0.2 6 个扇区的相别模板（你应该已经有 build_phase_id_sector）----
circNames = {'A+','A-','B+','B-','C+','C-'};  % 6 条电路
sectorCircuit = [4 3 6 5 2 1];    % 按你自己的相序填；这里只是一个例子
phase_id_sector = cell(6,1);      % 6 个扇区，每个一个 Nd×1 向量

for s = 1:6
    pid = sectorCircuit(s);       % 1..6
    phase_id_sector{s} = pid * ones(Nd,1);   % 整个扇区里的铜都挂到同一条电路
end

% ---- 0.3 材料 & 电路名 ----
mats = struct();
mats.air      = 'Air';
mats.iron     = 'Pure Iron';
mats.copper   = 'Copper';
mats.meshSize = 1.0;   % 设计域网格尺寸，自己调

% ---- 0.4 每个电路的"总匝数"设定（例子：每个电路 100 匝）----
N_phase_total = 100 * ones(1, numel(circNames));

% ---- 0.5 FEMM 模板文件 & 分组号 ----
ctx = struct();

ctx.domain             = domain;
ctx.sector_offsets_deg = 0:15:75;        % 或者 30+0:15:75 之类
ctx.phase_id_sector    = phase_id_sector;
ctx.groupId_design     = 30;             % 设计域 group 号你自己定

ctx.baseFemFile = 'blank_18x10.fem';

ctx.mats = struct();
ctx.mats.air      = 'Air';
ctx.mats.iron     = 'Pure Iron';
ctx.mats.copper   = 'Copper';
ctx.mats.meshSize = 4.0;

ctx.circNames = {'A+','A-','B+','B-','C+','C-'};

ctx.T_min       = 0.5;    % 目标平均转矩要求（先随便放一个）
ctx.penaltyCoef = 10;

ctx.N_phase_total   = N_phase_total;

ctx.groupId_core = 30;
ctx.groupId_ring = 31;
ctx.inset_r_ratio  = 0.2;
ctx.inset_th_ratio = 0.2;

ctx.cfg = cfg;

% 你 trit 的材料编码（按你实际改）
ctx.airCode  = 0;
ctx.ironCode = 1;
ctx.cuCode   = 2;

% 你的拓扑是扇区重复的，所以建议 θ 方向周期连通
ctx.thetaPeriodic = true;


%% ---------- 1. 用一条 bits 测试 eval_design_femm ----------
Nd   = domain.Nd;
Lbit = Nd;

S = load('best_result_0120_groupA2_120gen.mat', 'best_bits', 'cfg');
assert(numel(S.best_bits) == ctx.domain.Nd, 'seed_bits length mismatch: cfg changed?');
seed_bits = S.best_bits;

[J, T_avg, T_ripple] = eval_design_femm(seed_bits, ctx);

fprintf('Test design: J = %.4f, T_avg = %.3f Nm, T_ripple = %.3f Nm\n', ...
        J, T_avg, T_ripple);