


% CLONALG - Clonal Selection Algorithm for Optimization Problems
% For 3D MINIMIZATION Problems

% OUTPUTS:

% x,y,fx	-> fx = f(x,y) is the minimal value of the function
% vfx		-> Best fitness of population through generations

% Reference:
% de Castro, L. N., and Von Zuben, F. J. Learning and Optimization Using
% the Clonal Selection Principle. IEEE Transactions on Evolutionary
% Computation. v. 6(3). 2002. DOI: 10.1109/TEVC.2002.1011539

clear
close
clc

%%初始化
cfg = struct('nr',18,'nt',10,'r_inner',31.5,'r_outer',52,...
    'theta_span_deg',15,'slot_center_deg',7.5,...
    'lip_w',1.8,'lip_h',2.5,'top_w',8.7,'top_h',2,...
    'bottom_w',15.5,'bottom_h',12.645+2.5,'slot_depth',17.5,...
    'yoke_buffer_deg',0.1);
domain0 = stator_design_domain_blank(cfg);
theta0   = 0;                                      % 设计域在 FEMM 里的起始角
domain   = prepare_design_domain(domain0, theta0); % 加上 design_idx/x_c/y_c/…

Nd   = domain.Nd;
Lbit = Nd;                  % 你也可以把变量名改成 Lgene

circNames = {'A+','A-','B+','B-','C+','C-'};  % 6 条电路
sectorCircuit = [4 3 6 5 2 1];    % 按你自己的相序填；这里只是一个例子
phase_id_sector = cell(6,1);      % 6 个扇区，每个一个 Nd×1 向量

for s = 1:6
    pid = sectorCircuit(s);       % 1..6
    phase_id_sector{s} = pid * ones(Nd,1);   % 整个扇区里的铜都挂到同一条电路
end

N_phase_total = 100 * ones(1, numel(circNames));
%% ---------- 1. 这里组装 ctx（只做一次） ----------
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
ctx.mats.meshSize = 2.0;

ctx.circNames = {'A+','A-','B+','B-','C+','C-'};

ctx.T_min       = 0.8;    % 目标平均转矩要求
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
ctx.thetaPeriodic = false;
ctx.Amin = 4;
% ===== penalty config for floating islands (scheme A: count-based sigmoid) =====
ctx.penFloating = struct();

ctx.penFloating.enable = true;

% sigmoid: p = w * sigmoid(k*(nFloat-0.5))
ctx.penFloating.kFe = 50;
ctx.penFloating.wFe = 0.10;   % 注意：你的J_em≈0.08，wFe建议>=0.08量级
ctx.penFloating.kCu = 30;
ctx.penFloating.wCu = 0.05;

ctx.fastReject.enable = true;
ctx.fastReject.AminCu = 4;     % 面积阈值（按你的定义）
ctx.fastReject.badJ   = 1e6;   % 直接淘汰值（要明显比正常J大）
ctx.fastReject.wCu    = 50;    % 可选：按数量加一点"更差"
ctx.fastReject.wFe    = 50;
%%
% bits_test = randi([0,1], 1, domain.Nd);
% J = eval_design_femm(bits_test, ctx);

%% 制作参考模型的seed

% refFemFile = 'strukturFemm.fem';
%
% seed_bits = seed_from_reference_model(ctx, refFemFile);
%
% save('seed_bits_fine.mat', 'seed_bits', 'cfg');
%%
% dump_pv(39, 2.7);   % 铜区点
% dump_pv(32, 2.6);   % 铁区点
% dump_pv(32,0.5);  % 空气点
%%
%使用自己的bits
% seed_bits = paint_seed_bits(ctx);

% S = load('seed_bits_fine_groupB_5gen.mat', 'seed_bits', 'cfg');
% assert(numel(S.seed_bits) == ctx.domain.Nd, 'seed_bits length mismatch: cfg changed?');
% seed_bits = S.seed_bits;

%% 使用已经优化过的参考模型
S = load(['best_result_0305_groupF_900gen.mat'], 'best_bits', 'cfg');
assert(numel(S.best_bits) == ctx.domain.Nd, 'seed_bits length mismatch: cfg changed?');
seed_bits = S.best_bits;
%%
rng('shuffle') %default时会输出一样的结果

N    = 10;                  % Population size

gen  = 100;                   % Number of generations
% pm_start = 0.15;   % 原来的 pm
% pm_end   = 0.03;   % 后期别太小，0.03~0.08 都行
%
% alpha = (it-1)/(gen-1);          % 0 -> 1
% pm_t  = pm_start + (pm_end - pm_start)*alpha;                 % Mutation probability
d    = 0.2;                  % Population to suffer random reshuffle %保证N*d>1
beta = 0.4;                  % Proportion of clones %保
%
% 证N*beta>1 否则不足一个克隆，会报错

% 1) 手工种子（你来定义 seed_bits）

Ab = randi([0,2], N, Nd);     % 先全随机，防止漏填

Ab(1,:) = seed_bits;

n_near = round(0.6*(N-1));    % 近邻（小扰动）
n_mid  = round(0.2*(N-1));    % 中扰动
% 剩下的自动当 random

pm_near = 0.05;              % 0.03~0.08
pm_mid  = 0.12;              % 0.10~0.18
pm_far  = 0.25;              % 可选：最远（强扰动）

idx = 2;

% 近邻
for k = 1:n_near
    Ab(idx,:) = mutate_trit(seed_bits, pm_near);
    idx = idx + 1;
end

% 中扰动
for k = 1:n_mid
    Ab(idx,:) = mutate_trit(seed_bits, pm_mid);
    idx = idx + 1;
end

% 可选：塞一个强扰动（保证有"远跳"）
if idx <= N
    Ab(idx,:) = mutate_trit(seed_bits, pm_far);
end

% 剩下 idx+1..N 的保持 randi 随机（但也建议清理一下）
for i = (idx+1):N

end

% Function to optimization
p = gcp('nocreate');
if ~isempty(p), delete(p); end
parpool('local', N);

W = parallel.pool.Constant(@() femm_worker_init(ctx), @(S) femm_worker_cleanup(S));

f = @(bits) f_wrapper(bits, ctx, W);

% varMin = 20; % Lower bound
% varMax =  40; % Upper bound

fbest = 0; % Global f best (minimum)

% x = meshgrid(linspace(varMin, varMax, 61));
% y = meshgrid(linspace(varMin, varMax, 61))';
%
% vxp = x;
% vyp = y;
%
% vzp = f([x(:),y(:)]);
% vzp = reshape(vzp,size(x));

% % 先随便给一个占位，以便 imprime 不报错（反正 PRINT=0 不会画）
% vxp = 0; vyp = 0; vzp = 0;
%
% x = decode(Ab(:,1:22),varMin,varMax);
% y = decode(Ab(:,23:end),varMin,varMax);
%
% fit = f([x(:),y(:)]);
%
% figure
% imprime(0,vxp,vyp,vzp,x,y,fit)
%
% % Hypermutation controlling parameters
% pma  = pm;
% itpm = gen;
% pmr  = 0.8;

% General defintions
% 预分配
J_hist      = nan(gen,1);        % 每代最优 J
best_bits_hist = zeros(gen, Lbit);% 可选：记录每代的最优基因串

% % 先评估初始种群
% fit = zeros(N,1);
% for i = 1:N
%     fit(i) = f(Ab(i,:));
% end

%% 保存过程数据
Ab_hist  = zeros(gen, N, Lbit, 'uint8');  % 每代的种群基因
ind_hist = zeros(gen, N, 'uint16');       % 排名索引（N通常<=65535）
J_hist_all = zeros(gen, N);               % 每代所有个体的J（可选）

% ===== 保存悬浮岛统计（每代×每个体）=====
nFloatFe_hist = zeros(gen, N, 'uint16');
nFloatCu_hist = zeros(gen, N, 'uint16');

ratioFe_hist  = zeros(gen, N, 'single');   % 可选：比例更方便画图
ratioCu_hist  = zeros(gen, N, 'single');

nCompCu_hist  = zeros(gen, N, 'uint16');   % 可选：铜连通块数量（调试/分析）

%% --------- 迭代 ----------
globalBestJ    = inf;
globalBestBits = Ab(1,:);

Jcache = containers.Map('KeyType','char','ValueType','double');

for it = 1:gen
    % 1) 评估当前种群
    % J = zeros(N,1);
    % p = gcp();
    % parfor (i = 1:N, p.NumWorkers)
    %     J(i) = f(Ab(i,:));
    % end
    p = gcp();
    F = parallel.FevalFuture.empty(N,0);
    for i = 1:N
        bits = Ab(i,:);

        % ---- 悬浮岛检测（不修改 bits）----
        fe = detect_floating_iron(bits, ctx.cfg, ctx.ironCode, ctx.thetaPeriodic);
        cu = detect_floating_copper_small_islands(bits, ctx.cfg, ctx.cuCode, ctx.thetaPeriodic,ctx.Amin);

        nFloatFe_hist(it,i) = uint16(fe.nFloat);
        nFloatCu_hist(it,i) = uint16(cu.nFloat);

        ratioFe_hist(it,i)  = single(fe.ratio);
        ratioCu_hist(it,i)  = single(cu.ratio);

        nCompCu_hist(it,i)  = uint16(cu.nComp);

        % ---- 再去 FEMM 评估 ----
        %F(i) = parfeval(p, f, 1, bits);
    end
        [J, Jcache] = eval_bits_batch_cached(Ab, p, f, Jcache);
    % J = zeros(N,1);
    % for k = 1:N
    %     [idx, val] = fetchNext(F);
    %     J(idx) = val;
    % end

    % 2) 排序（越小越好）
    [J_sorted, ind] = sort(J);
    %% 保存过程数据
    Ab_hist(it,:,:) = uint8(Ab);
    ind_hist(it,:)  = uint16(ind);
    J_hist_all(it,:)= J(:).';
    rng_hist(it) = rng;

    nFloatFe_sorted = nFloatFe_hist(it, ind);
    nFloatCu_sorted = nFloatCu_hist(it, ind);

    ratioFe_sorted  = ratioFe_hist(it, ind);
    ratioCu_sorted  = ratioCu_hist(it, ind);
    %%
    bestJ = J_sorted(1);
    best_bits_hist(it,:) = Ab(ind(1),:);   % <--- 记录该代最优设计
    J_hist(it) = bestJ;
    fprintf('Gen %2d: best J = %.4f\n', it, bestJ);

    if bestJ < globalBestJ
        globalBestJ    = bestJ;
        globalBestBits = Ab(ind(1),:);
    end

    eliteBits = globalBestBits;
    eliteJ    = globalBestJ;
    % 如果已经是最后一代：别再生成下一代了，直接结束
    if it == gen
        break;
    end
    % 3) 按排名克隆
    % 每个个体克隆 cs(i) 个
    % 3) 按排名克隆（让好的个体多克隆）
    ns = max(0, N-2);   % 只克隆前 N-2 个父代
    cs_max = beta * N;  % 最大复制个数
    cs_min = 1;

    cs = zeros(1, N);   % 先全置0（后两个默认不克隆）

    if ns > 0
        rank = 1:ns;  % 只对前 ns 个(排名更好者)分配克隆数
        cs_sel = round( cs_max - (cs_max-cs_min) * (rank-1)/max(1,ns-1) );
        cs_sel(cs_sel < cs_min) = cs_min;
        cs(1:ns) = cs_sel;
    end

    pcs = cumsum(cs);
    T   = zeros(pcs(end), Lbit);  % 克隆池总大小 = sum(cs)

    % ===== 超变异率：按父代排名（簇编号）递增 =====
    pm_min = 0.03;
    pm_max = 0.20;
    gamma  = 2;

    pm_rank = zeros(1, N);  % 先给N个占位，后两个用不到

    if ns > 0
        rank = 1:ns;
        pm_rank(1:ns) = pm_min + (pm_max - pm_min) * ((rank-1)/max(1,ns-1)).^gamma;
    end

    start_idx = 1;
    for i = 1:ns
        stop_idx = pcs(i);
        T(start_idx:stop_idx, :) = repmat( Ab(ind(i),:), cs(i), 1 );
        start_idx = stop_idx + 1;
    end


    % ===== 第4步：按簇超变异（每个父代一个 pm）=====
    Mmut = false(size(T));   % 变异掩膜（与 T 同尺寸）
    start_idx = 1;

    for ii = 1:ns
        stop_idx = pcs(ii);

        % 这一簇的变异概率
        pmi = pm_rank(ii);

        % 对该簇生成掩膜
        Mmut(start_idx:stop_idx, :) = (rand(stop_idx-start_idx+1, Lbit) <= pmi);

        start_idx = stop_idx + 1;
    end

    % 应用变异：被选中的 trit 必跳到另一个值
    if any(Mmut(:))
        idx = find(Mmut);
        old = T(idx);
        delta = randi([1,2], size(old));
        T(idx) = mod(old + delta, 3);
    end

    % 5) 评估克隆池

    % M = size(T,1);
    % F = parallel.FevalFuture.empty(M,0);
    % for i = 1:M
    %     F(i) = parfeval(p, f, 1, T(i,:));
    % end
    % 
    % J_T = zeros(M,1);
    % for k = 1:M
    %     [idx, val] = fetchNext(F);
    %     J_T(idx) = val;
    % end
            [J_T, Jcache] = eval_bits_batch_cached(T, p, f, Jcache);

    [minCloneJ, idxMinClone] = min(J_T);
    if minCloneJ < globalBestJ
        globalBestJ    = minCloneJ;
        globalBestBits = T(idxMinClone,:);
    end

    % 6) 每个簇：父代 vs 最优克隆，择优进入 newAb
    newAb = zeros(size(Ab));
    start_idx = 1;
    for i = 1:ns
        stop_idx = pcs(i);

        parentBits = Ab(ind(i),:);   % 排名第 i 的父代
        parentJ    = J_sorted(i);

        [bestCloneJ, localBest] = min( J_T(start_idx:stop_idx) );
        bestCloneBits = T(start_idx + localBest - 1, :);

        % 变异没带来提升，就保留父代（防退化）
        if bestCloneJ <= parentJ
            newAb(i,:) = bestCloneBits;
        else
            newAb(i,:) = parentBits;
        end

        start_idx = stop_idx + 1;
    end

    % (B) 最后两个（i = ns+1:N）：不参与克隆，直接继承父代（按排名）
    for i = ns+1:N
        newAb(i,:) = Ab(ind(i),:);
    end

    % 7) Repertoire shift：随机重置一些个体（但不动精英）
    nedit = max(1, round(d * N));

    candidates = 2:N;  % 1号留给精英
    if ~isempty(candidates)
        rp = candidates(randperm(numel(candidates), min(nedit, numel(candidates))));
        pm_imm = 0.25;  % 比正常 pm 大，负责跳出局部
        for kk = 1:numel(rp)
            if rand < 0.5
                % 50%：围绕最优做强变异（局部跳）
                newAb(rp(kk),:) = mutate_trit(globalBestBits, pm_imm);
            else
                % 50%：真随机移民（全局跳）
                newAb(rp(kk),:) = randi([0,2], 1, Lbit);
            end
        end
    end

    % ====== 精英强行塞回（保证不丢）======
    newAb(1,:) = globalBestBits;

    % 更新种群
    Ab = newAb;
end
%%
% --------- 画收敛曲线 ----------
figure;
semilogy(J_hist,'-o');
grid on;
xlabel('Generation');
ylabel('Best J (log scale)');
title('Clonal Selection on Topology Bits');

%%
[bestOverall, genIdx] = min(J_hist);
fprintf('\nBest J found = %.4f at generation %d\n', bestOverall, genIdx);

best_bits = best_bits_hist(genIdx,:);   % 这一代的基因就是全局最优

% best_bits = remove_floating_iron(best_bits, ctx.cfg, ctx.ironCode, ctx.airCode, ctx.thetaPeriodic);

%% ---------- 保存最优结果（bits + FEMM 文件） ----------
outDir = 'C:\Users\Alexander\Desktop\FP\IA_test_Float';
if ~exist(outDir,'dir'), mkdir(outDir); end

tag = datestr(now,'yyyymmdd_HHMMSS');
save(fullfile(outDir, ['best_result_' tag '.mat']), ...
    'best_bits','bestOverall','genIdx','J_hist','best_bits_hist','cfg','ctx');

bestFemPath = fullfile(outDir, sprintf('best_J%.6g_%s.fem', bestOverall, tag));
bestPngPath = fullfile(outDir, sprintf('best_J%.6g_%s.png', bestOverall, tag));

%% 保存过程数据
save(fullfile(outDir, ['trace_all_' tag '.mat']), ...
    'Ab_hist','ind_hist','J_hist_all','J_hist','best_bits_hist', ...
    'nFloatFe_hist','nFloatCu_hist','ratioFe_hist','ratioCu_hist','nCompCu_hist', ...
    'cfg','ctx','N','gen','beta','d','-v7.3');
%%
traceFile = fullfile(outDir, ['trace_all_'  tag '.mat']);

S = load(traceFile, 'Ab_hist','ind_hist');   % traceFile 你保存的 trace_all_*.mat
Ab_hist  = S.Ab_hist;
ind_hist = S.ind_hist;

[gen, N, Lbit] = size(Ab_hist);
Ab_rank_hist = zeros(gen, N, Lbit, 'uint8');

for it = 1:gen
    Ab_rank_hist(it,:,:) = Ab_hist(it, ind_hist(it,:), :);
end
save(traceFile, 'Ab_rank_hist', '-append');

%%

save(fullfile(outDir, 'best_bits_coarse.mat'), 'best_bits', 'cfg', 'ctx', 'bestOverall'); %%保数数据

openfemm(1);
opendocument(ctx.baseFemFile);   % 你自己的模板 fem 路径（重要！）

% 把最优基因写入当前 fem
[mat_code, turns_per_cell] = compute_turns_per_cell(best_bits, domain, phase_id_sector, N_phase_total);
femm_apply_design_bits_rep6_inset(best_bits, domain,ctx, ctx.phase_id_sector, ctx.mats, ctx.circNames, ...
    ctx.groupId_core, ctx.groupId_ring,31, turns_per_cell);

% 保存 .fem
mi_saveas(bestFemPath);

% 可选：保存 png（一般要先求解）
mi_analyze;
mi_loadsolution;
mo_savebitmap(bestPngPath);
% %% 临时代码
% S = load('best_result_0204_groupN_100gen.mat.mat', 'seed_bits', 'cfg');
% seed_bits = S.seed_bits;
% save_best_design_femm(seed_bits, ctx, 'seed_bits_fine', 'seed_bits_fine'); %%保存结构图
% 
% %% 临时代码2
% 
% refFemFile = 'model.fem';
% 
% seed_bits = seed_from_reference_model(ctx, refFemFile);
% 
% save('seed_bits_filter.mat', 'seed_bits', 'cfg');
%%
% fprintf('Saved best FEMM model to:\n  %s\n', bestFemPath);
% fprintf('If analyzed, solution .ans will be next to it with same basename.\n');
% % Minimization problem
% x  = valx(1);
% y  = valy(1);
% fx = vfx(end);
%
% % Plot
% figure
% semilogy(vfx)
% title('Minimization')
% xlabel('Iterations')
% ylabel('Best f(x,y)')
% grid on
%
% txt2 = ['F Best: ', num2str(fbest)];
% text(0,1,txt2,'Units','normalized',...
%      'HorizontalAlignment','left','VerticalAlignment','bottom');
%
% txt3 = ['F Found: ', num2str(fx)];
% text(1,1,txt3,'Units','normalized',...
%      'HorizontalAlignment','right','VerticalAlignment','bottom');


%%
% INTERNAL FUNCTIONS

function imprime(PRINT,vx,vy,vz,x,y,fx)

if PRINT == 1
    meshc(vx,vy,vz)
    hold on
    title('Minimization')
    xlabel('x')
    ylabel('y')
    zlabel('f(x,y)')
    plot3(x,y,fx,'k*')
    colormap jet
    drawnow
    hold off
    pause(0.1)
end

end

function [T,pcs] = reprod(N,beta,ind,Ab)

% N	   -> number of clones
% beta -> multiplying factor
% ind  -> best individuals
% Ab   -> antibody population

% T	   -> temporary population
% pcs  -> final position of each clone

T = [];

for i = 1:N
    cs(i) = round(beta*N);
    pcs(i) = sum(cs);
    T = [T; ones(cs(i),1) * Ab(ind(end-i+1),:)];
end

end

function pm = pmcont(pm,pma,pmr,it,itpm)

% pma  -> initial value
% pmr  -> control rate
% itpm -> iterations for restoring

if rem(it,itpm) == 0
    pm = pm * pmr;
    if rem(it,10*itpm) == 0
        pm = pma;
    end
end

end

function z = decode(Ab,varMin,varMax)

% x	-> real value (precision: 6)
% v	-> binary string (length: 22)

Ab = fliplr(Ab);
s = size(Ab);
aux = 0:1:21;
aux = ones(s(1),1)*aux;
x1 = sum((Ab.*2.^aux),2);

% Keeping values between bounds
z = varMin + x1' .* (varMax - varMin)/(2^22 - 1);

end

function Ab = cadeia(n1,s1)

% Antibody (Ab) chains
Ab = randi([0,1], n1, s1);

end

function J = f_wrapper(bits, ctx, W)
t = getCurrentTask();
wid = -1; if ~isempty(t), wid = t.ID; end
fprintf('[%s] worker %d START\n', datestr(now,'HH:MM:SS.FFF'), wid);
% FEMM 评估
S = W.Value;
[J, ~, ~] = eval_design_femm_float(bits, ctx, S);

fprintf('[%s] worker %d END\n', datestr(now,'HH:MM:SS.FFF'), wid);
end


function y = mutate_trit(x, pm)
% trit 变异：0/1/2 随机跳到另外两个值之一
y = x;
M = rand(size(x)) < pm;
if any(M)
    idx = find(M);
    old = y(idx);
    delta = randi([1,2], size(old));
    y(idx) = mod(old + delta, 3);
end
end

function seed_bits = make_seed_bits_from_reference(ctx)
dom = ctx.domain;
Nd  = dom.Nd;

% 假设 dom 里有 x_c, y_c（如果是 r_c 更方便）
r = hypot(dom.x_c, dom.y_c);

rmin = min(r); rmax = max(r);
rho = (r - rmin) / (rmax - rmin + eps); % 0..1

seed_bits = zeros(1,Nd);    % 先全空气(0)

% 中外层设为铁(1)：让它天然连到外侧固定铁
seed_bits(rho > 0.8) = 1;

seed_bits( (rho > 0.2) & (rho < 0.8) ) = 2;

seed_bits(rho < 2) = 1;

% 内层先留空（你后面让算法去"长齿/补铁"）
% seed_bits(rho < 0.25) = 0;

% 如果你想让 seed 更"像图里那样有齿"，可以再加一个角度条纹/扇区规则
% （等你确认 dom 里 theta 的定义，我可以给你写得更贴合）
end


function save_best_design_femm(best_bits, ctx, femPath, pngPath)
% 作用：
%  1) 打开 baseFemFile
%  2) 应用 best_bits 对应的设计（与你 parfor 里一样的构建逻辑）
%  3) mi_saveas 保存为 femPath
%  4) mi_analyze -> 自动生成同名 .ans
%  5) 可选：导出一张位图 png

% 建议：这里用一个"干净的 FEMM 实例"来复现，避免并行 worker 状态影响
openfemm(1);

% 打开基础模型（你 ctx.baseFemFile）
opendocument(ctx.baseFemFile);

% ====== 关键：调用你现有的"应用bits到模型"的函数 ======
% 你在 eval_design_femm_worker 里已经做过同样的事情
% 推荐你把"删旧label + 打新label"的那段封装成一个函数，比如：
% femm_apply_design_bits_rep6(best_bits, ctx.domain, ...)
%
% 这里我假设你已经有 femm_apply_design_bits_rep6：
mi_selectgroup(ctx.groupId_core);  mi_deleteselected();
mi_selectgroup(ctx.groupId_ring);  mi_deleteselected();

domain          = ctx.domain;
phase_id_sector = ctx.phase_id_sector;
N_phase_total   = ctx.N_phase_total;

[mat_code, turns_per_cell] = compute_turns_per_cell(best_bits, domain, phase_id_sector, N_phase_total);
femm_apply_design_bits_rep6_inset(best_bits, domain,ctx, ctx.phase_id_sector, ctx.mats, ctx.circNames, ...
    ctx.groupId_core, ctx.groupId_ring,31, turns_per_cell);

% 保存 .fem
mi_saveas(femPath);

% 求解（会在同目录生成同名 .ans）
% mi_analyze(1);
% mi_loadsolution;

% 可选：导出一张图（磁密云图/矢量图等，取决于你当前显示设置）
% if nargin >= 4 && ~isempty(pngPath)
%     try
%         mo_savebitmap(pngPath);
%     catch
%         % 有些 FEMM 版本/状态下 bitmap 导出会失败，不影响 fem/ans 保存
%     end
% end

% 关闭后处理窗口（可选）
% try, mo_close; end
% try, mi_close; end
% try, closefemm; end
end

%%
function show_design_grid(ctx)
dom = ctx.domain;
nr = ctx.cfg.nr; nt = ctx.cfg.nt;

figure; hold on; axis equal; grid on;
scatter(dom.x_c, dom.y_c, 60, 'k', 'filled');

% 标线性索引 + (r,c)
for k = 1:dom.Nd
    [r,c] = ind2sub([nr,nt], k);
    text(dom.x_c(k), dom.y_c(k), sprintf('%d\n(%d,%d)', k, r, c), ...
        'FontSize', 8, 'HorizontalAlignment','center','VerticalAlignment','middle');
end
title('Design grid cell centers with indices');
xlabel('x'); ylabel('y');
end

function seed_bits = paint_seed_bits(ctx)
dom = ctx.domain;
nr = ctx.cfg.nr; nt = ctx.cfg.nt;

% 0=air,1=iron,2=copper
B = ones(nr, nt);               % 先全空气
seed_bits = B(:).';

fig = figure; hold on; axis equal; grid on;
title('Left click: toggle cell (0->1->2->0), Right click: finish');
xlabel('x'); ylabel('y');

while true
    clf(fig); hold on; axis equal; grid on;
    % 画当前状态
    % 画当前状态（固定颜色：Air=白, Iron=灰, Copper=橙）
    val = seed_bits(:);

    idx0 = (val==0);
    idx1 = (val==1);
    idx2 = (val==2);

    scatter(dom.x_c(idx0), dom.y_c(idx0), 160, 'blue', 'filled'); hold on
    scatter(dom.x_c(idx1), dom.y_c(idx1), 160, [0.5 0.5 0.5], 'filled');
    scatter(dom.x_c(idx2), dom.y_c(idx2), 160, [1.0 0.6 0.0], 'filled');

    legend({'Air(0)','Iron(1)','Copper(2)'}, 'Location','bestoutside');
    colormap(parula); caxis([0 2]); colorbar;
    drawnow;

    [x,y,button] = ginput(1);
    if isempty(button) || button ~= 1
        break; % 右键/回车结束
    end

    % 找最近的格子
    [~,k] = min((dom.x_c-x).^2 + (dom.y_c-y).^2);

    % 翻转 0->1->2->0
    seed_bits(k) = mod(seed_bits(k)+1, 3);
end

% 最后做一次悬浮铁修复（和你评估一致）
% seed_bits = remove_floating_iron(seed_bits, ctx.cfg, ctx.ironCode, ctx.airCode, ctx.thetaPeriodic);
end

function seed_bits = seed_from_reference_model(ctx, refFemFile)
% 从参考模型(refFemFile)采样每个design cell中心点，生成 seed_bits
% 规则：
%   copper: abs(J) > J_thr   -> 2
%   iron  : mu_r > mu_thr    -> 1
%   air   : else             -> 0

dom = ctx.domain;
Nd  = dom.Nd;

J_thr  = 1e-3;
mu_thr = 5;

openfemm;
opendocument(refFemFile);

% 必须求解并加载解
mi_analyze(1);
mi_loadsolution;

seed_bits = zeros(1, Nd);

for k = 1:Nd
    pv = mo_getpointvalues(dom.x_c(k), dom.y_c(k));

    Jz   = pv(9);
    mur  = pv(10);   % pv(10)=pv(11)，取一个即可

    if abs(Jz) > J_thr
        seed_bits(k) = 2;          % copper
    elseif mur > mu_thr
        seed_bits(k) = 1;          % iron
    else
        seed_bits(k) = 0;          % air
    end
end

% 可选：清理浮铁（只保留与外圈连通的铁）
seed_bits = remove_floating_iron(seed_bits, ctx.cfg, ctx.ironCode, ctx.airCode, ctx.thetaPeriodic);

try, mo_close; end
try, mi_close; end
try, closefemm; end
end


function dump_pv(x,y)
pv = mo_getpointvalues(x,y);
fprintf('pv length = %d\n', numel(pv));
for i = 1:numel(pv)
    fprintf('%2d : %.12g\n', i, pv(i));
end
end


function out = detect_floating_iron(bits, cfg, ironCode, thetaPeriodic)
% out: struct with fields
%   nIron, nFloat, ratio, keepMask(optional)

if nargin < 4, thetaPeriodic = false; end
nr = cfg.nr; nt = cfg.nt;

B = reshape(bits, [nt, nr]).';        % [nr,nt]
isIron = (B == ironCode);

out.nIron = nnz(isIron);
if out.nIron == 0
    out.nFloat = 0;
    out.ratio  = 0;
    out.keepMask = false(nr,nt);
    return;
end

% anchor: outer ring (back-yoke side)
support = false(nr, nt);
support(nr,:) = true;

seeds = find(isIron & support);

% if no anchored iron, then all iron is floating
if isempty(seeds)
    out.nFloat = out.nIron;
    out.ratio  = 1;
    out.keepMask = false(nr,nt);
    return;
end

% BFS from anchored iron
keep = false(nr, nt);
q = seeds(:);
keep(q) = true;

while ~isempty(q)
    idx = q(1); q(1) = [];
    [r,c] = ind2sub([nr,nt], idx);

    nb = [r-1 c; r+1 c; r c-1; r c+1];
    for k = 1:4
        rr = nb(k,1); cc = nb(k,2);

        if thetaPeriodic
            if cc < 1,  cc = nt; end
            if cc > nt, cc = 1;  end
        end

        if rr<1 || rr>nr || cc<1 || cc>nt, continue; end

        if isIron(rr,cc) && ~keep(rr,cc)
            keep(rr,cc) = true;
            q(end+1,1) = sub2ind([nr,nt], rr, cc);
        end
    end
end

floating = isIron & ~keep;

out.nFloat = nnz(floating);
out.ratio  = out.nFloat / (out.nIron + eps);
out.keepMask = keep;   % 可选：想省内存可删掉这个字段
end

function out = detect_floating_copper_largest(bits, cfg, cuCode, thetaPeriodic)
% out fields:
%   nCu, nFloat, ratio, nComp, compSizes

if nargin < 4, thetaPeriodic = false; end
nr = cfg.nr; nt = cfg.nt;

B = reshape(bits, [nt, nr]).';        % [nr,nt]
isCu = (B == cuCode);

out.nCu = nnz(isCu);
if out.nCu == 0
    out.nFloat = 0;
    out.ratio  = 0;
    out.nComp  = 0;
    out.compSizes = [];
    return;
end

visited = false(nr,nt);
labels  = zeros(nr,nt,'uint16');
compSizes = zeros(0,1);
cid = uint16(0);

for r0 = 1:nr
    for c0 = 1:nt
        if ~isCu(r0,c0) || visited(r0,c0), continue; end

        cid = cid + 1;
        q = sub2ind([nr,nt], r0, c0);
        visited(r0,c0) = true;
        labels(r0,c0) = cid;
        cnt = 0;

        while ~isempty(q)
            idx = q(1); q(1) = [];
            [r,c] = ind2sub([nr,nt], idx);
            cnt = cnt + 1;

            nb = [r-1 c; r+1 c; r c-1; r c+1];
            for k = 1:4
                rr = nb(k,1); cc = nb(k,2);

                if thetaPeriodic
                    if cc < 1,  cc = nt; end
                    if cc > nt, cc = 1;  end
                end

                if rr<1 || rr>nr || cc<1 || cc>nt, continue; end

                if isCu(rr,cc) && ~visited(rr,cc)
                    visited(rr,cc) = true;
                    labels(rr,cc)  = cid;
                    q(end+1,1) = sub2ind([nr,nt], rr, cc);
                end
            end
        end

        compSizes(double(cid),1) = cnt; %#ok<AGROW>
    end
end

out.nComp = numel(compSizes);
out.compSizes = compSizes;

% keep only the largest component
[~, imax] = max(compSizes);
mainMask = (labels == uint16(imax));
floatMask = isCu & ~mainMask;

out.nFloat = nnz(floatMask);
out.ratio  = out.nFloat / (out.nCu + eps);
end

function key = bits_key(bits01_2)
% bits01_2: 1×Lbit, uint8/double 都行
    b = uint8(bits01_2);
    key = char(b + uint8('0'));   % 0/1/2 -> '0''1''2'
end

function [J, cache] = eval_bits_batch_cached(X, p, f, cache)
% X: K×Lbit (double/uint8)
% p: parallel pool
% f: function handle, f(bits)->J
% cache: containers.Map  key(char) -> double(J)

    X8 = uint8(X);

    % --- 1) 批内去重 ---
    [U, ~, ic] = unique(X8, 'rows', 'stable');   % U: Ku×Lbit
    Ku = size(U,1);

    Ju = zeros(Ku,1);

    % --- 2) 先查缓存，没命中的才提交 ---
    F = parallel.FevalFuture.empty(0,1);
    fut2u = zeros(0,1);
    futKey = cell(0,1);

    for u = 1:Ku
        key = char(U(u,:) + uint8('0'));

        if isKey(cache, key)
            Ju(u) = cache(key);
        else
            % 注意：传给 f 的类型要和你 f 期望一致（你现在 bits 是 double 也行）
            F(end+1,1) = parfeval(p, f, 1, double(U(u,:)));
            fut2u(end+1,1) = u;
            futKey{end+1,1} = key;
        end
    end

    % --- 3) 回收并写回缓存 ---
    for k = 1:numel(F)
        [idx, val] = fetchNext(F);
        u = fut2u(idx);
        Ju(u) = val;
        cache(futKey{idx}) = val;
    end

    % --- 4) 映射回原始顺序 ---
    J = Ju(ic);
end