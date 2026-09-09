function [J, T_avg, T_ripple] = eval_design_femm(bits, ctx)
% bits : 1×(2*Nd) 基因串（2bit/单元）
% ctx  : 上面那个结构体

    domain          = ctx.domain;
    sector_offsets  = ctx.sector_offsets_deg;
    phase_id_sector = ctx.phase_id_sector;
    mats            = ctx.mats;
    circNames       = ctx.circNames;
    groupId         = ctx.groupId_design;
    N_phase_total   = ctx.N_phase_total;
    % ----------匝数分配 ----------
    [mat_code, turns_per_cell] = compute_turns_per_cell( ...
        bits, domain, phase_id_sector, N_phase_total);

    % 防止"所有绕组都没铜"的极端情况
    if all(turns_per_cell == 0)
        J       = 1e6;      % 给个巨大惩罚
        T_avg   = 0;
        T_ripple= 0;
        return;
    end

    % ---------- 1. 打开 FEMM 模板 ----------
openfemm(1);

% --- 每次评估用临时副本，避免污染原模板 ---
tmpFem = fullfile(tempdir, sprintf('tmp_eval_%s_%06d.fem', datestr(now,'yyyymmdd_HHMMSSFFF'), randi(1e6)));
copyfile(ctx.baseFemFile, tmpFem);

opendocument(tmpFem);


    % （如果模板里还有旧的设计块，保险起见先删）
    mi_selectgroup(groupId);
    mi_deleteselected();

    % ---------- 2. 根据基因填充 6 个 15° 设计域 ----------
    femm_apply_design_bits_rep6_inset(bits, domain,ctx, ctx.phase_id_sector, ctx.mats, ctx.circNames, ...
                             ctx.groupId_core, ctx.groupId_ring,31, turns_per_cell);
    
    % ---------- 3. 运行一次"转矩扫描" ----------

    [T_avg, T_ripple] = scantorque(3.5);

    % ---------- 4. 计算目标函数（越小越好） ----------
    % 例：希望减小 (T_ripple/T_avg)，同时保证 T_avg >= T_min
    J_ripple = T_ripple;

    if T_avg < ctx.T_min
        penalty = ctx.penaltyCoef * (ctx.T_min - T_avg)/ctx.T_min;
    else
        penalty = 0;
    end

    J_em = J_ripple + penalty;

    % ===== 新增：浮铁/浮铜检测 + sigmoid 惩罚 =====

    % 1) 检测悬浮铁/铜数量（这两个函数你已经验证可用）
    fe = detect_floating_iron(bits, ctx.cfg, ctx.ironCode, false);
    cu = detect_floating_copper_small_islands(bits, ctx.cfg, ctx.cuCode, false,ctx.Amin);

    nFloatFe = fe.nFloat;
    nFloatCu = cu.nFloat;

    % 3) sigmoid 惩罚（方案A：count-based）
    pFe = ctx.penFloating.wFe * (1 / (1 + exp(-ctx.penFloating.kFe *  (double(nFloatFe) - 0.5))));
    pCu = ctx.penFloating.wCu * (1 / (1 + exp(-ctx.penFloating.kCu * (double(nFloatCu) - 0.5))));

J = J_em + pFe + pCu;

    % ---------- 5. 可以关闭 FEMM 窗口（视情况而定） ----------
    % closefemm;   % 如果你希望每次都关掉的话

end
