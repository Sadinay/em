function [J, T_avg, T_ripple] = eval_design_femm_worker(bits, ctx, S)

    % ---------- 0) 早退：小悬浮岛直接判死刑（不进 FEMM） ----------
    if isfield(ctx,'fastReject') && ctx.fastReject.enable
        % 铜：小岛（Amin=4）
        cu = detect_floating_copper_small_islands(bits, ctx.cfg, ctx.cuCode, ctx.thetaPeriodic, ctx.Amin);

        % 铁：如果你只有 detect_floating_iron（没面积阈值），先按 nFloat 判死刑
        fe = detect_floating_iron(bits, ctx.cfg, ctx.ironCode, ctx.thetaPeriodic);

        if (cu.nFloat > 0) || (fe.nFloat > 0)
            J = ctx.fastReject.badJ ...
                + ctx.fastReject.wCu * double(cu.nFloat) ...
                + ctx.fastReject.wFe * double(fe.nFloat);
            T_avg = 0; 
            T_ripple = 0;
            return;
        end
    end

    domain          = ctx.domain;
    phase_id_sector = ctx.phase_id_sector;
    N_phase_total   = ctx.N_phase_total;

    [mat_code, turns_per_cell] = compute_turns_per_cell(bits, domain, phase_id_sector, N_phase_total);

    if all(turns_per_cell == 0)
        J = 1e6; T_avg = 0; T_ripple = 0;
        return;
    end

    % ====== [新增] 每次评估都从干净模板开始（防止删线累积） ======
    try, mo_close; end           % 若上次开过解窗口
    try, mi_close; end           % 关闭当前 fem 文档（不关 FEMM）
    opendocument(S.baseFemLocal); % 打开干净模板（强烈建议 worker 私有拷贝）
    mi_clearselected;

    mi_selectgroup(ctx.groupId_core);
    mi_deleteselected();
    mi_selectgroup(ctx.groupId_ring);
    mi_deleteselected();

    femm_apply_design_bits_rep6_inset(bits, domain,ctx, ctx.phase_id_sector, ctx.mats, ctx.circNames, ...
                             ctx.groupId_core, ctx.groupId_ring,31, turns_per_cell);
  
    % 强烈建议：每次分析前把模型保存到 worker 私有目录、并且文件名唯一
% --- 每次评估保存到唯一case（并行安全） ---
femCase = fullfile(S.tmpDir, ['case_' char(java.util.UUID.randomUUID) '.fem']);
mi_saveas(femCase);

    % scantorque 里如果自己也 mi_saveas / 导出文件，请同样改成用 femCase 前缀
    [T_avg, T_ripple] = scantorque(3.5);

    J_ripple = T_ripple;
    if T_avg < ctx.T_min
        penalty = ctx.penaltyCoef * (ctx.T_min - T_avg)/ctx.T_min;
    else
        penalty = 0;
    end
    J_em = J_ripple + penalty;


% ===== 新增：浮铁/浮铜检测 + sigmoid 惩罚 =====
pFe = 0; pCu = 0;

if isfield(ctx,'penFloating') && ctx.penFloating.enable
    % 1) 检测悬浮铁/铜数量（这两个函数你已经验证可用）
    fe = detect_floating_iron(bits, ctx.cfg, ctx.ironCode, false);
    cu = detect_floating_copper_small_islands(bits, ctx.cfg, ctx.cuCode, false,ctx.Amin);

    nFloatFe = fe.nFloat;
    nFloatCu = cu.nFloat;

    % 3) sigmoid 惩罚（方案A：count-based）
    pFe = ctx.penFloating.wFe * (1 / (1 + exp(-ctx.penFloating.kFe * (double(nFloatFe) - 0.5))));
    pCu = ctx.penFloating.wCu * (1 / (1 + exp(-ctx.penFloating.kCu * (double(nFloatCu) - 0.5))));
end

J = J_em + pFe + pCu;

    % ===== 清理临时文件 =====
try
    delete(femCase);                       % 删除 .fem
    ansCase = [femCase(1:end-4) '.ans'];   % 同名 .ans
    if exist(ansCase,'file'), delete(ansCase); end
catch
end
end
