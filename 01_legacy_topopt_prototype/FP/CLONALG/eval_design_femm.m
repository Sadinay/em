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

%     % ==== 统一放粗：设计域网格线段/弧段 ====
% groupGrid = 3;
% hseg      = 8;     % mm
% maxsegdeg = 15;    % deg
% 
% mi_clearselected();
% mi_selectgroup(groupGrid);
% 
% % 线段：elementsize=hseg, automesh=0
% mi_setsegmentprop('', hseg, 0, 0, groupGrid);
% 
% % 圆弧：maxsegdeg 控制圆弧离散；elementsize 控制弧段网格
% mi_setarcsegmentprop(maxsegdeg, hseg, 0, groupGrid);
% 
% mi_clearselected();


    % （如果模板里还有旧的设计块，保险起见先删）
    mi_selectgroup(groupId);
    mi_deleteselected();

    % ---------- 2. 根据基因填充 6 个 15° 设计域 ----------
    femm_apply_design_bits_rep6_inset_merge_cu_island_inset_v5_corner_tri(bits, domain,ctx, ctx.phase_id_sector, ctx.mats, ctx.circNames, ...
                             ctx.groupId_core, ctx.groupId_ring,31, turns_per_cell);
    

        % ----------设置三相电流（示例：单工况） ----------
    % % 这里可以把电角度当成一个参数，也可以固定在某个工况
    % theta_e = 0;                         % 电角（自己以后可以扫一圈）
    % 
    % Ipk = 10;                            % 相电流峰值（示意）
    % Ia  = Ipk * sin(theta_e);
    % Ib  = Ipk * sin(theta_e - 2*pi/3);
    % Ic  = Ipk * sin(theta_e - 4*pi/3);
    % 
    % % 先删旧的 circprop 再加（视模板情况而定）
    % % 这里假定 baseFemFile 里已经定义好了 A+/A-/B+/B-/C+/C- 的 circprop 名字，
    % % 只需要更新电流值：
    % mi_modifycircprop('A+', 1, Ia);
    % mi_modifycircprop('A-', 1, -Ia);
    % mi_modifycircprop('B+', 1, Ib);
    % mi_modifycircprop('B-', 1, -Ib);
    % mi_modifycircprop('C+', 1, Ic);
    % mi_modifycircprop('C-', 1, -Ic);

    % 视情况：是否要重新划分网格参数
    % mi_setgrid, mi_smartmesh(0) 等，按你原来的脚本来

    % --- 每次评估保存 ---
% mi_saveas('test.fem');
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

    J = J_ripple + penalty;

    % ---------- 5. 可以关闭 FEMM 窗口（视情况而定） ----------
    % closefemm;   % 如果你希望每次都关掉的话

end
