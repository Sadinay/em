function [mat_code, turns_per_cell] = compute_turns_per_cell(bits, domain, ...
                                      phase_id_sector, N_phase_total)
% bits           : 1×(Nd)，3bit/格子
% domain.Nd      : 设计单元数
% phase_id_sector: {6×1 cell}，每个 Nd×1，取值 0~6（0 = 非绕组）
% N_phase_total  : 1×nPhase，每个电路（A+/A-/...）的总匝数
%
% mat_code       : Nd×1，0=Air,1=Iron,2=Copper,3=预留
% turns_per_cell : 1×nPhase，每个电路下"每个铜格子的匝数"

    Nd       = domain.Nd;
    mat_code = decode_material_bits(bits, Nd);

    nPhase  = numel(N_phase_total);
    nSector = numel(phase_id_sector);

    % 统计每个电路下的铜格子数量
    nCell_phase = zeros(1, nPhase);
    for s = 1:nSector
        pid = phase_id_sector{s};    % Nd×1
        for p = 1:nPhase
            mask_p = (mat_code == 2) & (pid == p);
            nCell_phase(p) = nCell_phase(p) + sum(mask_p);
        end
    end

    % 把总匝数平均分配到各个铜格子
    turns_per_cell = zeros(1, nPhase);
    for p = 1:nPhase
        if nCell_phase(p) > 0
            turns_per_cell(p) = N_phase_total(p) / nCell_phase(p);
        else
            turns_per_cell(p) = 0;
        end
    end
end
