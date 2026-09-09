function [T_avg, T_ripple] = scantorque(inp)

    rotor_group = 1;      % 你把转子所有 block 都设成 group=1 了
    innerIndex  = 10;     % ia = Inner Angle, Deg 在 mi_addboundprop 里的索引是 10

    dtheta    = 3;        % 每步 1°
    maxAngle  = 15;       % 扫 0~90°
    theta_deg = 0:dtheta:maxAngle;
    nSteps    = numel(theta_deg);
    T         = zeros(nSteps,1);

delta_e_deg = 90;      % 电角度电流相位（90° ≈ 只在 q 轴上，表贴机型大致最大转矩）
delta_e = delta_e_deg*pi/180;

    % 先保存一次。
    % mi_saveas('temp_airgap_scan.fem');

    % --- 2) 扫描角度 ---
    for k = 1:nSteps
        th = theta_deg(k);          % 机械角，单位：deg

        % 2.1 只改 Air gap 边界的 Inner Angle = 转子角度
        mi_modifyboundprop('Air gap', innerIndex, th);
        % Outer Angle(11) 一直保持 0，不用动

            % ===== 新增：根据当前转子位置设置三相电流 =====
    theta_e = 4 * th * pi/180 + delta_e;  % 电角度（rad）

    Ia = inp * cos(theta_e);
    Ib = inp * cos(theta_e - 2*pi/3);
    Ic = inp * cos(theta_e - 4*pi/3);

    % 写入三相电路电流（propID=0 表示电流） //这里修改为了带+-的，所以不能再给参考模型的脚本使用
    mi_modifycircprop('A+', 1, Ia);
    mi_modifycircprop('A-', 1, -Ia);
    mi_modifycircprop('B+', 1, Ib);
    mi_modifycircprop('B-', 1, -Ib);
    mi_modifycircprop('C+', 1, Ic);
    mi_modifycircprop('C-', 1, -Ic);
        % 2.2 求解
        mi_analyze(1);
        mi_loadsolution;
        % mi_zoomnatural;           % 自动缩放到合适大小

        % 2.3 在转子组上做转矩积分
        mo_groupselectblock(rotor_group);
        T(k) = mo_blockintegral(22);    % 22 = Torque
        mo_clearblock;
        mo_close;
    end

    % % --- 3) 简单画图 ---
    % figure;
    % plot(theta_deg, T, '-o');
    % xlabel('Mechanical angle / deg');
    % ylabel('Electromagnetic torque / Nm');
    % grid on;
    % title('Loaded torque vs. rotor position (Air-gap BC)');

    % closefemm;   % 需要的话自己决定什么时候关

    T_avg    = mean(T);
    T_max    = max(T);
    T_min    = min(T);
    T_ripple = (T_max - T_min)/max(abs(T_avg), 1e-6);
end