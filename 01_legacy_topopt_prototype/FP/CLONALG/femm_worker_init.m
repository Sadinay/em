function S = femm_worker_init(ctx)
% 每个 worker 初始化一次
% S: worker 的私有状态（临时目录、模板副本路径、计数器等）

    % 给 worker 建独立临时目录
    baseTmp = fullfile(tempdir, 'femm_topopt_workers');
    if ~exist(baseTmp,'dir'), mkdir(baseTmp); end

    wid = get_worker_id();
    S.tmpDir = fullfile(baseTmp, sprintf('w%02d', wid));
    if ~exist(S.tmpDir,'dir'), mkdir(S.tmpDir); end

    % 拷贝模板到 worker 私有目录（避免 ans 覆盖）
    [~,name,ext] = fileparts(ctx.baseFemFile);
    S.baseFemLocal = fullfile(S.tmpDir, sprintf('%s_w%02d%s', name, wid, ext));
    copyfile(ctx.baseFemFile, S.baseFemLocal);

    S.caseCounter = 0;

    % 打开 FEMM（尽量不弹窗）
    openfemm(1);          % 如果你这边 openfemm(0) 不生效，就用 openfemm(1)+hidefemm
    try
        hidefemm;         % FEMM mfiles 里一般有 hidefemm
    catch
    end

    opendocument(S.baseFemLocal);
    % ==== 统一放粗：设计域网格线段/弧段 ====
groupGrid   = 3;     % 你自己定，但要和建模时一致
hseg        = 8;      % 线段最大边长（mm），你可以 5~12 之间调
maxsegdeg   = 10;     % 圆弧分段角度（deg），越大越粗，比如 10~20

mi_clearselected();
mi_selectgroup(groupGrid);
mi_setsegmentprop('', 0, hseg, 0, groupGrid);   % automesh=0 强制用 hseg
mi_clearselected();

mi_selectgroup(groupGrid);
mi_setarcsegmentprop(maxsegdeg, '', 0, groupGrid);
mi_clearselected();

mi_clearselected();
mi_saveas(S.baseFemLocal);
end
