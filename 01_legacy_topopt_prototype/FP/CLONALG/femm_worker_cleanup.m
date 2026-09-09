function femm_worker_cleanup(S)
% 每个 worker 退出时清理（可选）
    try
        closefemm;
    catch
    end
end
