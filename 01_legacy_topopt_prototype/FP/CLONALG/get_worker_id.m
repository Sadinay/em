function wid = get_worker_id()
% parfor 里取 worker id；不在并行时返回 0
    t = getCurrentTask();
    if isempty(t)
        wid = 0;
    else
        wid = t.ID;  % 足够用来区分 worker
    end
end
