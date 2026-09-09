function out = detect_floating_copper_small_islands(bits, cfg, cuCode, thetaPeriodic, Amin)
% 只把"面积 < Amin"的铜连通分量计为悬浮铜（小铜岛）
% 防御式 BFS：避免重复计数，并防止队列出现非法索引

if nargin < 4, thetaPeriodic = false; end
if nargin < 5, Amin = 4; end

nr = cfg.nr; nt = cfg.nt;
Nd = nr * nt;

% ---- 关键自检：bits 长度必须匹配网格 ----
assert(numel(bits) == Nd, 'bits length mismatch: got %d, expected %d (=nr*nt)', numel(bits), Nd);

% bits -> [nr,nt]
B = reshape(bits(:), [nt, nr]).';   % 强制 bits(:) 避免行列向量坑
isCu = (B == cuCode);

out.nCu = nnz(isCu);
if out.nCu == 0
    out.nFloat = 0;
    out.ratio  = 0;
    out.nComp  = 0;
    out.compSizes = [];
    out.nSmallComp = 0;
    return;
end

visited = false(nr,nt);
compSizes = zeros(0,1);

cid = 0;
nFloat = 0;
nSmall = 0;

for r0 = 1:nr
    for c0 = 1:nt
        if ~isCu(r0,c0) || visited(r0,c0)
            continue;
        end

 % ---- BFS for one component ----
cid = cid + 1;

q = zeros(out.nCu, 1, 'uint32');  % 预分配上限（最多nCu个点）
head = 1; tail = 1;

q(tail) = uint32(sub2ind([nr,nt], r0, c0));
cnt = 0;

while head <= tail
    idx = double(q(head)); head = head + 1;

    % 防御：idx 必须 1..Nd
    if idx < 1 || idx > Nd
        error('Queue corrupted: idx=%d (Nd=%d).', idx, Nd);
    end

    [r,c] = ind2sub([nr,nt], idx);

    if visited(r,c), continue; end
    visited(r,c) = true;
    cnt = cnt + 1;

    nb = [r-1 c; r+1 c; r c-1; r c+1];
    for k = 1:4
        rr = nb(k,1); cc = nb(k,2);

        if thetaPeriodic
            if cc < 1,  cc = nt; end
            if cc > nt, cc = 1;  end
        end

        if rr<1 || rr>nr || cc<1 || cc>nt
            continue;
        end

        if isCu(rr,cc) && ~visited(rr,cc)
            tail = tail + 1;
            q(tail) = uint32(sub2ind([nr,nt], rr, cc));
        end
    end
end
        compSizes(cid,1) = cnt; %#ok<AGROW>

        if cnt < Amin
            nFloat = nFloat + cnt;
            nSmall = nSmall + 1;
        end
    end
end

out.nComp = cid;
out.compSizes = compSizes;
out.nFloat = nFloat;
out.ratio  = nFloat / (out.nCu + eps);
out.nSmallComp = nSmall;

% ---- 守恒自检 ----
if sum(compSizes) ~= out.nCu
    error('compSizes inconsistent: sum=%d, nCu=%d', sum(compSizes), out.nCu);
end
end