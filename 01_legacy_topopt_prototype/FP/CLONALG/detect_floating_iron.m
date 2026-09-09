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
