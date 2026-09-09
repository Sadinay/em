function out = detect_floating_copper_largest(bits, cfg, cuCode, thetaPeriodic)
% out fields:
%   nCu, nFloat, ratio, nComp, compSizes

if nargin < 4, thetaPeriodic = false; end
nr = cfg.nr; nt = cfg.nt;

B = reshape(bits, [nt, nr]).';        % [nr,nt]
isCu = (B == cuCode);

out.nCu = nnz(isCu);
if out.nCu == 0
    out.nFloat = 0;
    out.ratio  = 0;
    out.nComp  = 0;
    out.compSizes = [];
    return;
end

visited = false(nr,nt);
labels  = zeros(nr,nt,'uint16');
compSizes = zeros(0,1);
cid = uint16(0);

for r0 = 1:nr
    for c0 = 1:nt
        if ~isCu(r0,c0) || visited(r0,c0), continue; end

        cid = cid + 1;
        q = sub2ind([nr,nt], r0, c0);
        visited(r0,c0) = true;
        labels(r0,c0) = cid;
        cnt = 0;

        while ~isempty(q)
            idx = q(1); q(1) = [];
            [r,c] = ind2sub([nr,nt], idx);
            cnt = cnt + 1;

            nb = [r-1 c; r+1 c; r c-1; r c+1];
            for k = 1:4
                rr = nb(k,1); cc = nb(k,2);

                if thetaPeriodic
                    if cc < 1,  cc = nt; end
                    if cc > nt, cc = 1;  end
                end

                if rr<1 || rr>nr || cc<1 || cc>nt, continue; end

                if isCu(rr,cc) && ~visited(rr,cc)
                    visited(rr,cc) = true;
                    labels(rr,cc)  = cid;
                    q(end+1,1) = sub2ind([nr,nt], rr, cc);
                end
            end
        end

        compSizes(double(cid),1) = cnt; %#ok<AGROW>
    end
end

out.nComp = numel(compSizes);
out.compSizes = compSizes;

% keep only the largest component
[~, imax] = max(compSizes);
mainMask = (labels == uint16(imax));
floatMask = isCu & ~mainMask;

out.nFloat = nnz(floatMask);
out.ratio  = out.nFloat / (out.nCu + eps);
end
