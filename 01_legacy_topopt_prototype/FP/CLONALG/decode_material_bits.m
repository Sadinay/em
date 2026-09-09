function mat_code = decode_material_bits(bits, Nd)
% gene: 1×Nd 或 Nd×1，取值 0/1/2
% mat_code: Nd×1，0=Air,1=Iron,2=Copper

    if numel(bits) ~= Nd
        error('gene length %d != Nd %d', numel(bits), Nd);
    end
    mat_code = bits(:);

    if any(mat_code < 0 | mat_code > 2)
        error('gene must be in {0,1,2}.');
    end
end