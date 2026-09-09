"""Read-only audit of the supplied MAT; writes evidence beside this script."""
from __future__ import annotations

import hashlib
import io
import json
import math
import re
import struct
import warnings
import zlib
from pathlib import Path

import numpy as np
import scipy
from scipy.io import loadmat
from scipy.io.matlab._mio5 import MatFile5Reader

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parents[1]
MAT = PROJECT / 'data_zone/raw/workspace_200.mat'
FEM = PROJECT / 'femm_zone/models/SPMSM_discrete.fem'
raw = MAT.read_bytes()
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    mat = loadmat(MAT, squeeze_me=True, struct_as_record=False)

evidence = {
    'mat_path': str(MAT), 'mat_size_bytes': len(raw),
    'mat_sha256': hashlib.sha256(raw).hexdigest(),
    'mat_header': mat['__header__'].decode('ascii'),
    'scipy_version': scipy.__version__,
    'load_warnings': [str(w.message) for w in caught],
    'top_level_names': [k for k in mat if not k.startswith('__')],
    'inp_field_names': mat['inp']._fieldnames,
    'draw_field_names': mat['draw']._fieldnames,
    'field_paths': [], 'angle_related_fields': [], 'numeric_29_matches': [],
    'function_handles': [], 'opaque_objects': [], 'unsupported': [],
}


def walk(value, path):
    if type(value).__name__ == 'MatlabFunction':
        evidence['function_handles'].append(path)
    if type(value).__name__ == 'MatlabOpaque':
        evidence['opaque_objects'].append(path)
    if hasattr(value, '_fieldnames'):
        for key in value._fieldnames:
            child = getattr(value, key)
            subpath = path + '.' + key
            evidence['field_paths'].append(subpath)
            if re.search(r'ang|schift|shift|theta|omega', key, re.I):
                a = np.asarray(child)
                evidence['angle_related_fields'].append({
                    'path': subpath, 'value': a.tolist() if a.size <= 30 else str(a.shape)})
            walk(child, subpath)
    elif isinstance(value, np.ndarray) and value.dtype.names:
        for i, record in enumerate(value.flat):
            for key in value.dtype.names:
                subpath = f'{path}[{i}].{key}'
                evidence['field_paths'].append(subpath)
                walk(record[key], subpath)
    elif isinstance(value, np.ndarray) and value.dtype.hasobject:
        for i, child in enumerate(value.flat):
            walk(child, f'{path}[{i}]')
    elif isinstance(value, (np.ndarray, np.generic, float, int, complex)):
        a = np.asarray(value)
        if a.dtype.kind in 'biufc' and a.size:
            hits = np.flatnonzero(np.isclose(a.reshape(-1), 29, rtol=0, atol=1e-12))
            if hits.size:
                evidence['numeric_29_matches'].append({
                    'path': path, 'shape': list(a.shape), 'count': int(hits.size),
                    'first_flat_indices_zero_based': hits[:8].tolist()})
        elif a.dtype.kind not in 'USbiufc':
            evidence['unsupported'].append({'path': path, 'type': str(a.dtype)})
    elif not isinstance(value, (str, bytes)):
        evidence['unsupported'].append({'path': path, 'type': type(value).__name__})


for name, value in mat.items():
    if not name.startswith('__'):
        walk(value, name)

# Decode the nested MAT records inside the saved function subsystem, recursively.
def decode_subsystem(blob, path):
    stream = io.BytesIO(blob)
    stream.seek(8)
    reader = MatFile5Reader(stream, byte_order='<', squeeze_me=True, struct_as_record=False)
    reader.initialize_read()
    records = []
    while not reader.end_of_stream():
        header, end = reader.read_var_header()
        value = reader.read_var_array(header)
        subpath = f'{path}.record{len(records)}'
        records.append({'name': header.name.decode('ascii'), 'mclass': header.mclass})
        walk(value, subpath)
        if isinstance(value, np.ndarray) and value.dtype == np.uint8:
            child = value.tobytes()
            if child[:8] == b'\x00\x01IM\x00\x00\x00\x00':
                records[-1]['nested_records'] = decode_subsystem(child, subpath)
        stream.seek(end)
    return records


subsystem = mat['__function_workspace__'].tobytes()
evidence['function_subsystem'] = {
    'bytes': len(subsystem), 'sha256': hashlib.sha256(subsystem).hexdigest(),
    'records': decode_subsystem(subsystem, 'function_subsystem'),
    'printable_strings': [s.decode('ascii') for s in re.findall(rb'[ -~]{4,}', subsystem)],
    'limitation': 'MCOS containers and nested MAT records inspected; no claim of full MATLAB runtime deserialization.',
}
handle = mat['sigmoid_func'].item().function_handle
evidence['sigmoid_handle'] = {'function': handle.function, 'type': handle.type, 'file': handle.file}

# Independently decompress every physical top-level record and search the bytes.
# This also searches opaque payloads, without treating matches as MATLAB semantics.
assert raw[126:128] == b'IM'
needles = ('ang_0', 'ang0', 'ang_r', 'innerangle', 'inner_angle', 'initial_rotor',
           'sliding_airgap', 'mi_modifyboundprop', 'rotorschift')
records, cursor = [], 128
while cursor < len(raw):
    tag, size = struct.unpack_from('<II', raw, cursor)
    assert tag in (14, 15), (cursor, tag)
    end = cursor + 8 + size
    assert end <= len(raw)
    payload = raw[cursor + 8:end]
    if tag == 15:
        decoder = zlib.decompressobj()
        payload = decoder.decompress(payload) + decoder.flush()
        assert decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail
    lowered = payload.lower()
    matches = {needle: sum(lowered.count(needle.encode(enc))
                           for enc in ('ascii', 'utf-16le', 'utf-16be'))
               for needle in needles}
    records.append({'file_offset': cursor, 'tag': tag, 'uncompressed_bytes': len(payload),
                    'token_hits': {k: v for k, v in matches.items() if v}})
    cursor = end if tag == 15 else end + (-size % 8)
assert cursor == len(raw)
evidence['raw_record_audit'] = {
    'record_count': len(records), 'reached_exact_file_end': cursor == len(raw),
    'uncompressed_bytes_total': sum(r['uncompressed_bytes'] for r in records),
    'token_hits': {n: sum(r['token_hits'].get(n, 0) for r in records) for n in needles},
    'records': records,
}

fem = FEM.read_text()
gap = next(b for b in re.findall(r'<BeginBdry>.*?<EndBdry>', fem, re.S)
           if '<BdryName> = "sliding_airgap"' in b)
evidence['original_fem'] = {
    'path': str(FEM), 'sha256': hashlib.sha256(FEM.read_bytes()).hexdigest(),
    'innerangle': float(re.search(r'<innerangle>\s*=\s*(\S+)', gap)[1]),
    'outerangle': float(re.search(r'<outerangle>\s*=\s*(\S+)', gap)[1]),
}
assert hashlib.sha256(MAT.read_bytes()).hexdigest() == evidence['mat_sha256']
(OUT / 'evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({
    'mat_sha256': evidence['mat_sha256'], 'named_top_level_count': len(evidence['top_level_names']),
    'inp_fields': len(evidence['inp_field_names']), 'draw_fields': len(evidence['draw_field_names']),
    'load_warnings': evidence['load_warnings'], 'unsupported': evidence['unsupported'],
    'raw_record_count': len(records), 'raw_token_hits': evidence['raw_record_audit']['token_hits'],
    'numeric_29_matches': evidence['numeric_29_matches'],
    'sigmoid_handle': evidence['sigmoid_handle'], 'subsystem_records': evidence['function_subsystem']['records'],
    'evidence_file': str(OUT / 'evidence.json'),
}, ensure_ascii=False, indent=2))
