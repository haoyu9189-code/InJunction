"""Strict, pickle-free NumPy concatenation with atomic output."""
from pathlib import Path
import os
import tempfile
import numpy as np


def merge_numpy_files(paths, output, additional_length_mode="auto"):
    paths = [Path(p).resolve() for p in paths]
    output = Path(output).resolve()
    if len(paths) < 2:
        raise ValueError('请至少选择两个文件。')
    if len(set(paths)) != len(paths):
        raise ValueError('存在重复文件，请移除后重试。')
    if output in paths:
        raise ValueError('输出文件不能覆盖输入文件。')
    suffix = paths[0].suffix.lower()
    if suffix not in ('.npy', '.npz') or any(p.suffix.lower() != suffix for p in paths):
        raise ValueError('只能合并相同类型的 .npy 或 .npz 文件。')
    if output.suffix.lower() != suffix:
        raise ValueError('输出文件扩展名必须与输入一致。')
    datasets = []
    counts = []
    for path in paths:
        try:
            if suffix == '.npz':
                with np.load(path, allow_pickle=False) as archive:
                    data = {k: archive[k] for k in archive.files}
            else:
                data = {'array': np.load(path, allow_pickle=False)}
            if not data or any(v.dtype.hasobject for v in data.values()):
                raise ValueError('空文件或含不支持的对象数组。')
            arrays = [v for v in data.values() if v.ndim > 0]
            if not arrays or arrays[0].shape[0] == 0:
                raise ValueError('没有可合并的数据行。')
            n = arrays[0].shape[0]
            if any(v.shape[0] != n for v in arrays):
                raise ValueError('各数组的第一维长度不同，无法保证逐行对应。')
            known = {'distance_array', 'conductance_array', 'length_array'}
            if known & data.keys():
                if not known <= data.keys():
                    raise ValueError('缺少 distance_array、conductance_array 或 length_array。')
                if data['distance_array'].ndim != 2 or data['distance_array'].shape != data['conductance_array'].shape:
                    raise ValueError('距离/电压与电导/电流数组必须是形状相同的二维数组。')
                if data['length_array'].shape != (n,):
                    raise ValueError('length_array 必须与曲线数量逐条对应。')
            if datasets:
                first = datasets[0]
                if data.keys() != first.keys():
                    raise ValueError('NPZ 字段名称不同。')
                for key, value in data.items():
                    ref = first[key]
                    if value.dtype != ref.dtype or value.ndim != ref.ndim or value.shape[1:] != ref.shape[1:]:
                        raise ValueError(f'字段 {key} 的数据类型或每行形状不同。')
                    if value.ndim == 0 and key != 'additional_length' and not np.array_equal(value, ref):
                        raise ValueError(f'标量参数 {key} 不一致。')
            datasets.append(data)
            counts.append(n)
        except (OSError, ValueError, KeyError, EOFError) as exc:
            raise ValueError(f'{path.name}：{exc}') from exc
    additional = None
    if 'additional_length' in datasets[0] and datasets[0]['additional_length'].ndim == 0:
        values = [d['additional_length'] for d in datasets]
        known = all('distance_array' in d and d['distance_array'].ndim == 2 for d in datasets)
        counts_match = known and all(v == n for v, n in zip(values, counts))
        points_match = known and all(v == d['distance_array'].shape[1] for v, d in zip(values, datasets))
        mode = additional_length_mode
        if mode not in ('auto', 'count', 'points', 'constant'):
            raise ValueError('未知的 additional_length 处理方式。')
        if mode == 'auto':
            if counts_match and points_match:
                raise ValueError('additional_length 同时等于曲线数量和点数，请在弹窗中明确选择其含义。')
            mode = 'count' if counts_match else ('points' if points_match else 'constant')
        if mode == 'count':
            if not counts_match:
                raise ValueError('additional_length 并非每个文件的曲线数量。')
            additional = np.asarray(sum(counts), dtype=values[0].dtype)
            if additional != sum(counts):
                raise ValueError('合并后的曲线数量超出 additional_length 数据类型范围。')
        elif mode == 'points':
            if not points_match:
                raise ValueError('additional_length 并非每条曲线的点数。')
            additional = values[0]
        else:
            if not all(np.array_equal(v, values[0]) for v in values):
                raise ValueError('additional_length 参数不一致，无法确定如何合并。')
            additional = values[0]
    merged = {key: (np.concatenate([d[key] for d in datasets], axis=0) if value.ndim else value)
              for key, value in datasets[0].items()}
    if additional is not None:
        merged['additional_length'] = additional
    # Write beside the destination so a failed write never damages an existing file.
    fd, temporary = tempfile.mkstemp(prefix='.numpy-merge-', dir=output.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            if suffix == '.npz':
                np.savez_compressed(stream, **merged)
            else:
                np.save(stream, merged['array'], allow_pickle=False)
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {'files': len(paths), 'rows': sum(counts), 'counts': counts, 'output': str(output)}
