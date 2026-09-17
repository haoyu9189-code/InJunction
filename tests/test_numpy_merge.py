from pathlib import Path
import numpy as np
import pytest
from numpy_merge import merge_numpy_files


def sample(path, n, start=0, points=4):
    x = np.arange(n * points, dtype=float).reshape(n, points) + start
    np.savez(path, distance_array=x, conductance_array=x + 100,
             length_array=np.arange(n, dtype=float) + start, additional_length=points)
    return path


def test_trace_alignment_and_scalar(tmp_path):
    a = sample(tmp_path / 'a.npz', 2)
    b = sample(tmp_path / 'b.npz', 3, 20)
    result = merge_numpy_files([b, a], tmp_path / 'out.npz')
    assert result['counts'] == [3, 2]
    with np.load(tmp_path / 'out.npz', allow_pickle=False) as d:
        assert d['distance_array'].shape == (5, 4)
        np.testing.assert_array_equal(d['conductance_array'], d['distance_array'] + 100)
        np.testing.assert_array_equal(d['length_array'], [20, 21, 22, 0, 1])
        assert d['additional_length'].shape == ()
        assert d['additional_length'] == 4


@pytest.mark.parametrize('fault', ['fields', 'dtype', 'shape', 'scalar', 'alignment', 'object'])
def test_incompatible_preserves_existing_output(tmp_path, fault):
    a = sample(tmp_path / 'a.npz', 2)
    with np.load(a) as z:
        data = {k: z[k] for k in z.files}
    if fault == 'fields':
        data['extra'] = np.ones(2)
    elif fault == 'dtype':
        data['length_array'] = data['length_array'].astype(np.float32)
    elif fault == 'shape':
        data['distance_array'] = data['distance_array'][:, :2]
        data['conductance_array'] = data['conductance_array'][:, :2]
    elif fault == 'scalar':
        data['additional_length'] = 99
    elif fault == 'alignment':
        data['length_array'] = np.ones(1)
    elif fault == 'object':
        data['length_array'] = np.array([{}, {}], dtype=object)
    b = tmp_path / 'bad.npz'
    np.savez(b, **data)
    out = tmp_path / 'out.npz'
    out.write_bytes(b'previous result')
    with pytest.raises(ValueError):
        merge_numpy_files([a, b], out)
    assert out.read_bytes() == b'previous result'


def test_npy_and_input_protection(tmp_path):
    a, b = tmp_path / 'a.npy', tmp_path / 'b.npy'
    np.save(a, np.array([[1, 2]]))
    np.save(b, np.array([[3, 4], [5, 6]]))
    merge_numpy_files([a, b], tmp_path / 'out.npy')
    np.testing.assert_array_equal(np.load(tmp_path / 'out.npy'), [[1, 2], [3, 4], [5, 6]])
    for inputs, output in [([a, a], tmp_path / 'out.npy'), ([a, b], a), ([a], tmp_path / 'out.npy')]:
        with pytest.raises(ValueError):
            merge_numpy_files(inputs, output)
    np.testing.assert_array_equal(np.load(a), [[1, 2]])


def test_atomic_write_failure(tmp_path, monkeypatch):
    a, b = sample(tmp_path / 'a.npz', 2), sample(tmp_path / 'b.npz', 2)
    out = tmp_path / 'out.npz'
    out.write_bytes(b'previous')
    def fail(*args, **kwargs):
        raise OSError('disk full')
    monkeypatch.setattr(np, 'savez_compressed', fail)
    with pytest.raises(OSError):
        merge_numpy_files([a, b], out)
    assert out.read_bytes() == b'previous'
    assert not list(tmp_path.glob('.numpy-merge-*'))


def test_curve_count_metadata(tmp_path):
    paths = []
    for n in (2, 3):
        p = sample(tmp_path / f'{n}.npz', n)
        with np.load(p) as z:
            d = {k: z[k] for k in z.files}
        d['additional_length'] = n
        np.savez(p, **d)
        paths.append(p)
    merge_numpy_files(paths, tmp_path / 'merged.npz')
    with np.load(tmp_path / 'merged.npz') as z:
        assert z['additional_length'] == 5


def test_ambiguous_metadata_requires_explicit_choice(tmp_path):
    paths = [sample(tmp_path / 'a.npz', 4), sample(tmp_path / 'b.npz', 4)]
    with pytest.raises(ValueError, match='同时'):
        merge_numpy_files(paths, tmp_path / 'out.npz')
    merge_numpy_files(paths, tmp_path / 'out.npz', 'points')
    with np.load(tmp_path / 'out.npz') as z:
        assert z['additional_length'] == 4
    merge_numpy_files(paths, tmp_path / 'out.npz', 'count')
    with np.load(tmp_path / 'out.npz') as z:
        assert z['additional_length'] == 8
