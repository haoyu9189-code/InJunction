import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest
from nptdms import TdmsWriter, ChannelObject

from ivDataProcessUtils import IVDataProcessUtils as IV, cacu_3fig
from CaculateHistogram import plt_2dIV


def cycle(plateau=40, negative=False):
    sweep = np.r_[np.linspace(0, .5, 51), np.linspace(.49, -.5, 100), np.linspace(-.49, 0, 50)]
    if negative:
        sweep = -sweep
    v = np.r_[np.full(20, .1), np.full(plateau, .2), np.zeros(5), sweep,
              np.zeros(5), np.full(plateau, .2), np.full(220, .1)]
    return v, 1e-5 * v, np.full(len(v), -4.)


def write_tdms(path, arrays, names=None, order=(0, 1, 2), extra=False):
    names = names or ['Bias (V)', 'Current (mA)', 'Log(G/G0)']
    objects = [ChannelObject('STM', names[i], arrays[i]) for i in order]
    if extra:
        objects.insert(0, ChannelObject('Aux', 'Time', np.arange(len(arrays[0]))))
        objects.insert(1, ChannelObject('STM', 'Temperature', np.full(len(arrays[0]), 25.)))
    with TdmsWriter(path) as writer:
        writer.write_segment(objects)
    return path


@pytest.mark.parametrize('plateau', [1, 11, 39, 100, 400])
@pytest.mark.parametrize('negative', [False, True])
def test_old_and_short_plateaus(tmp_path, plateau, negative):
    path = write_tdms(tmp_path / 'scan.tdms', cycle(plateau, negative))
    result = IV.iv_process(path, .1, -2.5, -5, False, raise_on_error=True)
    assert [len(a) for a in result] == [1] * 10
    assert np.all(np.diff(result[0][0]) >= 0)
    assert np.all(np.diff(result[3][0]) <= 0)
    for v, raw, log_i in [(result[0][0], result[6][0], result[1][0]),
                           (result[3][0], result[7][0], result[4][0])]:
        np.testing.assert_allclose(raw, v * 1e-5)
        nonzero = raw != 0
        np.testing.assert_allclose(log_i[nonzero], np.log10(np.abs(raw[nonzero])) + 6)
        assert np.all(log_i[~nonzero] == -3)


def test_semantic_channels_and_legacy_fallback(tmp_path):
    arrays = cycle()
    for kwargs in [dict(order=(2, 0, 1), extra=True), dict(names=['Untitled', 'Untitled 1', 'Untitled 2'])]:
        path = write_tdms(tmp_path / 'channels.tdms', arrays, **kwargs)
        for actual, expected in zip(IV.loadTMDSFile(path), arrays):
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('unit,scale', [('A', 1e3), ('mA', 1), ('uA', .001), ('nA', 1e-6)])
def test_explicit_current_units(tmp_path, unit, scale):
    v, current, g = cycle()
    path = write_tdms(tmp_path / 'units.tdms', (v, current / scale, g),
                      names=['Bias (V)', f'Current ({unit})', 'Log(G/G0)'])
    np.testing.assert_allclose(IV.loadTMDSFile(path)[1], current)


def test_mismatched_or_unknown_channels(tmp_path):
    v, i, g = cycle()
    path = write_tdms(tmp_path / 'bad.tdms', (v, i[:-1], g))
    with pytest.raises(ValueError, match='equal lengths'):
        IV.loadTMDSFile(path)
    path = write_tdms(path, (v, i, g), names=['Bias (V)', 'Current (pA)', 'Log(G/G0)'])
    with pytest.raises(ValueError, match='identify'):
        IV.loadTMDSFile(path)


@pytest.mark.parametrize('mode', ['flat', 'nozero', 'nonfinite_plateau', 'filtered', 'truncated', 'empty'])
def test_no_data_has_stable_contract(tmp_path, mode):
    arrays = list(cycle())
    if mode == 'flat': arrays[0][:] = .1
    if mode == 'nozero': arrays[0][arrays[0] == 0] = .001
    if mode == 'nonfinite_plateau': arrays[2][arrays[0] == .2] = np.nan
    if mode == 'filtered': arrays[2][:] = -8
    if mode == 'truncated': arrays = [a[:150] for a in arrays]
    if mode == 'empty': arrays = [a[:0] for a in arrays]
    path = write_tdms(tmp_path / 'bad.tdms', arrays)
    if mode != 'empty':
        assert IV.hysteresis(path) == (None,) * 5
    assert IV.iv_process(path, .1, -2.5, -5, False) == []
    with pytest.raises(ValueError, match='bad.tdms'):
        IV.iv_process(path, .1, -2.5, -5, False, raise_on_error=True)


def test_no_cross_cycle_splicing(tmp_path):
    one = cycle(39)
    two = cycle(50, negative=True)
    arrays = [np.r_[a, b] for a, b in zip(one, two)]
    arrays[1][len(one[0]):] *= 2
    path = write_tdms(tmp_path / 'two.tdms', arrays)
    result = IV.iv_process(path, .1, -2.5, -5, False, True)
    assert [len(a) for a in result] == [2] * 10
    for k, factor in enumerate([1, 2]):
        np.testing.assert_allclose(result[6][k], result[0][k] * 1e-5 * factor)
        np.testing.assert_allclose(result[7][k], result[3][k] * 1e-5 * factor)


def test_decapacitance_units_and_bounds(tmp_path):
    path = write_tdms(tmp_path / 'scan.tdms', cycle())
    result = IV.iv_process(path, .1, -2.5, -5, True, True)
    v, log_g = result[3][0], result[5][0]
    np.testing.assert_allclose(log_g[v != 0], np.log10(1e-8 / 77.6e-6))
    # Force a shift outside a reverse half-scan: it must not wrap into another cycle.
    data = list(IV.hysteresis(write_tdms(path, cycle(negative=True))))
    data[0][0][:] = 0
    data[0][0][0] = -20
    partitioned = IV.getPartitionData(*data, de_capicity=True)
    for x, y in zip(partitioned[3], partitioned[7]):
        assert len(x) == len(y)


def test_histogram_rejects_empty_and_masks_pairs():
    with pytest.raises(ValueError, match='No finite'):
        plt_2dIV([], [], range_y=[-3, 3])
    with pytest.raises(ValueError, match='matching'):
        plt_2dIV([[0, 1]], [[1]], range_y=[-3, 3])
    hist, _, _ = plt_2dIV([[0, 1, 2]], [[1, np.inf, 2]], range_y=[0, 3])
    assert hist.sum() == 2


def test_derivative_keeps_voltage_order(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    x = np.linspace(.5, -.5, 101)
    y = 1e-6 * (x + .75) ** 2
    with np.errstate(divide='ignore'):
        log_i = np.log10(np.abs(y)) + 6
    _, data = cacu_3fig([[x]], [[log_i]], [[np.full(len(x), -4.)]], [[y]],
                       sample_point=101, bin_1dhis=20, bin_2dhis=20)
    expected = np.log10(2e-6 * (x + .75))
    np.testing.assert_allclose(data[2][0][2:-2], expected[2:-2], atol=.002)
    assert (tmp_path / 'png_images/IVfigdata.png').is_file()
    assert not plt.get_fignums()
    with pytest.raises(ValueError, match='No valid IV'):
        cacu_3fig([], [], [], [])


@pytest.mark.skipif(not os.environ.get('IV_SAMPLE_DIR'), reason='Set IV_SAMPLE_DIR for private sample regression')
def test_real_new_machine_samples():
    paths = sorted(Path(os.environ['IV_SAMPLE_DIR']).glob('*.tdms'))
    assert len(paths) == 3
    for path, count in zip(paths, [99, 88, 80]):
        result = IV.iv_process(path, .1, -2.5, -5, False, True)
        assert [len(a) for a in result] == [count] * 10


def test_spawn_worker_reads_and_plots(tmp_path):
    import multiprocessing
    path = write_tdms(tmp_path / 'scan.tdms', cycle())
    with multiprocessing.get_context('spawn').Pool(1) as pool:
        result = pool.apply_async(IV.iv_process, (path, .1, -2.5, -5, False, True)).get(timeout=60)
        hist, data = pool.apply_async(cacu_3fig,
                                     ([result[0]], [result[1]], [result[2]], [result[6]]),
                                     {'sample_point': 101, 'output_dir': tmp_path}).get(timeout=60)
    assert len(data[0]) == 1
    assert hist[0].sum() > 0
    assert (tmp_path / 'IVfigdata.png').is_file()


LEGACY_RT_NAMES = ['RT-IO:AO1-Bias Volt. [V]', 'RT-PV:Current [mA]', 'RT-PV:Conductance LogG']


def test_legacy_rt_transients_and_selected_mean_alignment(tmp_path):
    rejected = cycle(400)
    rejected[2][:] = -8
    accepted = cycle(450)
    v, _, g = accepted
    edges = np.flatnonzero(np.diff(np.r_[False, v == .2, False]))
    for start, stop in zip(edges[::2], edges[1::2]):
        g[start:start + 100] = 0
        g[stop - 100:stop] = 0
    arrays = [np.r_[a, b] for a, b in zip(rejected, accepted)]
    path = write_tdms(tmp_path / 'legacy.tdms', arrays, names=LEGACY_RT_NAMES,
                      order=(2, 0, 1), extra=True)
    _, layout = IV.loadTMDSFile(path, return_layout=True)
    assert layout == 'legacy-rt'
    result = IV.iv_process(path, .1, -2.5, -5, False, True)
    assert [len(a) for a in result] == [1] * 10
    # The rejected first cycle must not contribute its -8 mean to this retained cycle.
    assert -4.01 < result[8][0] < -3.9
    assert result[8] == result[9]
    with pytest.raises(ValueError, match='No valid IV scans'):
        IV.iv_process(path, .1, -2.5, -5, False, True, plateau_mode='current')


def test_legacy_named_short_plateaus_fall_back_safely(tmp_path):
    path = write_tdms(tmp_path / 'short-rt.tdms', cycle(39), names=LEGACY_RT_NAMES)
    assert [len(a) for a in IV.iv_process(path, .1, -2.5, -5, False, True)] == [1] * 10


@pytest.mark.skipif(not os.environ.get('IV_OLD_SAMPLE'), reason='Set IV_OLD_SAMPLE for private legacy regression')
def test_real_old_machine_sample():
    path = Path(os.environ['IV_OLD_SAMPLE'])
    arrays, layout = IV.loadTMDSFile(path, return_layout=True)
    assert layout == 'legacy-rt'
    assert [len(a) for a in arrays] == [2500000] * 3
    result = IV.iv_process(path, .1, -2.5, -5, False, True)
    assert [len(a) for a in result] == [32] * 10
    for values in result[8:]:
        assert np.all((np.asarray(values) >= -5) & (np.asarray(values) <= -2.5))
    current = IV.iv_process(path, .1, -2.5, -5, False, True, plateau_mode='current')
    assert [len(a) for a in current] == [27] * 10
