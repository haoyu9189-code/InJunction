import json
import numpy as np
import pytest
from nptdms import TdmsWriter, ChannelObject
from iv_batch import main
from test_iv_processing import cycle, write_tdms


def test_batch_exports_gui_compatible_arrays(tmp_path):
    source = tmp_path / 'input'
    source.mkdir()
    write_tdms(source / 'scan.TDMS', cycle())
    output = tmp_path / 'output'
    assert main([str(source), '--output', str(output), '--sample-points', '101']) == 0
    report = json.loads((output / 'report.json').read_text())
    assert report['status'] == 'complete'
    with np.load(output / 'forward_logI.npz', allow_pickle=False) as data:
        assert data['distance_array'].shape == (1, 101)
        assert data['conductance_array'].shape == (1, 101)
        assert data['length_array'].shape == (1,)
        assert data['additional_length'] == 101
    with pytest.raises(SystemExit):
        main([str(source), '--output', str(output)])


def test_batch_reports_empty_files_without_plotting(tmp_path):
    source = tmp_path / 'input'
    source.mkdir()
    with TdmsWriter(source / 'empty.tdms') as writer:
        writer.write_segment([ChannelObject('STM', name, np.array([], dtype=float))
                              for name in ['Bias (V)', 'Current (mA)', 'Log(G/G0)']])
    output = tmp_path / 'output'
    assert main([str(source), '--output', str(output)]) == 1
    assert not list(output.glob('*.png'))
    report = json.loads((output / 'report.json').read_text())
    assert report['status'] == 'failed'
    assert 'no IV samples' in report['files'][0]['error']


def test_partial_batch_retains_valid_file_and_reports_failure(tmp_path):
    source = tmp_path / 'input'
    source.mkdir()
    write_tdms(source / 'good.tdms', cycle())
    (source / 'broken.tdms').write_bytes(b'invalid')
    output = tmp_path / 'partial'
    assert main([str(source), '--output', str(output), '--sample-points', '101',
                 '--log-di-range', '-9', '-3', '--current-range', '-0.00001', '0.00001']) == 2
    report = json.loads((output / 'report.json').read_text())
    assert report['status'] == 'partial'
    assert len([f for f in report['files'] if 'error' in f]) == 1
    assert (output / 'forward.npz').is_file()
