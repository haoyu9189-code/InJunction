"""Process a TDMS directory without the desktop GUI; keep an auditable run report."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import numpy as np

from ivDataProcessUtils import IVDataProcessUtils, cacu_3fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='Directory containing TDMS files')
    parser.add_argument('--output', required=True, type=Path, help='New output directory (must not exist)')
    parser.add_argument('--bias-base', type=float, default=.1)
    parser.add_argument('--peak-start', type=float, default=-2.5)
    parser.add_argument('--peak-end', type=float, default=-5.)
    parser.add_argument('--sample-points', type=int, default=1000)
    parser.add_argument('--decapacitance', action='store_true')
    parser.add_argument('--log-di-range', nargs=2, type=float, default=[-3., 3.],
                        metavar=('MIN', 'MAX'), help='log10 |dI/dV| display range, I in mA')
    parser.add_argument('--current-range', nargs=2, type=float, default=[-1e-6, 1e-6],
                        metavar=('MIN', 'MAX'), help='Current display range in mA')
    args = parser.parse_args(argv)
    if not args.input.is_dir():
        parser.error('Input must be a directory')
    paths = sorted(p for p in args.input.iterdir() if p.is_file() and p.suffix.lower() == '.tdms')
    if not paths:
        parser.error('No TDMS files found')
    if (not np.isfinite([args.bias_base, args.peak_start, args.peak_end]).all()
            or args.bias_base == 0 or args.peak_end > args.peak_start or args.sample_points < 3):
        parser.error('Invalid bias, conductance limits or sample point count')
    for limits in (args.log_di_range, args.current_range):
        if not np.isfinite(limits).all() or limits[0] >= limits[1]:
            parser.error('Display ranges must be finite and increasing')
    try:
        args.output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error('Output directory already exists; choose a new directory')
    report = {'parameters': {'bias_base': args.bias_base, 'peak_start': args.peak_start,
                            'peak_end': args.peak_end, 'sample_points': args.sample_points,
                            'decapacitance': args.decapacitance, 'log_di_range': args.log_di_range,
                            'current_range_mA': args.current_range}, 'files': [], 'status': 'failed'}
    results = []
    for path in paths:
        try:
            result = IVDataProcessUtils.iv_process(path, args.bias_base, args.peak_start, args.peak_end,
                                                  args.decapacitance, raise_on_error=True)
            if not result[0] or not result[3]:
                raise ValueError('No complete forward/reverse scans')
            results.append(result)
            report['files'].append({'file': path.name, 'forward': len(result[0]), 'reverse': len(result[3])})
            print(f'{path.name}: {len(result[0])} forward, {len(result[3])} reverse')
        except Exception as exc:
            report['files'].append({'file': path.name, 'error': str(exc)})
            print(str(exc), file=sys.stderr)
    exit_code = 1
    try:
        if not results:
            raise ValueError('No valid IV scans; no plots generated')
        for direction, indices in [('forward', (0, 1, 2, 6, 8)), ('reverse', (3, 4, 5, 7, 9))]:
            voltage, log_i, log_g, raw_i, mean_g = [[r[index] for r in results] for index in indices]
            hist, data = cacu_3fig(voltage, log_i, log_g, raw_i, label=direction,
                                  sample_point=args.sample_points, output_dir=args.output,
                                  logdI_min_max=args.log_di_range, I_min_max=args.current_range)
            means = np.asarray([g for group in mean_g for g in group], dtype=float)
            np.savez_compressed(args.output / f'{direction}.npz', voltage=data[0], log_current_nA=data[1],
                                log_dI_dV_mA_per_V=data[2], log_conductance=data[3], current_mA=data[4],
                                mean_conductance=means)
            # Same field names as the GUI's Save_iv, for reloading and clustering.
            for index, name in enumerate(('logI', 'log_dI_dv', 'logG', 'I')):
                np.savez_compressed(args.output / f'{direction}_{name}.npz', distance_array=data[0],
                                    conductance_array=data[index + 1], length_array=means,
                                    additional_length=args.sample_points)
            np.savez_compressed(args.output / f'{direction}_histograms.npz',
                                **{f'hist_{i}': a for i, a in enumerate(hist[:4])},
                                **{f'edges_{i}_{axis}': edge for i, pair in enumerate(hist[4:])
                                   for axis, edge in zip(('x', 'y'), pair)})
        report['status'] = 'partial' if any('error' in f for f in report['files']) else 'complete'
        exit_code = 2 if report['status'] == 'partial' else 0
    except Exception as exc:
        report['error'] = str(exc)
        print(str(exc), file=sys.stderr)
    finally:
        (args.output / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
