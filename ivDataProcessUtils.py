# -*- coding: utf-8 -*-
# @Time   : 2025/3/27 23:21
# @Author : Gang/wang
# @File   : ivDataProcessUtils.py
import os
import matplotlib.pyplot as plt
from nptdms import TdmsFile
import numpy as np
import CaculateHistogram as cacu
import statsmodels.api as sm


def cacu_3fig(V_data, logI_data, logG, I_data, butter_parameter=1, sample_point=1000, label='data', bin_1dhis=400,
              bin_2dhis=200, threshold=255, logI_min_max=None, logdI_min_max=None, logG_min_max=None, I_min_max=None,
              color_2d='jet', output_dir='png_images'):
    if logI_min_max is None:
        logI_min_max = [-3, 3]
    if logdI_min_max is None:
        logdI_min_max = [-3, 3]
    if logG_min_max is None:
        logG_min_max = [-6, -1]
    if I_min_max is None:
        I_min_max = [-1e-6, 1e-6]
    datasets = (V_data, logI_data, logG, I_data)
    if len({len(data) for data in datasets}) != 1:
        raise ValueError("IV datasets must contain the same files")
    trace_count = 0
    for files in zip(*datasets):
        if len({len(traces) for traces in files}) != 1:
            raise ValueError("IV datasets must contain the same traces")
        for traces in zip(*files):
            if len({len(trace) for trace in traces}) != 1 or len(traces[0]) < 3:
                raise ValueError("IV traces must have matching lengths and at least three samples")
            if not np.isfinite(traces[0]).all() or not np.isfinite(traces[3]).all():
                raise ValueError("IV voltage and current must be finite")
            trace_count += 1
    if not trace_count:
        raise ValueError("No valid IV scans to plot; check scan markers and conductance limits")
    if not isinstance(sample_point, (int, np.integer)) or sample_point < 3:
        raise ValueError("sample_point must be an integer >= 3")
    if not 0 < butter_parameter <= 100:
        raise ValueError("Smoothing parameter must be in (0, 100]")
    V_data_sample = cacu.sample_data(V_data, sample_point)
    I_data_sample = cacu.sample_data(I_data, sample_point)
    logI_data_sample = cacu.sample_data(logI_data, sample_point)
    logG_sample = cacu.sample_data(logG, sample_point)

    I_data_sample_derivative_list = []

    for i in range(len(I_data_sample)):
        x = np.asarray(V_data_sample[i], dtype=float)
        y = np.asarray(I_data_sample[i], dtype=float)
        # LOWESS sorts x; restore the derivative to the original sweep order.
        # Merge repeated voltages (e.g. the spliced zero point) before gradient.
        unique_x, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
        if len(unique_x) < 3:
            raise ValueError("An IV scan needs at least three distinct voltages")
        unique_y = np.bincount(inverse, weights=y) / counts
        frac = min(1.0, max(0.01 * butter_parameter, 3.0 / len(unique_x)))
        smoothed_y = sm.nonparametric.lowess(
            unique_y, unique_x, frac=frac, it=0, is_sorted=True, return_sorted=False)
        derivative = np.gradient(smoothed_y, unique_x)
        with np.errstate(divide="ignore"):
            I_data_sample_derivative = np.log10(np.abs(derivative[inverse]))

        I_data_sample_derivative_list.append(I_data_sample_derivative)

    I_data_sample_derivative_list = np.array(I_data_sample_derivative_list)

    hist_logI, extent_logI, edg_logI = cacu.plt_2dIV(V_data_sample, logI_data_sample, bin_1dhis=bin_1dhis,
                                                     bin_2dhis=bin_2dhis,
                                                     threshold=threshold, range_y=[logI_min_max[0], logI_min_max[1]])

    hist_logDI, extent_logDI, edg_logDI = cacu.plt_2dIV(V_data_sample, I_data_sample_derivative_list,
                                                        bin_1dhis=bin_1dhis,
                                                        bin_2dhis=bin_2dhis, threshold=threshold,
                                                        range_y=[logdI_min_max[0], logdI_min_max[1]])

    hist_logG, extent_logG, edg_logG = cacu.plt_2dIV(V_data_sample, logG_sample, bin_2dhis=bin_2dhis,
                                                     bin_1dhis=bin_1dhis,
                                                     threshold=threshold, range_y=[logG_min_max[0], logG_min_max[1]])

    hist_I, extent_I, edg_I = cacu.plt_2dIV(V_data_sample, I_data_sample, bin_2dhis=bin_2dhis, bin_1dhis=bin_1dhis,
                                            threshold=threshold, range_y=[I_min_max[0], I_min_max[1]])

    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(20, 3.7))
    im0 = ax[0].imshow(hist_logI.T, origin='lower', extent=extent_logI, aspect='auto', cmap=color_2d)
    ax[0].set_title(f'log(I(nA)) 2D histogram({label})')
    ax[0].set_xlabel('V(V)')
    ax[0].set_ylabel('log(I(nA))')
    im1 = ax[1].imshow(hist_logDI.T, origin='lower', extent=extent_logDI, aspect='auto', cmap=color_2d)
    ax[1].set_title(f'log(dI/dV) 2D histogram({label})')
    ax[1].set_xlabel('V(V)')
    ax[1].set_ylabel('log(|dI/dV| (mA/V))')
    im2 = ax[2].imshow(hist_logG.T, origin='lower', extent=extent_logG, aspect='auto', cmap=color_2d)
    ax[2].set_title(f'log(G/G0) 2D histogram({label})')
    ax[2].set_xlabel('V(V)')
    ax[2].set_ylabel('log(G/G0)')

    im3 = ax[3].imshow(hist_I.T, origin='lower', extent=extent_I, aspect='auto', cmap=color_2d)
    ax[3].set_title(f'I 2D histogram({label})')
    ax[3].set_xlabel('V(V)')
    ax[3].set_ylabel('I(mA)')

    plt.colorbar(im0, ax=ax[0])
    plt.colorbar(im1, ax=ax[1])
    plt.colorbar(im2, ax=ax[2])
    plt.colorbar(im3, ax=ax[3])

    plt.tight_layout()
    folder_name = output_dir
    # 返回 :
    # 1. 图像地址
    os.makedirs(folder_name, exist_ok=True)
    image_path0 = os.path.join(folder_name, f'IVfig{label}.png')
    try:
        fig.savefig(image_path0, dpi=100)
    finally:
        plt.close(fig)

    # image_path2 = os.path.join(folder_name, f'IVfig{label}.eps')
    # plt.savefig(image_path2, format='eps', dpi=100)

    # 2. 图像数据
    his_data = (hist_logI, hist_logDI, hist_logG, hist_I, edg_logI, edg_logDI, edg_logG, edg_I)
    # 3.处理后数据
    data = (V_data_sample, logI_data_sample, I_data_sample_derivative_list, logG_sample, I_data_sample)
    return his_data, data


class IVDataProcessUtils:
    # logger = MyLog("IVDataProcessUtils", BASEDIR)

    @classmethod
    def loadTMDSFile(cls, filePath, return_layout=False):
        """
        加载tdms文件（单个）
        :param filePath:
        :param file_path:tdms文件路径
        :return:采样电压（numpy）
        """
        # Prefer semantic names so extra/reordered channels cannot change the mapping.
        aliases = {
            "bias": {"bias(v)": 1.0, "biasvolt": 1.0, "voltage(v)": 1.0,
                     "rt-io:ao1-biasvolt.[v]": 1.0},
            "current": {"current(ma)": 1.0, "current(a)": 1000.0,
                        "current(ua)": 0.001, "current(na)": 0.000001, "rt-pv:current[ma]": 1.0},
            "cond": {"log(g/g0)": 1.0, "logg": 1.0, "rt-pv:conductancelogg": 1.0},
        }
        layout = "positional"
        with TdmsFile.open(filePath) as tdmsFile:
            matches = []
            groups = tdmsFile.groups()
            recognized = False
            for group in groups:
                mapping = {}
                for channel in group.channels():
                    name = channel.name.lower().replace(" ", "").replace("μ", "u").replace("µ", "u")
                    for role, names in aliases.items():
                        if name in names:
                            recognized = True
                            if role in mapping:
                                raise ValueError(f"Ambiguous IV channels in group {group.name!r}")
                            mapping[role] = (channel, names[name])
                if len(mapping) == 3:
                    matches.append(mapping)
            if len(matches) == 1:
                names = [matches[0][role][0].name.lower().replace(" ", "")
                         for role in ("bias", "current", "cond")]
                layout = ("legacy-rt" if names == ["rt-io:ao1-biasvolt.[v]",
                           "rt-pv:current[ma]", "rt-pv:conductancelogg"] else "named")
                arrays = [np.asarray(matches[0][role][0][:], dtype=float) * matches[0][role][1]
                          for role in ("bias", "current", "cond")]
            elif not matches and not recognized and len(groups) == 1 and len(groups[0].channels()) == 3:
                # Legacy unnamed three-channel layout, in its original order and units.
                arrays = [np.asarray(channel[:], dtype=float) for channel in groups[0].channels()]
            else:
                layout = {group.name: [c.name for c in group.channels()] for group in groups}
                raise ValueError(f"Cannot identify a unique IV channel group: {layout}")
        if any(a.ndim != 1 for a in arrays) or len({len(a) for a in arrays}) != 1:
            raise ValueError("IV voltage, current and conductance channels must have equal lengths")
        if not len(arrays[0]):
            raise ValueError("TDMS contains no IV samples")
        return (arrays, layout) if return_layout else arrays

    @classmethod
    def hysteresis(cls, filePath, bias_base=0.1, peakStart=-2.5, peakEnd=-5.5, plateau_mode="auto"):
        """返回偏压，电导，电流三个二维数组，每一条占用一行
        """
        if not np.isfinite(bias_base) or bias_base == 0:
            raise ValueError("bias_base must be finite and nonzero")
        if not np.isfinite([peakStart, peakEnd]).all() or peakEnd > peakStart:
            raise ValueError("Conductance limits must be finite, with peakEnd <= peakStart")
        if plateau_mode not in ("auto", "current", "legacy"):
            raise ValueError("plateau_mode must be auto, current or legacy")
        (biasVolt, current, cond), layout = cls.loadTMDSFile(filePath, return_layout=True)
        legacy_plateaus = plateau_mode == "legacy" or (plateau_mode == "auto" and layout == "legacy-rt")
        biasVTrace, currentTrace, condTrace = [], [], []
        diffBiasV = np.concatenate((np.diff(biasVolt), np.array([10.0])))
        # 偏压从0.1到0.2阶跃中0.2v处的索引
        start_candi = \
            np.where((np.isclose(biasVolt, bias_base, rtol=1e-4, atol=1e-8)) & (np.isclose(diffBiasV, bias_base, rtol=1e-4, atol=1e-8)))[0] + 1

        # 从0.2到0.1的阶跃中0.2处的索引
        end_candi = \
            np.where((np.isclose(biasVolt, bias_base * 2, rtol=1e-4, atol=1e-8)) & (np.isclose(diffBiasV, -bias_base, rtol=1e-4, atol=1e-8)))[0]
        # 确保每个对应位置上结束点索引大于起始点索引
        startIdx = []
        endIdx = []
        for i in range(len(start_candi)):
            end_pos = np.searchsorted(end_candi, start_candi[i], side="right")
            if end_pos < len(end_candi):
                end = end_candi[end_pos]
                # Never join two cycles across an incomplete recording.
                if i + 1 < len(start_candi) and start_candi[i + 1] < end:
                    continue
                startIdx.append(start_candi[i])
                endIdx.append(end)
        startIdx = np.array(startIdx)
        endIdx = np.array(endIdx)

        # 得到扫面区间，接下来就是把中间的切开！！
        for i in range(startIdx.shape[0]):
            biasVTrace.append(biasVolt[startIdx[i]:endIdx[i] + 1])
            currentTrace.append(current[startIdx[i]:endIdx[i] + 1])
            condTrace.append(cond[startIdx[i]:endIdx[i] + 1])
        biasVTrace = np.array(biasVTrace, dtype='object')
        currentTrace = np.array(currentTrace, dtype='object')
        condTrace = np.array(condTrace, dtype='object')

        if biasVTrace.shape[0] == 0:
            return (None,) * 5

        condPeakStart = peakStart
        condPeakEnd = peakEnd
        # 寻找电压是0v的起始和终点
        plateau_means = np.full(biasVTrace.shape[0], np.nan)
        cutStart, cutEnd = np.full(biasVTrace.shape[0], -1, dtype=int), np.full(biasVTrace.shape[0], -1, dtype=int)
        for i in range(biasVTrace.shape[0]):
            trace = np.asarray(biasVTrace[i], dtype=float)
            if not np.isfinite(trace).all() or not np.isfinite(np.asarray(currentTrace[i], dtype=float)).all():
                continue
            zero_idx = np.flatnonzero(np.isclose(trace, 0, rtol=0, atol=1e-8))
            if len(zero_idx) < 2 or zero_idx[0] == 0 or zero_idx[-1] == len(trace) - 1:
                continue

            # 条件1 至少3个零点
            # if (len(zero_idx) < 3):
            #     continue

            # 条件2 必须有从2*bias_base-> 0 和 从 0 -> 2*bias_base的跳跃
            if abs(trace[zero_idx[0] - 1] - 2 * bias_base) > 0.0001 or abs(
                    trace[zero_idx[-1] + 1] - 2 * bias_base) > 0.0001:
                continue

            # 条件3 偏压在2*bias_base时的平均电导必须在范围内
            # Preserve historical RT plateau windows when the 100-point margins fit.
            # Short plateaus use the current window, never an empty legacy slice.
            front_end = zero_idx[0] - 1
            back_start = zero_idx[-1]
            back_end = len(trace) - 1
            if legacy_plateaus and front_end > 200:
                cond_start = np.asarray(condTrace[i][100:front_end - 100], dtype=float)
            else:
                cond_start = np.asarray(condTrace[i][:max(1, front_end)], dtype=float)
            if legacy_plateaus and back_end - back_start > 200:
                cond_end = np.asarray(condTrace[i][back_start + 100:back_end - 100], dtype=float)
            else:
                cond_end = np.asarray(condTrace[i][back_start + 1:-1], dtype=float)
                if not len(cond_end):
                    cond_end = np.asarray(condTrace[i][back_start + 1:], dtype=float)
            cond_start = cond_start[np.isfinite(cond_start)]
            cond_end = cond_end[np.isfinite(cond_end)]
            if not len(cond_start) or not len(cond_end):
                continue
            cond_start_mean = cond_start.mean()
            cond_end_mean = cond_end.mean()
            plateau_means[i] = (cond_start_mean + cond_end_mean) / 2
            if cond_start_mean < condPeakEnd or cond_start_mean > condPeakStart or cond_end_mean < condPeakEnd or cond_end_mean > condPeakStart:
                continue

            for j in range(zero_idx.shape[0] - 1):  # 找到正式扫描的起点
                if (zero_idx[j + 1] - zero_idx[j] > 1) and (trace[zero_idx[j] + 1] != 0):
                    cutStart[i] = zero_idx[j]
                    break
            if cutStart[i] == -1:
                continue
            # 寻找可能的终点
            temp_end = -1
            for j in range(zero_idx.shape[0] - 1, 0, -1):
                if zero_idx[j] - zero_idx[j - 1] > 1:
                    temp_end = zero_idx[j]
                    break
            if temp_end == -1:
                continue
            # 此时cut_start[i] 和 temp_end 均已找到
            peak_index = np.where((trace == trace.min()) | (trace == trace.max()))[0]
            first_peak = peak_index[0]
            if trace[first_peak] > 0 and trace[temp_end - 1] > 0:  # 从0到1，结尾必须从-1到0
                for j in range(temp_end - 1, 0, -1):
                    if trace[j] <= 0 and trace[j + 1] > 0:
                        temp_end = j
                        break
            elif trace[first_peak] < 0 and trace[temp_end - 1] < 0:  # 从0到-1，结尾必须从1到0
                for j in range(temp_end - 1, 0, -1):
                    if trace[j] >= 0 and trace[j + 1] < 0:
                        temp_end = j
                        break
            cutEnd[i] = temp_end

        # 删除不完整的
        trueIndex = np.where((cutStart == -1) | (cutEnd == -1), False, True)
        biasVTrace = biasVTrace[trueIndex]
        currentTrace = currentTrace[trueIndex]
        condTrace = condTrace[trueIndex]
        cutStart = cutStart[trueIndex]
        cutEnd = cutEnd[trueIndex]
        plateau_means = plateau_means[trueIndex]
        # 再次检查！！！
        if biasVTrace.shape[0] == 0:
            return (None,) * 5

        # 通过偏压把电导曲线切出来
        # 注意这里的这几个data其中每一行的数据维度都是不一致的！
        biasVData = np.empty(biasVTrace.shape[0], dtype=object)
        currentData = np.empty(biasVTrace.shape[0], dtype=object)
        condData = np.empty(biasVTrace.shape[0], dtype=object)

        for i in range(biasVTrace.shape[0]):
            biasVData[i] = biasVTrace[i][cutStart[i]:cutEnd[i] + 1]
            currentData[i] = currentTrace[i][cutStart[i]:cutEnd[i] + 1]
            condData[i] = condTrace[i][cutStart[i]:cutEnd[i] + 1]

        currentData_so = []
        # 对电流进行处理
        for i in range(currentData.shape[0]):
            currentData_so.append(currentData[i])
            with np.errstate(divide="ignore"):
                currentData[i] = np.log10(np.abs(np.asarray(currentData[i], dtype=float))) + 6
            currentData[i] = np.where(currentData[i] == -np.inf, -3, currentData[i])
        # 删除超过scanRange的数据
        # scanRange = 0.8 + 0.0001
        # tureIdx = [(data <= scanRange).all() for data in biasVData]
        # biasVData = biasVData[tureIdx]
        # currentData = currentData[tureIdx]
        # condData = condData[tureIdx]
        # Historical code forgot to filter meanCond along with the selected scans.
        # Keep those means aligned here; current mode retains its existing auxiliary field.
        meanCond = plateau_means if legacy_plateaus else [data[0] for data in condData]
        # currentData_so = currentData_so[tureIdx]

        # 再次检查！！！
        if biasVData.shape[0] == 0:
            return (None,) * 5
        else:
            return currentData, condData, biasVData, currentData_so, meanCond

    @classmethod
    def getPartitionData(cls, currentData, condData, biasVData, currentData_source, meanCond, de_capicity=False):
        """Split and splice within each cycle, including negative-first sweeps."""
        forward, reverse = [[] for _ in range(5)], [[] for _ in range(5)]
        if biasVData is None:
            return tuple([] for _ in range(10))
        for i, raw_trace in enumerate(biasVData):
            trace = np.asarray(raw_trace, dtype=float)
            if len(trace) < 3 or not (trace.min() < 0 < trace.max()):
                continue
            sources = [trace, np.asarray(currentData[i], dtype=float),
                       np.asarray(condData[i], dtype=float),
                       np.asarray(currentData_source[i], dtype=float)]
            if any(len(a) != len(trace) for a in sources):
                raise ValueError("IV trace channels must have equal lengths")
            peaks = np.flatnonzero((trace == trace.max()) | (trace == trace.min()))
            segments = []
            # Join the two half-scans only within this cycle, preserving legacy order.
            half_indices = np.r_[np.arange(peaks[-1], len(trace)), np.arange(peaks[0] + 1)]
            segments.append((trace[peaks[0]] > 0, half_indices))
            for start, end in zip(peaks[:-1], peaks[1:]):
                if trace[start] != trace[end]:
                    segments.append((trace[end] > trace[start], np.arange(start, end + 1)))
            for is_forward, indices in segments:
                values = [a[indices] for a in sources]
                if de_capicity and not is_forward:
                    # Shift inside this cycle only. Reject a shift that would wrap/truncate.
                    offset = int(np.argmin(values[1])) - len(indices) // 2
                    shifted = indices + offset
                    if shifted.min() < 0 or shifted.max() >= len(trace):
                        continue
                    values[1] = sources[1][shifted]
                    values[3] = sources[3][shifted]
                    # Use raw mA, not the logI display floor at zero current.
                    # Convert mA to A before dividing by V and G0 (siemens).
                    with np.errstate(divide="ignore", invalid="ignore"):
                        values[2] = (np.log10(np.abs(values[3])) - 3
                                     - np.log10(np.abs(values[0])) - np.log10(77.6e-6))
                target = forward if is_forward else reverse
                for destination, value in zip(target[:4], values):
                    destination.append(value)
                target[4].append(meanCond[i])
        return (forward[0], forward[1], forward[2], reverse[0], reverse[1], reverse[2],
                forward[3], reverse[3], forward[4], reverse[4])

    @classmethod
    def iv_process(cls, tdms_file, bias_base, peakStart, peakEnd, de_capicity, raise_on_error=False, plateau_mode="auto"):
        """
        Process IV curve data.

        Args:
            tdms_file (str): Path to the TDMS file.
            bias_base (float): Bias base value.
            peakStart (float): Peak start value.
            peakEnd (float): Peak end value.
            de_capicity (int): De-capacity value.
            raise_on_error (bool): Raise a file-specific error instead of returning [].
            plateau_mode (str): auto selects historical windows for known legacy RT channels;
                current/legacy explicitly select a plateau policy.

        Returns:
            Tuple: A tuple containing processed biasVDataFor, currentDataFor, condDataFor,
            biasVDataReve, currentDataReve, condDataReve, currentData_sourceFor,
            currentData_sourceReve, meanCond_sourceFor, meanCond_sourceReve.
        """
        try:
            currentData, condData, biasVData, currentData_source, meanCond = IVDataProcessUtils.hysteresis(tdms_file,
                                                                                                           bias_base=bias_base,
                                                                                                           peakStart=peakStart,
                                                                                                           peakEnd=peakEnd,
                                                                                                           plateau_mode=plateau_mode)

            if biasVData is None:
                raise ValueError(
                    f"No valid IV scans: bias_base={bias_base}, conductance=[{peakEnd}, {peakStart}]. "
                    "Check scan markers and conductance limits.")

            biasVDataFor, currentDataFor, condDataFor, biasVDataReve, currentDataReve, condDataReve, currentData_sourceFor, currentData_sourceReve, meanCond_sourceFor_new, meanCond_sourceReve = IVDataProcessUtils.getPartitionData(
                currentData, condData, biasVData, currentData_source, meanCond, de_capicity=bool(de_capicity))

            return (biasVDataFor, currentDataFor, condDataFor, biasVDataReve, currentDataReve,
                    condDataReve, currentData_sourceFor, currentData_sourceReve, meanCond_sourceFor_new,
                    meanCond_sourceReve)
        except Exception as e:
            # Log the error or handle it in a more appropriate way
            if raise_on_error:
                raise ValueError(f"{os.path.basename(tdms_file)}: {e}") from e
            print(f"IV processing failed for {tdms_file}: {e}")
            return []
