import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score
from sklearn.mixture import GaussianMixture
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.stats import norm


# from PySide6.QtCore import QMetaObject, Qt, Q_ARG
# import time
# import threading



# def button_clicked_thread(func, progressBar, speed):
#

#     thread = threading.Thread(target=func, daemon=True)
#     thread.start()
#
#
#     if progressBar is not None:
#         # 启动更新进度条的线程
#         update_thread = threading.Thread(target=update_progress, args=(progressBar, speed), daemon=True)
#         update_thread.start()
#     return
#
# #
# def update_progress(progressBar, speed):
#     while progressBar.value() < 100:
#         # 使用局部变量 value 来更新进度条的值
#         QMetaObject.invokeMethod(progressBar, "setValue", Qt.QueuedConnection, Q_ARG(int, progressBar.value() + 1))
#         time.sleep(speed)  # 控制更新速度，可以根据需要调整
#     return


def gaussian_fit(x, y):
    y = y / np.max(y)  # 将幅度归一化为1
    mean_guess = np.argmax(y)  # 使用峰值位置作为初始猜测
    params, _ = curve_fit(gaussian, x, y, p0=[mean_guess, len(x) / 6, 1.], maxfev=800000)
    mean, std, amplitude = params
    # 返回高斯分布的参数
    return mean


def gaussian_fit_2his(hist, range_y):
    mean_value = []
    hist = np.array(hist)
    x = np.arange(len(hist[0]))

    for i in range(len(hist)):
        mean_value_i = gaussian_fit(x, hist[i])
        mean_value_i2 = mean_value_i / len(hist[0]) * (range_y[1] - range_y[0]) + range_y[0]
        mean_value.append(mean_value_i2)
    # 返回高斯分布的参数
    return mean_value


def plt_2dIV_1div(object_data_x, object_data_y, name=None, bin_1dhis=200, bin_2dhis=200, threshold=255, sort_range=0.8,
                  color_2d='jet', color_1d='g', label='log(I(nA)/V', range_y=None, the_abs=False, gauss_fit=False,
                  range_x=None,file_path2=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))

    # 增加一步数据清洗
    # 计算每个数据点的方差/绝对值（如果是分段的话，此处需要调整）
    object_data_y = np.array(object_data_y)
    object_data_x = np.array(object_data_x)

    if the_abs:
        axes[0].set_title(f'fit_curve of top {sort_range * 100} % absolute data')
        sorted_indices_data = np.sum(np.abs(object_data_y), axis=1)  # 也可以进行绝对值大小的排序
    else:
        axes[0].set_title(f'fit_curve of top {sort_range * 100} % stable data')
        sorted_indices_data = np.var(object_data_y, axis=1)  # 进行稳定性排序
    # 对数据点根据方差排序
    sorted_indices = np.argsort(sorted_indices_data)

    # 选择方差最小的前50%的数据点
    num_selected_points = int(sort_range * len(object_data_y))
    selected_data_y = object_data_y[sorted_indices[:num_selected_points]]
    selected_data_x = object_data_x[sorted_indices[:num_selected_points]]
    # 计算筛选后的hist
    hist, extent, his2d_edg_list = plt_2dIV(selected_data_x, selected_data_y, bin_1dhis=bin_1dhis,bin_2dhis=bin_2dhis,
                                            threshold=threshold, range_y=range_y)

    # 计算拟合曲线
    if gauss_fit:
        # 计算高斯Mean
        mean_value = gaussian_fit_2his(hist, range_y)
        mean_x = np.arange(len(mean_value)) / len(mean_value) * (max(object_data_x[0]) - min(object_data_x[0])) + min(
            object_data_x[0])
    else:
        # 计算均值
        mean_value = np.sum(selected_data_y, axis=0) / num_selected_points
        mean_x = object_data_x[0]

    axes[0].plot(mean_x, mean_value, c=color_1d, linewidth=1, label='mean')
    im0 = axes[0].imshow(hist.T, origin='lower', extent=extent, aspect='auto', cmap=color_2d)
    plt.colorbar(im0, ax=axes[0])
    axes[0].set_xlabel('V(V)')
    axes[0].set_ylabel(label)
    if range_x is not None:
        extent = range_x
    axes[0].set_xlim(extent[0], extent[1])
    axes[0].set_ylim(range_y[0], range_y[1])
    bin_edges_con, histogram_con, con_his_list = calculate_his(object_data_y, bins_1Dcon=bin_1dhis,
                                                               low_cut_1Dcon=range_y[0],
                                                               high_cut_1Dcon=range_y[1])
    axes[1].bar(bin_edges_con[:-1], histogram_con, width=np.diff(bin_edges_con), align='edge', color=color_1d)
    axes[1].set_title('1D histogram')
    axes[1].set_xlabel(label)
    axes[1].set_ylabel('count')
    axes[1].grid(True)  # 显示网格线

    # 调整布局
    plt.tight_layout()
    folder_name = "png_images"
    image_path = os.path.join(folder_name, f'plt_2dIV_1div{name}.png')
    plt.savefig(image_path)
    parent_folder = os.path.dirname(file_path2)
    image_path2 = os.path.join(parent_folder, f'plt_2dIV_1div{name}.eps')
    plt.savefig(image_path2, format='eps', dpi=400)
    # zip转换回去

    hist_data = [bin_edges_con, con_his_list, his2d_edg_list, hist.T]
    red_line = [mean_x, mean_value]
    return image_path, hist_data, red_line, sorted_indices


def change_nan(data):
    for i in range(len(data)):
        if i == 0 and np.isnan(data[i]):
            data[i] = 0
        elif np.isnan(data[i]):
            data[i] = data[i - 1]
    return data


def sample_data(datas, sample_point=1000):
    sample_list = []
    for data in datas:
        for single_data in data:
            # 使用线性插值函数 interp1d 对 single_biasVDataReve 进行插值
            f = interp1d(np.arange(len(single_data)), single_data, kind='linear')
            # 生成一个新的采样点序列 x_new，采样点数量为 sample_point
            x_new = np.linspace(0, len(single_data) - 1, sample_point)
            # 将插值后的数据添加到 biasVDataReve_sample_list 中
            sample_list.append(f(x_new))
    return sample_list


def new_data(source_distance, source_conductance, ivData=False):
    # 考虑到采样可能会使得实验失真，在聚类分析时，可以先采样，再用源数据进行绘图
    conductance_cut = []
    distance_cut = []
    if ivData is False:
        for i in range(len(source_conductance)):
            conductance_cut_i, min_index = single_new_data(source_conductance[i])
            conductance_cut.append(conductance_cut_i)
            distance_cut.append(source_distance[i][:min_index])
        return conductance_cut, distance_cut
    else:
        return source_conductance, source_distance


def single_new_data(source_conductance):
    # 找到最小值的索引
    min_index = np.argmin(source_conductance)
    # 使用索引来切片 source_conductance 数组
    conductance_cut_pre = source_conductance[:min_index]
    return conductance_cut_pre, min_index


# 直接计算一维电导直方图的方法，其中输入为 single trance 中的conductance_array,mesh代表直方图分成几个网格，cut代表截至电导
def calculate_his(conductance, bins_1Dcon=1000, low_cut_1Dcon=-6, high_cut_1Dcon=1):
    number = len(conductance)
    histogram_con = np.zeros(bins_1Dcon)  # 创建一个空数组用于累积电导计数
    bin_edges_con = np.linspace(low_cut_1Dcon, high_cut_1Dcon, bins_1Dcon + 1)  # 电导坐标轴横轴
    for i in range(number):
        conductance_array = conductance[i]
        bin_counts2, _ = np.histogram(conductance_array, bins=bin_edges_con)
        histogram_con += bin_counts2

    # max_index = np.argmax(histogram_con)
    # histogram_con[max_index] = 0
    con_his_list = list(zip(bin_edges_con, histogram_con))
    return bin_edges_con, histogram_con, con_his_list


def plt_2dIV(biasV_list, current_list,bin_1dhis=400, bin_2dhis=200, threshold=255, range_y=None):
    if len(biasV_list) != len(current_list) or any(
            len(x) != len(y) for x, y in zip(biasV_list, current_list)):
        raise ValueError("IV histogram voltage/current traces must have matching lengths")
    x = np.asarray([item for sublist in biasV_list for item in sublist], dtype=float)
    y = np.asarray([item for sublist in current_list for item in sublist], dtype=float)
    finite_x = x[np.isfinite(x)]
    if not len(finite_x):
        raise ValueError("No finite IV voltage samples to plot")
    valid = np.isfinite(x) & np.isfinite(y)
    if range_y is None:
        if not valid.any():
            raise ValueError("No finite paired IV samples to plot")
        hist, x_edges, y_edges = np.histogram2d(x[valid], y[valid], bins=[bin_1dhis, bin_2dhis])
    else:
        range_x = [finite_x.min(), finite_x.max()]
        hist, x_edges, y_edges = np.histogram2d(
            x[valid], y[valid], bins=[bin_1dhis, bin_2dhis], range=[range_x, range_y])
    # 实际投影的范围
    his2d_edg_list = [x_edges, y_edges]
    # 将大于阈值的部分设置为饱和红色所对应的最大数据值
    hist[hist > threshold] = threshold
    # 图像只显示的范围
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    return hist, extent, his2d_edg_list


def calculate_2d_his(distance, conductance, bins_2dhis_x=200, bins_2dhis_y=100, distance_0=-0.2, distance_1=1.2,
                     conductance_0=1, conductance_1=-6.5, red=999):
    # 提取 x 和 y 坐标
    x = distance.flatten()  # 将多维数组展平成一维数组
    y = conductance.flatten()

    # 创建布尔掩码以过滤为 0 的元素
    mask = (x != 0) & (y != 0)

    # 使用掩码过滤数组中的元素
    x = x[mask]
    y = y[mask]

    # 设置直方图的网格大小和范围
    range_x = [distance_0, distance_1]
    range_y = [conductance_1, conductance_0]

    # 计算二维直方图，并同时得到 x_edges 和 y_edges
    hist, x_edges, y_edges = np.histogram2d(x, y, bins=[bins_2dhis_x, bins_2dhis_y], range=[range_x, range_y])
    # from itertools import zip_longest
    his2d_edg_list = [x_edges, y_edges]
    # his2d_edg_list = list(zip_longest(x_edges, y_edges))

    # 将大于阈值的部分设置为饱和红色所对应的最大数据值
    hist[hist > red] = red
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

    return hist, extent, his2d_edg_list


# 定义高斯函数
def gaussian(x, mean, std, amplitude):
    return amplitude * norm.pdf(x, mean, std)


def chose_cluster(conductance_normalized, clusters_way=0, n_clusters=2):
    # 清除缓存,参数传递
    labels = []
    if clusters_way == 0:
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10)
        kmeans.fit(conductance_normalized)
        # 获取每个样本的聚类标签
        labels = kmeans.labels_

    elif clusters_way == 1:
        gaussian_cluster = GaussianMixture(n_components=n_clusters, init_params='random')
        gaussian_cluster.fit(conductance_normalized)
        # 获取聚类标签
        labels = gaussian_cluster.predict(conductance_normalized)

    elif clusters_way == 2:
        from tslearn.clustering import KShape
        seed = 0
        np.random.seed(seed)
        ks = KShape(n_clusters=n_clusters, verbose=True, random_state=seed)
        labels = ks.fit_predict(conductance_normalized)

    elif clusters_way == 3:
        from tslearn.clustering import TimeSeriesKMeans
        seed = 0
        np.random.seed(seed)
        ks = TimeSeriesKMeans(n_clusters=n_clusters, verbose=True, random_state=seed)
        labels = ks.fit_predict(conductance_normalized)
    return labels


def cacu_chfig(labels, n_clusters, conductance_normalized):
    # 计算ch指数
    ch_score = calinski_harabasz_score(conductance_normalized, labels)
    ch_score = f"{ch_score:.1f}"
    # 使用PCA降维----------------------------------------------
    if len(conductance_normalized[0]) > 3:
        pca = PCA(n_components=3)
        conductance_pca = pca.fit_transform(conductance_normalized)
    else:
        conductance_pca = conductance_normalized
    # 可视化聚类结果
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制样本点
    ax.scatter(conductance_pca[:, 0], conductance_pca[:, 1], conductance_pca[:, 2], c=labels, cmap='viridis')

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    centroids_pca = []
    for i in range(n_clusters):
        # 获得聚类中心坐标
        class_i_indices = np.where(labels == i)[0]
        j = class_i_indices[int(len(class_i_indices) / 2)]
        centroids_pca.append(conductance_pca[j])
        ax.text(centroids_pca[i][0], centroids_pca[i][1], centroids_pca[i][2], str(i + 1), color='blue')
    plt.title('Clustering in 3D  CH:' + str(ch_score))
    plt.figure(fig.number)  # 或者使用 plt.figure(fig2) 传递图像对象
    folder_name = "png_images"
    path = os.path.join(folder_name, f'CH_scan{n_clusters}.png')
    plt.savefig(path)

    eps_path = os.path.join(folder_name, f'ch{n_clusters}.eps')
    plt.savefig(eps_path, format='eps')  # 保存为 EPS 文件
    return path


def clustering(distance, conductance, length, object_data=None, n_clusters=4, bins_1Dcon=1000,
               low_cut_1Dcon=-6,
               high_cut_1Dcon=1, bins_2dhis_x=500, bins_2dhis_y=500, distance_0=-0.2, distance_1=1.2,
               conductance_0=1, conductance_1=-6.5, bins_1Dlen=100, clusters_way=0, color_1d='blue',
               color_2d='viridis', red=999, ivData=False, label=None, his_1d_dis_0=0, his_1d_dis_1=2):
    # 进行聚类-------------------------------------------------
    plt.close('all')
    conductance_normalized = object_data
    if label is None:
        labels = chose_cluster(conductance_normalized, clusters_way=clusters_way, n_clusters=n_clusters)
    else:
        labels = label

    # 获取数据点的分类索引
    class_indices = []  # 用于存储各个分类的索引
    class_conductance = []  # 用于存储各个分类的conductance
    class_distance = []  # 用于存储各个分类的distance
    class_length = []  # 用于存储各个分类的distance

    # 显示原始图像

    for i in range(n_clusters):
        class_i_indices = np.where(labels == i)[0]
        class_i_conductance = conductance[class_i_indices]
        class_i_distance = distance[class_i_indices]
        class_i_length = length[class_i_indices]
        # 存储每次循环的结果
        class_indices.append(class_i_indices)
        class_conductance.append(class_i_conductance)
        class_distance.append(class_i_distance)
        class_length.append(class_i_length)
    # 一次性计算所有结果,创建图表

    datas = [class_distance, class_conductance, class_length]

    his_data, plot_paths, merge_figure = draw_merge_figure(datas, distance, conductance, length,
                                                           n_clusters=n_clusters,
                                                           bins_1Dcon=bins_1Dcon,
                                                           low_cut_1Dcon=low_cut_1Dcon,
                                                           high_cut_1Dcon=high_cut_1Dcon,
                                                           bins_2dhis_x=bins_2dhis_x,
                                                           bins_2dhis_y=bins_2dhis_y,
                                                           distance_0=distance_0,
                                                           distance_1=distance_1,
                                                           conductance_0=conductance_0,
                                                           conductance_1=conductance_1,
                                                           bins_1Dlen=bins_1Dlen, color_1d=color_1d,
                                                           color_2d=color_2d, red=red, ivData=ivData,
                                                           his_1d_dis_0=his_1d_dis_0,
                                                           his_1d_dis_1=his_1d_dis_1
                                                           )

    return datas, plot_paths, his_data, merge_figure, labels  # 返回图像文件路径列表,


def draw_merge_figure(datas, distance, conductance, length, n_clusters=4, bins_1Dcon=1000,
                      low_cut_1Dcon=-6,
                      high_cut_1Dcon=1, bins_2dhis_x=500, bins_2dhis_y=500, distance_0=-0.2, distance_1=1.2,
                      conductance_0=1, conductance_1=-6.5, bins_1Dlen=100, color_1d='blue', color_2d='viridis',
                      red=999, ivData=False, his_1d_dis_0=0, his_1d_dis_1=2):
    con_his_list_class = []
    len_his_list_class = []
    edges_2dhis_class = []
    hist_class = []
    plot_paths = []  # 存储图像文件路径的列表
    folder_name = "png_images"
    results = []
    len_conductance = len(conductance)

    Primitive_data = draw_figure(
        distance, conductance, length,
        bins_1Dcon=bins_1Dcon,
        low_cut_1Dcon=low_cut_1Dcon,
        high_cut_1Dcon=high_cut_1Dcon,
        bins_2dhis_x=bins_2dhis_x,
        bins_2dhis_y=bins_2dhis_y,
        distance_0=distance_0,
        label="raw_data",
        distance_1=distance_1,
        conductance_0=conductance_0,
        conductance_1=conductance_1,
        bins_1Dlen=bins_1Dlen,
        color_1d=color_1d,
        color_2d=color_2d,
        red=red,
        ivData=ivData,
        his_1d_dis_0=his_1d_dis_0,
        his_1d_dis_1=his_1d_dis_1
    )

    image_path = os.path.join(folder_name, 'raw_data.png')
    plot_paths.append(image_path)
    results.append(Primitive_data)
    for i in range(n_clusters):
        # 绘制1D电导直方图
        for_title = [len_conductance, n_clusters, i]

        cacu_data = draw_figure(
            datas[0][i], datas[1][i], datas[2][i],
            bins_1Dcon=bins_1Dcon,
            low_cut_1Dcon=low_cut_1Dcon,
            high_cut_1Dcon=high_cut_1Dcon,
            bins_2dhis_x=bins_2dhis_x,
            bins_2dhis_y=bins_2dhis_y,
            distance_0=distance_0,
            label=f'class_{i + 1}',
            distance_1=distance_1,
            conductance_0=conductance_0,
            conductance_1=conductance_1,
            for_title=for_title,
            bins_1Dlen=bins_1Dlen,
            color_1d=color_1d,
            color_2d=color_2d,
            red=red,
            ivData=ivData,
            his_1d_dis_0=his_1d_dis_0,
            his_1d_dis_1=his_1d_dis_1
        )

        results.append(cacu_data)
        temp_file_path = os.path.join(folder_name, f'class_{i + 1}.png')
        plot_paths.append(temp_file_path)  # 将临时文件路径添加到列表中
    # 储存所有文件
    for i in range(n_clusters + 1):
        (con_his_list, len_his_list, edges_2dhis, hist) = results[i]
        con_his_list_class.append(con_his_list)
        len_his_list_class.append(len_his_list)
        edges_2dhis_class.append(edges_2dhis)
        hist_class.append(hist)
    # 绘制大图
    fig, axes = plt.subplots(n_clusters + 1, 1, figsize=(18, 4 * (n_clusters + 1)))
    for i in range(n_clusters + 1):
        if i == 0:
            image_path = os.path.join(folder_name, 'raw_data.png')
        else:
            image_path = os.path.join(folder_name, f'class_{i}.png')
        image = plt.imread(image_path)
        axes[i].imshow(image)
        axes[i].axis('off')

    # 调整子图间距
    plt.subplots_adjust()
    # 调整布局
    plt.tight_layout(pad=0)
    # 显示图像叠加后的结果
    folder_name = "png_images"
    merged_image_path = os.path.join(folder_name, 'merged_image.png')
    plt.savefig(merged_image_path, dpi=80)



    his_data = [con_his_list_class, len_his_list_class, edges_2dhis_class, hist_class]
    return his_data, plot_paths, merged_image_path


# 绘图函数
def draw_figure(distance, conductance, length, bins_1Dcon=1000,
                low_cut_1Dcon=-6, for_title=None,
                high_cut_1Dcon=1, bins_2dhis_x=500, bins_2dhis_y=500, distance_0=-0.2, distance_1=1.2,
                conductance_0=1, conductance_1=-6.5, bins_1Dlen=100, label=None,
                show=False, color_1d='blue', color_2d='viridis', red=99, ivData=False, his_1d_dis_0=0, his_1d_dis_1=1):
    # 显示原始图像‘
    if show:
        plt.close('all')

    conductance, distance = new_data(distance, conductance, ivData=ivData)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    bin_edges_con, histogram_con, con_his_list = calculate_his(conductance, bins_1Dcon=bins_1Dcon,
                                                               low_cut_1Dcon=low_cut_1Dcon,
                                                               high_cut_1Dcon=high_cut_1Dcon)
    axes[0].bar(bin_edges_con[:-1], histogram_con, width=np.diff(bin_edges_con), align='edge', color=color_1d)
    axes[0].set_title(
        '1D histogram')
    axes[0].set_xlabel('conductance')
    axes[0].set_ylabel('count')
    axes[0].grid(True)  # 显示网格线
    if for_title:
        len_conductance, n_clusters, i = for_title
        value = f"{float(len(conductance) / len_conductance) * 100:.1f}"
        axes[0].set_title(str(i + 1) + '/' + str(n_clusters) + ' class   ' + str(
            len(conductance)) + '/' + str(len_conductance) + ' ' + str(value) + ' %')

    # 确定所有子列表的最大长度
    max_length = max(len(sublist) for sublist in conductance)
    # 使用np.pad函数向子列表添加零填充
    conductance_padded = [np.pad(sublist, (0, max_length - len(sublist)), 'constant', constant_values=0) for sublist in
                          conductance]
    distance_padded = [np.pad(sublist, (0, max_length - len(sublist)), 'constant', constant_values=0) for sublist in
                       distance]
    # 将列表转换为NumPy数组
    conductance = np.array(conductance_padded)
    distance = np.array(distance_padded)

    # 绘制2D直方图
    hist, extent, edges_2dhis = calculate_2d_his(distance, conductance, bins_2dhis_x=bins_2dhis_x,
                                                 bins_2dhis_y=bins_2dhis_y,
                                                 distance_0=distance_0, distance_1=distance_1,
                                                 conductance_0=conductance_0, conductance_1=conductance_1, red=red)
    axes[1].set_title('2D histogram')
    im = axes[1].imshow(hist.T, origin='lower', extent=extent, aspect='auto', cmap=color_2d)
    axes[1].set_xlabel('distance')
    axes[1].set_ylabel('conductance')
    # aspect=auto将根据图像窗口的尺寸自动调整图像内容的宽高比。

    # 绘制1D距离直方图
    hist_counts, hist_bins, _ = axes[2].hist(length, range=[his_1d_dis_0, his_1d_dis_1], bins=bins_1Dlen, alpha=0.8,
                                             label='Data', color=color_1d)
    len_his_list = list(zip(hist_bins, hist_counts))

    axes[2].set_title('1D distance')
    axes[2].set_xlabel('distance')
    axes[2].set_ylabel('count')
    axes[2].grid(True)  # 显示网格线
    # 生成临时文件路径
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)  # 调整子图间距
    # 添加color bar
    plt.colorbar(im, ax=axes[1])
    # 保存图像至临时文件
    folder_name = "png_images"
    if label:
        path = os.path.join(folder_name, f'{label}.png')

    else:
        path = os.path.join(folder_name, 'temp_figure.png')
    plt.savefig(path)

    # image_path2 = os.path.join(folder_name, f'{label}.eps')
    # plt.savefig(image_path2, format='eps', dpi=80)
    # 分别提取 x 和 y 边界
    x_edges, y_edges = edges_2dhis

    # 使用网格生成二维坐标
    x_grid, y_grid = np.meshgrid(x_edges, y_edges)

    # 将网格转换为二维坐标点列表
    coordinates = np.column_stack((x_grid.ravel(), y_grid.ravel()))

    his_data = [con_his_list, len_his_list, coordinates, hist]
    return his_data


# ---------------------------模块一 裁剪（距离为从0到np.mean(data['length_array'])) 获得的数据是等长的
def single_tailor(distance, conductance, length, dis_low_cut_input, dis_high_cut_input):
    # 裁剪合理值

    if dis_high_cut_input == -1:
        dis_high_cut = float(np.mean(length))
    else:
        dis_high_cut = dis_high_cut_input
    if dis_low_cut_input == -1:
        dis_low_cut = float(0)
    else:
        dis_low_cut = dis_low_cut_input
    conductance_clusters = []
    distance_clusters = []

    for i in range(len(conductance)):
        dis_low_cut_index = max(np.where(distance[i] < dis_low_cut)[0])
        dis_high_cut_index = max(np.where(distance[i] < dis_high_cut)[0])
        conductance_single_clusters = conductance[i][dis_low_cut_index:dis_high_cut_index]
        conductance_clusters.append(conductance_single_clusters)

        distance_single_clusters = distance[i][dis_low_cut_index:dis_high_cut_index]
        distance_clusters.append(distance_single_clusters)

    conductance_clusters = per_length(conductance_clusters)
    distance_clusters = per_length(conductance_clusters)

    return conductance_clusters, distance_clusters


# 使得数据等长PADDING
def per_length(data):
    max_length = max(len(item) for item in data)
    padded_array = []
    for item in data:
        if len(item) < max_length:
            padded_item = np.pad(item, (0, max_length - len(item)), mode='constant', constant_values=0)
        else:
            padded_item = item
        padded_array.append(padded_item)
    return padded_array


# ----------------------------模块二 投影至电导维度（保留了上升沿信息，因为其对峰值影响不大）
def single_1d_his(conductance, bins=100, conductance_low_cut=-7., conductance_high_cut=0.,
                  ivData=False):  # 可以设置局域范围进行聚类
    conductance_his_clusters = []
    for j in range(len(conductance)):
        # bin_counts2代表histogram的高度，bin_edges_con为坐标位置，做单曲线直方统计
        if ivData:
            conductance_down = conductance[j]
        else:
            conductance_down, _ = single_new_data(conductance[j])  # 截取下降沿
        bin_counts2, _ = np.histogram(conductance_down, bins=bins,
                                      range=(conductance_low_cut, conductance_high_cut))  # 取目标范围
        conductance_his_clusters.append(bin_counts2)
    return conductance_his_clusters


# ----------------------------模块三 投影至二维保留了上升沿信息，length_range( float(0) - float(np.mean(length) + 0.2)))，conductance_range 为 conductance_low_cut 到 conductance_high_cut
def single_2d_his(distance, conductance, length, dis_low_cut_input, dis_high_cut_input, bins, conductance_low_cut=-7.,
                  conductance_high_cut=1.):
    if dis_high_cut_input == -1:
        dis_high_cut = float(np.mean(length) + 0.2)
    else:
        dis_high_cut = dis_high_cut_input
    if dis_low_cut_input == -1:
        dis_low_cut = float(0)
    else:
        dis_low_cut = dis_low_cut_input

    range_x = [dis_low_cut, dis_high_cut]  # x 轴范围
    range_y = [conductance_low_cut, conductance_high_cut]  # y 轴范围
    his2d_clusters = []
    for i in range(len(conductance)):
        # 二维直方图
        H, _, _ = np.histogram2d(distance[i], conductance[i], bins=bins, range=[range_x, range_y])
        H = H.reshape(-1)  # 合并为一维数组
        his2d_clusters.append(H)
    return his2d_clusters


# ---------------------------模块四 对每条曲线通过模块一后取10个散点，内积其对应的（length，conductance）
# 线性插值
def single_linear_interpolation(distance, conductance, length, dis_low_cut_input, dis_high_cut_input, point_numbers=30,
                                use_weight=False):  # 取多少个点
    mean_length = np.mean(length)
    points = []
    con_length = []
    conductance_clusters, distance_clusters = single_tailor(distance, conductance, length, dis_low_cut_input,
                                                            dis_high_cut_input)
    for i in range(len(conductance)):
        interval = len(distance_clusters[i]) // point_numbers
        indices_length = np.where(distance[i] == length[i])[0]
        # 如果为零的话赋值为均值
        if indices_length.size > 0:
            conductance_length = conductance[i][indices_length][0]
        else:
            conductance_length = mean_length
        point = (conductance_clusters[i][::interval])
        con_length.append(abs(conductance_length))
        points.append(point)
    con_length = (con_length - min(con_length)) / (max(con_length) - min(con_length))
    points = (np.array(points))
    if use_weight:
        arr = np.array(con_length).reshape((len(con_length), 1))
        ons_arr = np.ones((len(con_length), len(points[0])))
        con_length = arr * ons_arr
        points = con_length * points
    return points


# --------------------------------------- 单个曲线绘图
def single_trance(distance, conductance, length, i, dis_low_cut_input, dis_high_cut_input, low_cut_conductance,
                  high_cut_conductance, color_1d='blue', color_2d='viridis', single_2d_his_bin=100,
                  single_1d_his_bin=100, ivData=False):
    conductance_array = conductance[i]
    distance_array = distance[i]
    length_array = length[i]

    if dis_high_cut_input == -1:
        dis_high_cut = float(np.mean(length))
    else:
        dis_high_cut = dis_high_cut_input
    if dis_low_cut_input == -1:
        dis_low_cut = float(0)
    else:
        dis_low_cut = dis_low_cut_input

    # ----------------------------------------绘制第一个子图 源数据图
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    # indices_length = np.where(distance_array == length_array)[0]
    # 画虚线 distance = distance_array
    # axes[0, 0].axvline(x=distance_array[indices_length], linestyle='dashed', color='g')
    axes[1, 1].axvline(x=dis_low_cut, linestyle='--', color='gray')
    axes[0, 1].axvline(x=dis_low_cut, linestyle='--', color='gray')
    axes[1, 1].axvline(x=dis_high_cut, linestyle='--', color='gray')
    axes[0, 1].axvline(x=dis_high_cut, linestyle='--', color='gray')
    axes[0, 1].axvspan(dis_low_cut, dis_high_cut, color='r', alpha=0.2)

    axes[0, 0].plot(distance_array, conductance_array, label='length = ' + str(round(length_array, 4)), color=color_1d)
    axes[0, 0].set_xlabel('Distance')  # 设置 x 轴标签
    axes[0, 0].set_ylabel('Conductance')  # 设置 y 轴标签
    axes[0, 0].set_title('Raw data & length')  # 设置标题
    axes[0, 0].grid(True)  # 显示网格线
    axes[0, 0].legend()  # 创建图例

    # ----------------------------------------绘制第二个子图：1D conductance直方图

    if ivData:
        conductance_array_cut = conductance_array
        distance_array_cut = distance_array
    else:
        conductance_array_cut, min_index = single_new_data(conductance_array)  # 截取下降沿
        distance_array_cut = distance_array[:min_index]

    axes[1, 0].axvline(x=low_cut_conductance, linestyle='--', color='gray')
    axes[1, 1].axhline(y=low_cut_conductance, linestyle='--', color='gray')
    axes[1, 0].axvline(x=high_cut_conductance, linestyle='--', color='gray')
    axes[1, 1].axhline(y=high_cut_conductance, linestyle='--', color='gray')
    axes[1, 0].axvspan(low_cut_conductance, high_cut_conductance, color='r', alpha=0.2)
    n_con, bin_con, _ = axes[1, 0].hist(conductance_array_cut, bins=single_1d_his_bin, color=color_1d)
    hist1d = list(zip(n_con, bin_con))
    axes[1, 0].set_xlabel('Conductance')  # 设置 x 轴标签
    axes[1, 0].set_ylabel('Count')  # 设置 y 轴标签
    axes[1, 0].set_title('Tailor & 1 D HIS (recommended)')  # 设置标题
    axes[1, 0].grid(True)  # 显示网格线

    # ----------------------------------------绘制第三个子图：切片图与取散点图
    # 线性插值
    x_interp = np.linspace(dis_low_cut, dis_high_cut, single_2d_his_bin)  # 插值点的 x 坐标
    y_interp = np.interp(x_interp, distance_array_cut, conductance_array_cut)  # 根据插值点的 x 坐标，计算对应的插值点的 y 坐标

    # 绘制原始曲线和折线近似
    axes[0, 1].plot(distance_array_cut, conductance_array_cut, color=color_1d)
    axes[0, 1].plot(x_interp, y_interp, 'o-')
    axes[0, 1].set_xlabel('Distance')
    axes[0, 1].set_ylabel('Conductance')
    axes[0, 1].set_title('Tailor data & interpolation')
    axes[0, 1].grid(True)

    # ----------------------------------------绘制第四个字图 2D直方图
    # 设置网格参数

    range_x = [distance_array_cut.min(), distance_array_cut.max()]  # x 轴范围
    range_y = [conductance_array_cut.min(), conductance_array_cut.max()]  # y 轴范围

    # 绘制二维直方图
    hist, xedges, yedges, _ = plt.hist2d(distance_array_cut, conductance_array_cut, bins=single_2d_his_bin,
                                         range=[range_x, range_y],
                                         cmap=color_2d)
    edges_2d = list(zip(xedges, yedges))
    # 设置标签和标题
    axes[1, 1].set_xlabel('Distance')
    axes[1, 1].set_ylabel('Conductance')
    axes[1, 1].set_title('Tailor & 2 D HIS')
    rect = Rectangle((dis_low_cut, low_cut_conductance), (dis_high_cut - dis_low_cut),
                     (high_cut_conductance - low_cut_conductance), linewidth=1, edgecolor='r', facecolor='none')
    axes[1, 1].add_patch(rect)
    # 添加color bar
    plt.colorbar(ax=axes[1, 1])
    plt.subplots_adjust(hspace=0.3)  # 调整子图间距
    # 生成临时文件路径
    folder_name = "png_images"
    image_path = os.path.join(folder_name, 'single_figure.png')
    plt.savefig(image_path)


    image_path2 = os.path.join(folder_name, 'SINGLEZ.eps')
    plt.savefig(image_path2, format='eps', dpi=400)

    single_trance_data = [distance_array, conductance_array]
    data_save = [hist1d, hist, edges_2d, single_trance_data]
    return image_path, data_save


def single_trance_iv(biasVData, currentData, i=0, bins=50):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.7))

    axes[1].hist(currentData[i], bins=bins, color='g')
    axes[1].set_xlabel('currentData')  # 设置 x 轴标签
    axes[1].set_ylabel('Count')  # 设置 y 轴标签
    axes[1].set_title('1 D HIS ')  # 设置标题
    axes[1].grid(True)  # 显示网格线

    axes[0].plot(biasVData[i], currentData[i])
    axes[0].set_xlabel('biasVData')
    axes[0].set_ylabel('currentData')
    axes[0].set_title('1 D LINE')
    axes[0].grid(True)
    # 绘制二维直方图
    axes[2].hist2d(biasVData[i], currentData[i], bins=bins)
    axes[2].set_xlabel('biasVData')
    axes[2].set_ylabel('currentData')
    axes[2].set_title('2 D HIS')

    folder_name = "png_images"
    image_path = os.path.join(folder_name, 'single_figure_iv.png')
    plt.savefig(image_path)
    return image_path


def save_data(data, paths):
    def save_single_data(data, path, formatter, delimiter='\t', end='\n'):
        """通用数据保存函数"""
        with open(path, 'w') as file:
            for item in data:
                formatted_line = delimiter.join(formatter(element) for element in item)
                file.write(formatted_line + end)

    # 数据保存逻辑
    data_0, data_1, data_2, data_3 = data
    save_path_con, save_path_len, save_path_2d_edges, save_path_2d_hist = paths

    # 保存数据 0 和 1，保留两位小数
    save_single_data(data_0, save_path_con, lambda x: f'{x:.2f}')
    save_single_data(data_1, save_path_len, lambda x: f'{x:.2f}')
    save_single_data(data_2, save_path_2d_edges, lambda x: f'{x:.2f}')

    # 保存数据 3，格式化为整齐的整数行
    save_single_data(data_3, save_path_2d_hist, lambda x: f'{int(x):<6}', delimiter=' ', end='\n')


def signal_window(text):
    import pymsgbox
    # 弹出信息窗口
    pymsgbox.alert(text, 'Tips')


def hamming_distance(str1, str2):
    # 确保两个字符串长度相同
    if len(str1) != len(str2):
        raise ValueError("字符串长度不一致")
    # 计算汉明距离
    distance = (np.abs(str1 - str2) > 0.016).astype(int)
    distance = np.sum(distance)
    return distance


# 预处理....
def MPVC_parameter(input_data):
    # R 为个点绝对值之和最大值,即为隧穿图
    R_abs = np.sum(np.abs(input_data[0]))
    R = input_data[0]
    # Ym 以及平均距离delta
    Y = []
    delta_X = []
    for_cluster = []
    for single_trance in input_data:
        R_pre = np.sum(np.abs(single_trance))
        if R_abs < R_pre:
            R_abs = R_pre
            R = single_trance
    # R= np.sum(input_data, axis=0)/len(input_data)
    #  得到隧穿R之后
    for single_trance in input_data:
        delta_X.append(np.sum(np.abs(single_trance - R)))
        Y.append(single_trance - R)
    delta_X = delta_X / max(delta_X)
    # 计算向量的R范数（绝对值）
    R_absolute_value = np.linalg.norm(R)
    R_r = R / R_absolute_value
    for m in range(len(input_data)):
        # 计算向量的Ym范数（绝对值）
        ym_absolute_value = np.linalg.norm(Y[m]) + 10e-8  # 避免除零警告
        value = np.dot(R, Y[m])
        absolute_value = R_absolute_value * ym_absolute_value
        cos_sita_m = -(value / absolute_value)
        sita_m = np.arccos(cos_sita_m)
        # 计算向量的汉明距离
        y_rm = Y[m] / ym_absolute_value
        h_rm = hamming_distance(R_r, y_rm) / len(input_data[0])
        for_cluster.append([delta_X[m], sita_m, h_rm])
    for_cluster = np.array(for_cluster)
    return for_cluster

# def calculate_1d_len(length, bins_1Dlen=50):
#     # 拟合高斯分布,去掉边缘噪声
#     peak_value = []
#     x = []
#     y = []
#     hist_counts, hist_bins, _ = plt.hist(length, bins=bins_1Dlen)
#     # 拟合前去掉边界噪音
#     if len(length) > 200:
#         hist_counts_fit1 = hist_counts[1:]
#         hist_bins_fit1 = hist_bins[1:]
#         hist_counts_fit = hist_counts_fit1[:-2]
#         hist_bins_fit = hist_bins_fit1[:-2]
#         bin_centers = (hist_bins_fit[:-1] + hist_bins_fit[1:]) / 2
#         mean_guess = np.argmax(y)  # 使用峰值位置作为初始猜测
#         params, _ = curve_fit(gaussian, bin_centers, hist_counts_fit, p0=[mean_guess, len(x) / 6, 1.], maxfev=800000)
#         # 提取拟合参数
#         mean, std, amplitude = params
#         # 计算高斯曲线的范围
#         x_min = mean - 3 * std  # 设置为平均值减去 3 倍标准差
#         x_max = mean + 3 * std  # 设置为平均值加上 3 倍标准差
#
#         # 生成限制在范围内的 x 值
#         x = np.linspace(x_min, x_max, 30)
#
#         # 计算对应的 y 值
#         y = gaussian(x, mean, std, amplitude)
#
#         # 标出峰值位置
#         peak_index = np.argmax(y)
#         peak_value = x[peak_index]
#
#     # 生成直方图数据
#     len_his_list = list(zip(hist_bins, hist_counts))
#
#     return peak_value, x, y, length, bins_1Dlen, len_his_list
