InJunction — IV 新旧机器兼容版（Windows x64）

使用：
1. 解压本文件夹，双击 InJunction.exe，无须安装 Python。
2. 首次启动可能需要等待依赖解压完成。
3. 在 IV 页选择包含 .tdms 文件的文件夹，然后运行、保存。
4. 新机器 STM 通道使用短平台规则；已验证的旧 RT 通道自动保留历史长平台筛选规则。
5. 电流单位保留为 mA；logI 是 log10(|I/nA|)。电导阈值按原实验条件设置。

默认复核参数：bias_base=0.1，电导范围 [-5.0, -2.5]，关闭去电容。
本机真实样本验证：新样本 99/88/80 条正扫和同数量反扫；旧样本 32 条正扫和 32 条反扫。
Windows 云端使用构造的新旧格式进行回归；未上传原始实验数据。

本版本修复空结果崩溃、旧数据筛选行为、平均电导错位及反扫导数方向。
导数和去电容后的电导结果可能与旧版错误算法不同，详见源码 review 报告。

工作缓存及错误日志：%LOCALAPPDATA%\InJunction\
最终结果仍通过软件的 Save 功能保存到选择的数据目录。

构建来源见 build-commit.txt，EXE 的 SHA256 见 SHA256.txt。
本构建未做商业代码签名。Windows 如果显示发布者未知，请按所在单位的软件使用策略处理。
自动验收在 Windows Server 2022 / Python 3.10 x64 环境完成，涵盖实际 EXE 的 Qt 界面、
新旧格式的多进程读取与绘图、8 份 NPZ 回读，以及 NumPy/Numba/UMAP/tslearn 依赖。

开源组件声明见 LICENSE 和 licenses 文件夹。
