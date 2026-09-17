# InJunction: STM-BJ data processing
> The UI interface is created by Mr.Wanderson-Magalhaes, more detail please refer:
> https://github.com/Wanderson-Magalhaes/Modern_GUI_PyDracula_PySide6_or_PyQt6
> some code from:XMe_DataAnalysis
> https://github.com/Pilab-XMU/XMe_DataAnalysis
> Python 3.10 ,The package refer to requirements.txt
> 
> 如果您在论文中使用了本软件，请引用以下文章作为引用来源，该工作首次展示了该软件在分子电子实验数据处理中的应用：

Wang, H.; et al. Electrochemical Gating Enables Reversible and High-Performance Molecular Rectification. J. Am. Chem. Soc. 2024, 146, 14, 10020–10028. https://doi.org/10.1021/jacs.4c13773
> If you use this software in your work, please cite the following article, which first demonstrated its application in single-molecule electronic data processing:

Wang, H.; et al. Electrochemical Gating Enables Reversible and High-Performance Molecular Rectification. J. Am. Chem. Soc. 2024, 146 (14), 10020–10028. https://doi.org/10.1021/jacs.4c13773
# Multiple approach
> The script invove 3 main data for STM-BJ,and we have developed various tpye ways for classing, clustering,drawing fig,and data processing.

This software has contributed to multiple top-tier publications within our research group. Proper citation is appreciated.
![the icon](https://github.com/wanHAnuy/wanHAnuy/blob/main/fig1.png)
> **OPEN MAIN PAGE**
> so fist, when you has prepared all files and environment,you can type that code on terminal to oackage it (about 150MB) or just run main.py to open:
```
pyinstaller -F --icon=icon.ico main.py --hidden-import matplotlib.backends.backend_ps
```
![the main page](https://github.com/wanHAnuy/wanHAnuy/blob/main/fig2.png)

> 1. CH SCAN! find the best number to cluster('cluster cacu'page)
> 2. IV Clustering('cluster cacu'page)
> 3. Merge saving('cluster cacu'page)
> 4. 'length' subsection
> 5. FN transform
> 6. IV data processing:'tdms'-->'npz'
> 7. IV data processing -- decapicity
> 8. psd data processing!
>    ......

![1](https://github.com/user-attachments/assets/0c8532b7-809c-450c-a46b-ff979eeb2c2b)
![2](https://github.com/user-attachments/assets/28a23120-c86c-48e6-a09a-913b4235986d)
![3](https://github.com/user-attachments/assets/9ce261f3-3095-4843-94c0-aa6edb3e8238)
![4](https://github.com/user-attachments/assets/361300a9-c61c-4c2b-9984-e5ea3ec01a13)
![5](https://github.com/user-attachments/assets/a0afe283-7f20-435b-9285-bf930b018426)
![6](https://github.com/user-attachments/assets/fd1f2848-c874-41bd-8b2f-47f372e6a42e)
![7](https://github.com/user-attachments/assets/7046af22-ef90-4613-b15d-75028a127f44)
![8](https://github.com/user-attachments/assets/761fe70b-1cde-4184-89d3-add9ce5192e9)
![9](https://github.com/user-attachments/assets/976f94de-22ed-4368-ad2d-5a4ded5413f9)
![10](https://github.com/user-attachments/assets/d71094a5-e6c1-4a6b-b102-4c0f1b20c962)
![11](https://github.com/user-attachments/assets/ba71509a-781e-479c-954b-165d4031e19f)
![12](https://github.com/user-attachments/assets/1e94a65b-b2d9-43ea-ac9d-b824125ca1cf)
![13](https://github.com/user-attachments/assets/25a9f7d4-e4cc-47a4-85da-51edf18faac8)



## IV compatibility update

For short check plateaus from newer acquisition systems, diagnostics, regression coverage,
and current GUI packaging limitations, see [IV compatibility review](docs/IV_COMPATIBILITY_REVIEW.md).
A headless processor is available without the Qt assets:

```sh
python -m pip install -r requirements-iv.txt
python iv_batch.py "path/to/TDMS-folder" --output "path/to/new-output-folder"
```

Run regression tests with `python -m pytest -q tests -W error::RuntimeWarning` after installing pytest.
