## med-img-classify

这部分代码主要用于对 Sagittal T1、Sagittal T2/STIR 和 Axial T2 三类不同的图片进行分类，因为用户实际上不能很好的区分自己上传的 MRI 到底属于哪种具体的成像方式，而本系统同样希望这些对于用户来说是透明的，即用户只需要上传图片，剩下的事情只需要完全的交给系统即可。

执行流程：
1. 处理数据集: mic-making-dataset.py
2. 分割数据集: mic-split-dataset.py
3. 模型训练: mic-train.py
4. 模型推理: mic-predict.py

