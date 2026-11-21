## 1.提出背景
- 传统3D重建方法SfM(Structure from Motion)过于复杂,计算速度慢
- SfM具体方法:先获取大量数据,再算出来每张图片的全局算子或者全局embedding以及局部embedding和局部特征点,随后再找到相似的图片(全局算子)然后它们进行局部特征点匹配(利用局部算子),再进行三角测量找到对应的3D点的位置,再估计相对位姿,再累加把一整张图的3D位姿都估计出来.
- DUSt3R,MASt3R:优化算法,与VGGT比较接近,但还是需要找两个图片进行匹配.它们无法做到直接把所有的图片信息直接进行一个整合进行直接的输出
- 在此之前无法做到端到端完成整个3D重建工作,最多只能端到端完成VGGT的某个工作比如稠密点云预测等
## 2.模型结构
![](./articlefig1.png)
- 输入:先对每张图片进行patchify工作将图片分成一堆小块,作者用的是DINOv2,将每一个patch转化为token最后得到了一个二维序列
- concat:经过加入camera param和register token(给每一帧做登记,在输出端被舍弃)后再concat再输入transformer
- alternating transformer:global attention 和frame attention交替,先做全局检索,在做一个帧内信息的检索,此时就需要用到register token
- transformer输出:相机输出会送入CameraHead,一个MLP进行最终预测
- DPT:输入是经过transformer层的concatenate到一起的token,输出是pixel level的points,DPT 把离散 token 恢复成和原图对齐的像素级特征，再拿去预测 depth map /point map /tracking feature,其本身只是一个decoder
- tracks:格式是一个feature map,每一个点对应一组特征,不能直接输出tracking,而需要一些postprocess去完成query到tracking的工作

**register token某种意义上也是一个位置编码,但VGGT删除了位置编码结构而设计了register token,用来标识第一帧(世界帧)和其余帧的不同**

**注:最后输出的所有图像的参数不管是3D点云还是其他的位姿都是相对于第一张图片而言的**
## 3.实验参数
1. Camera parameter
  - **内参 K**：焦距、主点，决定像素坐标怎么映射到相机坐标系射线；
  - **外参 R,t**：旋转 + 平移，代表这个相机在**世界坐标系**里的位置和朝向（位姿 /pose）。
2. Depth map
  - 定义：**每个像素的值是该点到相机光心的距离（相机坐标系下的深度）**。
- 特点：
  1. 是**相机局部量**，依赖当前相机；换一个相机，同一个 3D 点的 depth 就变了。
  2. 只有距离，没有 X/Y；想要 3D 坐标，必须用相机内参 K 做反投影
  再乘外参转到世界坐标系。
3. Point map
  **对图像上每一个像素，直接输出它在世界坐标系下的 (X,Y,Z) 三维坐标**
  和 depth map（深度图）对比：
  - depth map：每个像素只有**距离相机的深度值**，是相机坐标系下的径向距离；想要 3D 点，必须再用 K、外参反投影（unproject）；
  - point map：**直接就是世界坐标 3D 点**，不需要你再做反投影；而且 VGGT 的 point map 是**viewpoint-invariant（视角无关）**，所有视图的 point map 共享同一个世界坐标系（一般以第一张图为基准），天然跨视图一致。
4. tracking map
  对一张图上的每个像素，tracking map 记录：这个像素对应的 3D 点，在其他所有视图里会出现在哪个像素位置。
  也就是**像素级的跨视图匹配轨迹**
  形式：每个像素携带一个 track id / 或者直接存其他视图上的匹配坐标；同一个 3D 点在不同图像上的像素，拥有相同 track 标识。
## 4.实验细节
- 作者实际上是尝试过1.用cross attention 2.全局用global attention 但两种方式不是很好,第一种效果很差,第二种太吃显存
- 训练数据:运用了大量的synthetic data及适量的真实data相结合
- 实验1-camera pose estimation:VGGT达到了比过往所有模型都更好的实验精确度,且时间是最短的,但如果再加上bundle adjustment(BA)进行全局后处理优化后效果还能进一步优化
- 实验2-dense MVS estimation:即使模型不知道groundtruth的camera,它的表现也比知道groundtruth的camera的模型表现要好,相比于DUSt3R来说要好得多得多
- 实验3-point map estimation:表现优于其余模型,作者还发现如果直接去使用深度估计加上camera去重建出来的这个密集点云比他们直接去估计出来的3D点的位置效果还要好,及利用camera param来估计效果更好(各参数之间有关联)
## 5.局限性
- 对大规模场景来说显存要求很高,可能仍然不适用
- 对有时序信息无法融合进该框架
可在未来加入时序信息且加入一些memory将其变为一个无限延长的tracking的框架