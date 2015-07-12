# mnist_toy
1.这是一个用简单的多层感知机学习mnist数据集的小程序，希望对各位初学机器学习的同学有所帮助。包括如下功能:
  a.查看mnist图像
  b.对mnist图像进行elastic distortion,affine distortion,并可以查看distortion的效果
  c.优化方法包括:SGD(随机梯度下降)，SGD linesearch(一维搜索的随机梯度下降法),CG(共轭梯度法),LFBGS(限域拟牛顿法)
  d.损失函数支持:L1,L2,ce(互熵),softmax
  e.支持L1和L2正则化技术，对不同层的感知机可以采用不同的正则方案
  f.可以在输入数据中加入高斯噪音
  g.支持逐层训练（自编码)
  h.支持训练多个MLP进行bagging
  
2.本程序的编译运行环境是：window 64x,并且要能够运行 cuda程序。

3.使用本程序请下载mnist_toy_exe文件夹中的所有文件，直接执行mnist_toy.exe。

4.mnist_toy_exe文件夹中有一个sample,是本人已经训练好的一个实例。这个是一个单隐层的、800个节点的神经网络，在本人的GTX980ti显卡上：
  使用固定minibatch大小，SGD linesearch,5分钟的训练时间，达到了99.3%以上的正确率。
  使用mini-batch递增技术和LBFGS迭代次数递减技术，可以在3分钟内达到99.3%的正确率

5.本程序默认的优化方式是结合了一维搜索的随机梯度下降法(SGD Linesearch),是一种二次规划方法,无需调节学习率。

6.建议的神经网络设置是：输出层采用sigmoid函数，中间层采用tanh函数，损失函数采用cross entropy,层数和节点数可以自己试验，其他参数可以采用默认设置。

有问题欢迎与本人联系:qq275244351

