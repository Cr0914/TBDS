环境配置：
ray=1.9.1
pytorch=1.10.1
torchvision=0.11.2
matplotlib=3.5.0
prettytable=2.5.0
apex: https://github.com/NVIDIA/apex
# *不同防御方案下的模型训练（以ResNet-18为例）**

```
python training/main_pytorch.py --scheme [defense schemes] --data [imagenet-folder with train] --result-root [path of checkpoints]
```

其中 --scheme [defense schemes] 表示选定的防御方案，可选方案为： pure RA DP-SGD GH 。 --result-root [path of checkpoints] 是存储经过训练的模型参数的路径。

例如：选择GH方案，并将训练好的模型放在测试文件夹中：

python training/main_pytorch.py --scheme GH --data /data/imagenet/train --result-root /data/test

## **1. 获取共享梯度**

通过在指定的模型参数和设置上获得相应的梯度，该梯度可用于以下 GIA。

### **1.1 梯度由单个 GPU 计算**

假设序列号为 0 的 GPU 用于梯度计算

```
python training/get_gradients.py --gpu 0 [training settings] --data [imagenet-folder with train] --results [path of results] --pretrained [checkpoint of trained model (.tar) or parameters of model (.pth)]
```

### **1.2 梯度由多个 GPU 计算**

可用 GPU 的数量可以通过在文件中 training/get_gradients.py 进行设置 os.environ["CUDA_VISIBLE_DEVICES"] 来调整。默认情况下，我们使用 4 个 GPU 来计算梯度。

```
python training/get_gradients.py [training settings] --data [imagenet-folder with train] --results [path of results] --pretrained [checkpoint of trained model (.tar) or parameters of model (.pth)] --multiprocessing-distributed --dist-url tcp://127.0.0.1:10023 --dist-backend nccl --world-size 1 --rank 0
```

### **可选的训练设置**

1.可以填写 [training settings] 的内容在这里详细列出。

2.基于 GA 的防御方案： --kernel-size-of-maxpool 19 --ra

3.向渐变添加噪声 （DP-SGD）： --enable-dp --sigma 0.01 --max-per-sample-grad_norm 1 --delta 1e-5

4.使用同步 BatchNorm（默认使用异步 BatchNorm）： --syncbn 。仅在多 GPU 环境中才有意义

5.设置局部迭代次数： --epochs [the number of local iterations] ，单次迭代为 --epochs 1

6.模拟重复标签（每个标签有 4 个重复标签，批次大小必须为 32）： --duplicate-label

7.设置批处理大小（默认值为 32）： -b [batch size] 。第79组： -b 79

8.Dropout：我们使用模型 VGG11 来测试Dropout的影响。默认情况下，Dropout 功能处于启用状态。如果要关闭它，则需要添加到 --model-eval 命令行中。例如，获取 Dropout 处于非活动状态时的梯度： --arch vgg11 --model-eval

## **2. 梯度反演攻击 （GIA）**

### **2.1 单 GPU**

```
python main_run.py --gpu 0 --checkpoint [path of the gradients(.tar)] --min-grads-loss --metric
```

### **2.2 多个 GPU**

可用 GPU 的数量可以通过在文件中 main_run.py 进行设置 os.environ["CUDA_VISIBLE_DEVICES"] 来调整。默认情况下，我们使用 4 个 GPU 来计算梯度

```
python main_run.py --gpu 0 --checkpoint [path of the gradients(.tar)] --min-grads-loss --metric --world-size 1 --rank 0 --dist-url tcp://127.0.0.1:10036 --dist-backend nccl --multiprocessing-distributed
```