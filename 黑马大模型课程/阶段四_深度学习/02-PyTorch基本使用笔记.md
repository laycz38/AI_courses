# PyTorch 基本使用

> 对应 PPT: `01-PyTorch基本使用.pptx`

## 知识点大纲

- 张量(Tensor)的概念和创建方式
- 张量 ↔ NumPy 互相转换
- 张量数值计算（基本运算、点乘、矩阵乘法）
- 张量运算函数（mean/sum/sqrt/exp/log）
- 张量索引操作（行列/列表/范围/布尔/多维）
- 张量形状操作（reshape/squeeze/unsqueeze/transpose/permute）
- 张量拼接（cat/stack）
- 自动微分模块（autograd）
- 案例：线性回归

---

## 一、什么是 PyTorch

基于 Python 的深度学习框架，核心数据结构是**张量(Tensor)**——就是多维矩阵。

- 类似 NumPy，但支持 **GPU 加速**（CUDA）
- 内置**自动微分**系统（训练神经网络的核心）
- **动态计算图**（比 TensorFlow 1.x 的静态图更灵活）

安装：
```bash
pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 二、张量创建

### 2.1 基本创建方式

```python
import torch

# 1. torch.tensor() — 根据指定数据创建
torch.tensor(10)                    # 标量
torch.tensor([10., 20., 30.])       # 从列表
torch.tensor(np.random.randn(2, 3)) # 从 numpy

# 2. torch.Tensor() — 根据形状创建（默认 float32）
torch.Tensor(2, 3)    # 2行3列，未初始化的值
torch.Tensor([10])    # 传入列表则创建指定元素的张量

# 3. 指定类型创建
torch.IntTensor(2, 3)       # int32
torch.FloatTensor(2, 3)     # float32
torch.DoubleTensor(2, 3)    # float64
torch.LongTensor(2, 3)      # int64
torch.ShortTensor(2, 3)     # int16
```

### 2.2 线性和随机张量

```python
# 线性
torch.arange(0, 10, 2)      # [0, 2, 4, 6, 8]，按步长
torch.linspace(0, 9, 10)    # 0到9均分10个点

# 随机
torch.manual_seed(100)       # 设随机种子，保证可复现
torch.randn(2, 3)            # 标准正态分布 N(0,1)
torch.initial_seed()          # 查看当前种子
```

### 2.3 全0 / 全1 / 指定值

```python
torch.zeros(2, 3)       # 全0
torch.ones(2, 3)        # 全1
torch.full([2, 3], 10)  # 全10

# _like 系列：根据已有张量的形状创建
torch.zeros_like(data)
torch.ones_like(data)
torch.full_like(data, 20)
```

---

## 三、类型转换

```python
# 方式一：.type()
data = data.type(torch.DoubleTensor)  # 转 float64

# 方式二：便捷方法
data.double()    # float64
data.float()     # float32
data.int()       # int32
data.long()      # int64
data.short()     # int16
data.half()      # float16
```

---

## 四、张量 ↔ NumPy

```python
# 张量 → NumPy（默认共享内存！）
arr = tensor.numpy()
arr = tensor.numpy().copy()  # 不共享内存

# NumPy → 张量
tensor = torch.from_numpy(arr)      # 共享内存
tensor = torch.tensor(arr)          # 不共享内存

# 标量张量取数值
val = tensor.item()  # 只有一个元素时用
```

---

## 五、数值计算

### 5.1 基本运算

```python
data.add(10)      # 等价 data + 10，不修改原数据
data.add_(10)     # 带下划线的就地修改，等价 data += 10
data.sub(100)     # 减
data.mul(100)     # 乘
data.div(100)     # 除
data.neg()        # 取负
```

### 5.2 点乘（Hadamard积）

```python
# 相同形状张量，对应位置相乘
torch.mul(a, b)   # 等价 a * b
```

### 5.3 矩阵乘法

```python
# 要求：(n, m) @ (m, p) → (n, p)
a @ b              # 运算符
torch.matmul(a, b)  # 函数形式，形状限制更宽松
```

---

## 六、运算函数

```python
data.mean()        # 均值
data.mean(dim=0)   # 按列
data.mean(dim=1)   # 按行
data.sum()         # 求和
torch.pow(data, 2) # 平方
data.sqrt()        # 平方根
data.exp()         # e^n
data.log()         # ln
data.log2()        # log2
data.log10()       # lg
```

---

## 七、索引操作

```python
# 简单行列
data[0]            # 第0行
data[:, 0]         # 第0列

# 列表索引
data[[0, 1], [1, 2]]          # (0,1)和(1,2)两个位置
data[[[0], [1]], [1, 2]]       # 0、1行的1、2列

# 范围索引
data[:3, :2]       # 前3行前2列
data[2:, :2]       # 第2行开始的前2列

# 布尔索引
data[data[:, 2] > 5]           # 第3列>5的行
data[:, data[1] > 5]           # 第2行>5的列

# 多维索引
data[0, :, :]      # 第0轴第一个
data[:, 0, :]      # 第1轴第一个
data[:, :, 0]      # 第2轴第一个
```

---

## 八、形状操作

```python
# reshape — 改形状，数据不变
data.reshape(1, 6)

# squeeze — 删除大小为1的维度（降维）
data.squeeze()

# unsqueeze — 在指定位置添加大小为1的维度（升维）
data.unsqueeze(dim=0)   # 在第0维前加
data.unsqueeze(dim=1)   # 在第1维前加
data.unsqueeze(dim=-1)  # 在最后一维后加

# transpose — 交换两个维度
torch.transpose(data, 1, 2)  # 交换维度1和2

# permute — 一次交换多个维度
data.permute([1, 2, 0])      # 把(3,4,5)变成(4,5,3)
```

---

## 九、拼接操作

```python
# cat — 在已有维度上拼接，不增加维度数
torch.cat([a, b], dim=0)  # 沿第0维拼接

# stack — 在新维度上拼接，增加一个维度
torch.stack([a, b], dim=0)  # 在第0维前新增维度
```

| | cat | stack |
|------|-----|-------|
| 维度数 | 不变 | +1 |
| 要求 | 其他维度大小相同 | 所有输入形状完全相同 |

---

## 十、自动微分（autograd）

**这是训练神经网络的核心机制。**

```python
# 1. 标量对张量求导
x = torch.tensor(10, requires_grad=True, dtype=torch.float32)
y = 2 * x ** 2        # y = 2x²
y.backward()           # 计算梯度
print(x.grad)          # y' = 4x, x=10 → 40

# 2. 向量需要先转为标量
x = torch.tensor([10, 20], requires_grad=True, dtype=torch.float32)
y = 2 * x ** 2
y.sum().backward()     # 必须先 .sum()
print(x.grad)          # tensor([40., 80.])

# 3. 梯度清零
x.grad.zero_()         # 梯度会累加，每次更新前要清！
```

### 梯度下降法求最优解

```python
# 求 y = x² + 20 的极小值
x = torch.tensor(10, requires_grad=True, dtype=torch.float32)
for i in range(1001):
    y = x ** 2 + 20          # 前向传播
    if x.grad is not None:
        x.grad.zero_()       # 梯度清零
    y.backward()             # 反向传播
    x.data = x.data - 0.01 * x.grad  # 更新参数: w = w - lr*grad
# 结果：x → 0, y → 20（极小值）

### detach() 的作用

```python
# requires_grad=True 的张量不能直接 .numpy()
x2 = x.detach()  # 生成新张量，共享数据但不追踪梯度
x2.numpy()       # OK
```

---

## 十一、线性回归案例

PyTorch 模型构建四步模板：

```
准备数据 → 构建模型 → 设置损失函数和优化器 → 训练循环
```

```python
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn, optim
from sklearn.datasets import make_regression

# 1. 准备数据
x, y, coef = make_regression(n_samples=100, n_features=1, noise=10,
                              coef=True, bias=14.5, random_state=0)
x, y = torch.tensor(x), torch.tensor(y)
dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 2. 构建模型
model = nn.Linear(in_features=1, out_features=1)

# 3. 损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-2)

# 4. 训练循环
for epoch in range(100):
    for train_x, train_y in dataloader:
        y_pred = model(train_x.float())
        loss = criterion(y_pred, train_y.reshape(-1, 1).float())

        optimizer.zero_grad()   # 梯度清零
        loss.backward()         # 反向传播
        optimizer.step()        # 更新参数
```

### API 对照表

| 手动实现 | PyTorch 替换 |
|---------|-------------|
| 假设函数 | `nn.Linear(in, out)` |
| 平方损失 | `nn.MSELoss()` |
| 优化器 | `optim.SGD(params, lr)` |
| 数据加载 | `DataLoader(dataset, batch_size, shuffle)` |

---

## 关键词

`PyTorch` `Tensor` `张量` `NumPy` `矩阵乘法` `自动微分` `autograd` `backward` `梯度下降` `DataLoader` `nn.Linear` `MSELoss` `SGD` `线性回归` `GPU` `CUDA`
