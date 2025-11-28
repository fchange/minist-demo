# MNIST 手写数字识别 Web 应用

基于 ONNX Runtime Web 的纯前端 AI 应用，实现手写数字识别功能。无需服务器，所有推理计算在浏览器端完成。

## 项目演示

在画布上绘制 0-9 的数字，AI 模型会实时识别并显示：
- 预测结果和置信度
- 每个数字的概率分布（柱状图）
- 模型输入的 28x28 灰度图预览

## 核心原理

### 1. 模型训练（PyTorch）

使用 PyTorch 在 MNIST 数据集上训练一个卷积神经网络（CNN）：

```
输入层 (28x28x1)
  ↓
卷积层1 (32通道) + BatchNorm + ReLU + MaxPool → 14x14x32
  ↓
卷积层2 (64通道) + BatchNorm + ReLU + MaxPool → 7x7x64
  ↓
Dropout (0.25) → 展平 → 全连接层1 (128) + ReLU + Dropout (0.5)
  ↓
全连接层2 (10) → 输出 logits
```

**关键步骤**：
- 数据归一化：`(pixel/255 - 0.1307) / 0.3081`
- 损失函数：CrossEntropyLoss
- 优化器：Adam (lr=0.001)
- 训练轮数：5 epochs
- 准确率：~98%

### 2. 模型导出（ONNX）

将训练好的 PyTorch 模型导出为 ONNX 格式：

```python
torch.onnx.export(
    model,
    dummy_input,              # 示例输入 (1, 1, 28, 28)
    'mnist.onnx',
    opset_version=12,         # ONNX 算子版本
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={...}        # 支持动态批次大小
)
```

**ONNX 优势**：
- 跨平台兼容（浏览器、移动端、服务器）
- 高性能推理引擎
- 模型大小仅 ~300KB

### 3. 浏览器推理（ONNX Runtime Web）

前端使用 ONNX Runtime Web 加载模型并执行推理：

```javascript
// 加载模型
session = await ort.InferenceSession.create('./mnist.onnx');

// 预处理：画布 → 28x28 灰度图 → 归一化
input[i] = (pixel/255 - 0.1307) / 0.3081;

// 推理
const inputTensor = new ort.Tensor('float32', input, [1, 1, 28, 28]);
const results = await session.run({ input: inputTensor });

// 后处理：logits → softmax → 概率分布
const probabilities = softmax(results.output.data);
```

**关键技术点**：
- **图像预处理**：边界框检测 → 缩放到 20x20 → 居中到 28x28
- **归一化一致性**：推理时必须使用与训练时相同的归一化参数
- **实时推理**：浏览器端计算，无网络延迟

## 项目结构

```
minist-demo/
├── train_mnist.py      # 模型训练脚本
├── mnist.onnx          # 导出的 ONNX 模型
├── index.html          # 网页结构
├── app.js              # 前端逻辑（推理引擎）
├── style.css           # 界面样式
└── README.md           # 项目文档
```

## 文件详解

### `train_mnist.py` - 模型训练脚本

**功能**：训练 CNN 模型并导出为 ONNX 格式

**核心组件**：
- `MNISTNet` 类：定义 CNN 架构（2 层卷积 + 2 层全连接）
- `train_model()` 函数：
  - 加载 MNIST 数据集（60,000 训练 + 10,000 测试）
  - 数据增强与归一化
  - 训练循环（5 epochs）
  - 测试集验证
- `export_onnx()` 函数：
  - 导出单文件 ONNX 模型
  - 验证模型完整性
  - 打印模型信息

**使用方法**：
```bash
# 安装依赖
pip install torch torchvision onnx

# 运行训练
python train_mnist.py
```

**输出**：
- `mnist.onnx` - 训练好的模型文件（~300KB）
- 控制台打印：每个 epoch 的损失和准确率

---

### `mnist.onnx` - ONNX 模型文件

**格式**：ONNX (Open Neural Network Exchange)

**规格**：
- **输入**：`input` - float32[1, 1, 28, 28]（批次, 通道, 高, 宽）
- **输出**：`output` - float32[1, 10]（10 个类别的 logits）
- **大小**：~300KB
- **算子版本**：ONNX opset 12

**特性**：
- 包含所有权重参数（卷积核、偏置、BatchNorm 参数）
- 支持动态批次大小（可一次预测多个图像）
- 经过常量折叠优化

---

### `index.html` - 网页结构

**功能**：定义应用界面布局

**主要区块**：
1. **Header**：标题和副标题
2. **Canvas Section**：
   - 绘图画布（280x280）
   - 清除/识别按钮
3. **Result Section**：
   - 识别结果卡片（大数字显示）
   - 概率分布图（10 个柱状图）
   - 模型输入预览（28x28 灰度图）
4. **Footer**：
   - 模型加载状态指示器
   - 技术栈说明

**技术栈引用**：
- ONNX Runtime Web 1.16.3（CDN 加载）
- 纯 HTML5 Canvas API

---

### `app.js` - 前端推理引擎

**功能**：处理用户交互、图像预处理、模型推理

**核心类 `MNISTApp`**：

#### 1. 画布管理
- `setupCanvas()` - 初始化画布（黑底白笔）
- `startDrawing()` / `draw()` / `stopDrawing()` - 绘图事件处理
- `clearCanvas()` - 清空画布并重置结果

#### 2. 图像预处理（关键！）
- `preprocessImage()` - 主流程：
  ```javascript
  原始画布 (280x280)
    ↓ getBoundingBox() - 找到数字边界
    ↓ 裁剪并缩放到 20x20
    ↓ 居中到 28x28（留 4px 边距）
    ↓ 归一化：(pixel/255 - 0.1307) / 0.3081
    ↓ 输出 Float32Array[1, 1, 28, 28]
  ```

- `getBoundingBox()` - 边界框检测：
  - 扫描画布找到非黑色像素的最小/最大坐标
  - 添加 20px 边距防止数字被裁切

**关键技术点**：
- **归一化参数必须一致**：训练和推理都用 `mean=0.1307, std=0.3081`
- **缩放策略**：保持宽高比，缩放到 20x20（MNIST 标准）
- **居中对齐**：数字始终在 28x28 画布中心

#### 3. 模型推理
- `loadModel()` - 异步加载 ONNX 模型
- `predict()` - 主推理流程：
  ```javascript
  1. 预处理图像 → Float32Array
  2. 创建 ONNX Tensor
  3. 运行推理 → logits
  4. Softmax 转换 → 概率
  5. 更新 UI（结果、置信度、柱状图）
  ```

- `softmax()` - 数值稳定的 softmax 实现：
  ```javascript
  // 减去最大值防止数值溢出
  const max = Math.max(...arr);
  const exp = arr.map(x => Math.exp(x - max));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map(x => x / sum);
  ```

#### 4. UI 更新
- `initProbabilityBars()` - 动态生成 10 个概率柱状图
- `updateProbabilityBars()` - 更新柱状图宽度和高亮

---

### `style.css` - 界面样式

**功能**：现代化 UI 设计

**设计系统**：
- **CSS 变量**：统一管理颜色、阴影、间距
- **响应式布局**：Grid 布局，移动端自适应
- **视觉反馈**：
  - 按钮悬停效果（transform + box-shadow）
  - 状态指示器动画（loading/ready/error）
  - 柱状图过渡动画（0.3s ease）

**主要模块**：
1. **画布区域**：黑色背景，白色画笔，交叉光标
2. **结果卡片**：大数字显示（5rem），绿色置信度
3. **概率图**：渐变柱状图，最高概率高亮显示
4. **预览画布**：像素化渲染（image-rendering: pixelated）

**动画效果**：
- `pulse` - 加载状态呼吸灯效果
- `fadeIn` - 卡片淡入动画
- `transition` - 柱状图宽度过渡

## 使用指南

### 快速开始

1. **训练模型**（可选，已提供预训练模型）：
```bash
python train_mnist.py
```

2. **启动 Web 服务器**（避免 CORS 错误）：
```bash
# 使用 Python 自带的 HTTP 服务器
python -m http.server 8000

# 或使用 Node.js 的 http-server
npx http-server
```

3. **打开浏览器**：
```
http://localhost:8000
```

4. **开始识别**：
   - 在画布上绘制数字（0-9）
   - 点击"识别"按钮
   - 查看预测结果和概率分布

### 浏览器要求

- Chrome 90+
- Edge 90+
- Safari 15+
- Firefox 88+

**原因**：需要支持 WebAssembly、ES6+ 特性

## 技术栈

### 后端训练
- **PyTorch 2.0+** - 深度学习框架
- **torchvision** - MNIST 数据集加载
- **ONNX 1.14+** - 模型格式转换

### 前端推理
- **ONNX Runtime Web 1.16.3** - 浏览器端推理引擎
- **HTML5 Canvas API** - 图像绘制和处理
- **ES6+ JavaScript** - 现代 JavaScript 特性

### 设计
- **CSS Grid & Flexbox** - 响应式布局
- **CSS Custom Properties** - 主题变量
- **CSS Animations** - 流畅过渡效果

## 常见问题

### Q1: 为什么模型总是预测同一个数字？

**A**: 数据预处理不一致。确保推理时的归一化参数与训练时完全相同：

```javascript
// ✅ 正确
const mean = 0.1307;
const std = 0.3081;
input[i] = (pixel/255 - mean) / std;

// ❌ 错误（缺少归一化）
input[i] = pixel / 255;
```

### Q2: 模型加载失败怎么办？

**A**: 检查以下几点：
1. 使用 HTTP 服务器（不能直接打开 `file://` 协议）
2. 确保 `mnist.onnx` 文件在同级目录
3. 检查浏览器控制台的错误信息
4. 验证 ONNX 模型文件完整性

### Q3: 识别准确率不高？

**A**: 可能原因：
1. **绘制风格差异**：模型训练时用的是 MNIST 手写体，尽量模仿手写风格
2. **数字居中**：确保数字居中绘制，不要太靠边缘
3. **笔画粗细**：使用中等粗细的笔画（默认 20px）

### Q4: 如何提高模型准确率？

**A**: 改进方向：
1. **增加训练轮数**：从 5 epochs 增加到 10-20 epochs
2. **数据增强**：添加旋转、缩放、平移等增强
3. **模型架构**：增加卷积层或使用 ResNet 结构
4. **超参数调优**：调整学习率、dropout 比例

### Q5: 可以识别其他图像吗？

**A**: 当前模型仅针对 MNIST 手写数字训练。如需识别其他内容：
1. 准备新数据集
2. 修改模型架构（调整输出层类别数）
3. 重新训练
4. 更新前端预处理逻辑

## 性能指标

| 指标 | 数值 |
|------|------|
| 模型大小 | ~300KB |
| 加载时间 | <1s (首次) |
| 推理时间 | <50ms (浏览器端) |
| 训练准确率 | ~98% |
| 测试准确率 | ~98% |
| 支持设备 | 桌面 & 移动端 |

## 项目亮点

1. **零后端依赖**：所有计算在浏览器完成，无需服务器
2. **实时推理**：<50ms 延迟，即时反馈
3. **跨平台**：支持桌面和移动端浏览器
4. **隐私友好**：数据不上传，完全本地处理
5. **易于部署**：静态文件托管即可（GitHub Pages、Vercel 等）

## 扩展建议

### 功能扩展
- [ ] 添加撤销/重做功能
- [ ] 支持多笔画颜色选择
- [ ] 保存绘图为图片
- [ ] 批量识别（上传图片）
- [ ] 识别历史记录

### 技术升级
- [ ] 使用 WebGL 加速推理
- [ ] 集成 Web Worker 避免阻塞主线程
- [ ] 添加模型量化（INT8）减小体积
- [ ] PWA 支持（离线可用）
- [ ] 增加更多数据集（字母、符号等）

## 参考资料

- [ONNX Runtime Web 官方文档](https://onnxruntime.ai/docs/tutorials/web/)
- [PyTorch ONNX 导出指南](https://pytorch.org/docs/stable/onnx.html)
- [MNIST 数据集介绍](http://yann.lecun.com/exdb/mnist/)
- [Canvas API 参考](https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API)

## License

MIT License

---

**开发者**: Franco
**最后更新**: 2025-11-28
