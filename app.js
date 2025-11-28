// MNIST 手写数字识别应用
// 使用 ONNX Runtime Web 进行浏览器端推理

class MNISTApp {
    constructor() {
        this.canvas = document.getElementById('drawCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.previewCanvas = document.getElementById('previewCanvas');
        this.previewCtx = this.previewCanvas.getContext('2d');

        this.isDrawing = false;
        this.hasDrawn = false;
        this.session = null;

        this.init();
    }

    async init() {
        this.setupCanvas();
        this.setupEventListeners();
        this.initProbabilityBars();
        await this.loadModel();
    }

    setupCanvas() {
        // 设置画布样式
        this.ctx.fillStyle = '#000000';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.strokeStyle = '#ffffff';
        this.ctx.lineWidth = 20;
        this.ctx.lineCap = 'round';
        this.ctx.lineJoin = 'round';
    }

    setupEventListeners() {
        // 鼠标事件
        this.canvas.addEventListener('mousedown', (e) => this.startDrawing(e));
        this.canvas.addEventListener('mousemove', (e) => this.draw(e));
        this.canvas.addEventListener('mouseup', () => this.stopDrawing());
        this.canvas.addEventListener('mouseout', () => this.stopDrawing());

        // 触摸事件
        this.canvas.addEventListener('touchstart', (e) => this.startDrawing(e));
        this.canvas.addEventListener('touchmove', (e) => this.draw(e));
        this.canvas.addEventListener('touchend', () => this.stopDrawing());

        // 按钮事件
        document.getElementById('clearBtn').addEventListener('click', () => this.clearCanvas());
        document.getElementById('predictBtn').addEventListener('click', () => this.predict());
    }

    getCoordinates(e) {
        const rect = this.canvas.getBoundingClientRect();
        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;

        if (e.touches) {
            return {
                x: (e.touches[0].clientX - rect.left) * scaleX,
                y: (e.touches[0].clientY - rect.top) * scaleY
            };
        }
        return {
            x: (e.clientX - rect.left) * scaleX,
            y: (e.clientY - rect.top) * scaleY
        };
    }

    startDrawing(e) {
        e.preventDefault();
        this.isDrawing = true;
        const coords = this.getCoordinates(e);
        this.ctx.beginPath();
        this.ctx.moveTo(coords.x, coords.y);

        // 隐藏提示
        if (!this.hasDrawn) {
            this.hasDrawn = true;
            document.getElementById('canvasOverlay').classList.add('hidden');
        }
    }

    draw(e) {
        if (!this.isDrawing) return;
        e.preventDefault();

        const coords = this.getCoordinates(e);
        this.ctx.lineTo(coords.x, coords.y);
        this.ctx.stroke();
    }

    stopDrawing() {
        if (this.isDrawing) {
            this.isDrawing = false;
            this.ctx.closePath();
        }
    }

    clearCanvas() {
        this.ctx.fillStyle = '#000000';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        this.hasDrawn = false;
        document.getElementById('canvasOverlay').classList.remove('hidden');

        // 重置结果显示
        document.getElementById('predictionResult').textContent = '-';
        document.getElementById('confidenceResult').textContent = '-';

        // 清除预览
        this.previewCtx.fillStyle = '#000000';
        this.previewCtx.fillRect(0, 0, 28, 28);

        // 重置概率条
        this.updateProbabilityBars(new Array(10).fill(0));
    }

    initProbabilityBars() {
        const container = document.getElementById('probabilityBars');
        container.innerHTML = '';

        for (let i = 0; i < 10; i++) {
            const row = document.createElement('div');
            row.className = 'probability-row';
            row.innerHTML = `
                <span class="probability-label">${i}</span>
                <div class="probability-bar-container">
                    <div class="probability-bar" id="bar-${i}" style="width: 0%"></div>
                </div>
                <span class="probability-value" id="value-${i}">0%</span>
            `;
            container.appendChild(row);
        }
    }

    updateProbabilityBars(probabilities, predictedClass = -1) {
        for (let i = 0; i < 10; i++) {
            const bar = document.getElementById(`bar-${i}`);
            const value = document.getElementById(`value-${i}`);
            const prob = probabilities[i] * 100;

            bar.style.width = `${prob}%`;
            value.textContent = `${prob.toFixed(1)}%`;

            if (i === predictedClass) {
                bar.classList.add('highlight');
            } else {
                bar.classList.remove('highlight');
            }
        }
    }

    preprocessImage() {
        // 创建临时画布用于缩放
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = 28;
        tempCanvas.height = 28;
        const tempCtx = tempCanvas.getContext('2d');

        // 找到绘制内容的边界框
        const imageData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
        const bounds = this.getBoundingBox(imageData);

        if (!bounds) {
            return null; // 画布为空
        }

        // 计算缩放和居中
        const sourceWidth = bounds.maxX - bounds.minX;
        const sourceHeight = bounds.maxY - bounds.minY;
        const maxDim = Math.max(sourceWidth, sourceHeight);

        // 缩放到20x20，留出4像素边距
        const scale = 20 / maxDim;
        const scaledWidth = sourceWidth * scale;
        const scaledHeight = sourceHeight * scale;

        // 居中绘制
        const offsetX = (28 - scaledWidth) / 2;
        const offsetY = (28 - scaledHeight) / 2;

        tempCtx.fillStyle = '#000000';
        tempCtx.fillRect(0, 0, 28, 28);
        tempCtx.drawImage(
            this.canvas,
            bounds.minX, bounds.minY, sourceWidth, sourceHeight,
            offsetX, offsetY, scaledWidth, scaledHeight
        );

        // 更新预览画布
        this.previewCtx.drawImage(tempCanvas, 0, 0);

        // 获取像素数据并归一化
        const scaledImageData = tempCtx.getImageData(0, 0, 28, 28);
        const input = new Float32Array(1 * 1 * 28 * 28);

        // MNIST标准归一化参数（与训练时保持一致）
        const mean = 0.1307;
        const std = 0.3081;

        for (let i = 0; i < 784; i++) {
            // 使用红色通道（灰度图像R=G=B）
            // 应用与训练时相同的归一化: (pixel/255 - mean) / std
            const normalized = scaledImageData.data[i * 4] / 255.0;
            input[i] = (normalized - mean) / std;
        }

        return input;
    }

    getBoundingBox(imageData) {
        const data = imageData.data;
        const width = imageData.width;
        const height = imageData.height;

        let minX = width, minY = height, maxX = 0, maxY = 0;
        let hasContent = false;

        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const idx = (y * width + x) * 4;
                // 检查是否有非黑色像素
                if (data[idx] > 10 || data[idx + 1] > 10 || data[idx + 2] > 10) {
                    hasContent = true;
                    minX = Math.min(minX, x);
                    minY = Math.min(minY, y);
                    maxX = Math.max(maxX, x);
                    maxY = Math.max(maxY, y);
                }
            }
        }

        if (!hasContent) return null;

        // 添加一些边距
        const padding = 20;
        return {
            minX: Math.max(0, minX - padding),
            minY: Math.max(0, minY - padding),
            maxX: Math.min(width, maxX + padding),
            maxY: Math.min(height, maxY + padding)
        };
    }

    async loadModel() {
        const statusDot = document.querySelector('.status-dot');
        const statusText = document.getElementById('statusText');
        const predictBtn = document.getElementById('predictBtn');

        try {
            statusText.textContent = '正在加载模型...';

            // 加载ONNX模型
            this.session = await ort.InferenceSession.create('./mnist.onnx');

            statusDot.classList.remove('loading');
            statusDot.classList.add('ready');
            statusText.textContent = '模型已就绪';
            predictBtn.disabled = false;

            console.log('MNIST模型加载成功');
            console.log('输入:', this.session.inputNames);
            console.log('输出:', this.session.outputNames);

        } catch (error) {
            console.error('模型加载失败:', error);
            statusDot.classList.remove('loading');
            statusDot.classList.add('error');
            statusText.textContent = '模型加载失败: ' + error.message;
            predictBtn.disabled = true;
        }
    }

    softmax(arr) {
        const max = Math.max(...arr);
        const exp = arr.map(x => Math.exp(x - max));
        const sum = exp.reduce((a, b) => a + b, 0);
        return exp.map(x => x / sum);
    }

    async predict() {
        if (!this.session) {
            alert('模型尚未加载完成，请稍候');
            return;
        }

        if (!this.hasDrawn) {
            alert('请先绘制一个数字');
            return;
        }

        try {
            // 预处理图像
            const inputData = this.preprocessImage();
            if (!inputData) {
                alert('画布为空，请绘制一个数字');
                return;
            }

            // 创建输入张量
            const inputTensor = new ort.Tensor('float32', inputData, [1, 1, 28, 28]);

            // 运行推理
            const feeds = { [this.session.inputNames[0]]: inputTensor };
            const results = await this.session.run(feeds);

            // 获取输出
            const output = results[this.session.outputNames[0]];
            const logits = Array.from(output.data);

            // 应用softmax获取概率
            const probabilities = this.softmax(logits);

            // 找到最大概率的类别
            let maxProb = 0;
            let predictedClass = 0;
            for (let i = 0; i < 10; i++) {
                if (probabilities[i] > maxProb) {
                    maxProb = probabilities[i];
                    predictedClass = i;
                }
            }

            // 更新UI
            document.getElementById('predictionResult').textContent = predictedClass;
            document.getElementById('confidenceResult').textContent = `${(maxProb * 100).toFixed(1)}%`;

            // 更新概率条
            this.updateProbabilityBars(probabilities, predictedClass);

            console.log('预测结果:', predictedClass, '置信度:', maxProb);

        } catch (error) {
            console.error('预测失败:', error);
            alert('预测失败: ' + error.message);
        }
    }
}

// 页面加载完成后初始化应用
document.addEventListener('DOMContentLoaded', () => {
    window.app = new MNISTApp();
});
