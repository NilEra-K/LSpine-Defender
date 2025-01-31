<template>
    <div class="imaging-predict-container">
        <!-- 标题和介绍部分 -->
        <div class="section-header">
            <h2 style="font-size: large; font-family: 'Times New Roman', Times, serif;">🔥 基于ResNet50的成像方式预测</h2>
            <p class="description">
                * 通过上传腰椎图像，系统将对输入图像进行分类。<br>
                * 分类类别为 Sagittal T1、Sagittal T2/STIR、Axial T2 三类。<br>
                * 分类结果会显示在 Image Result 和 Result List 部分，Image Result 会显示用户输入图片的最终的分类结果，Result List 会显示每个类别的概率。<br>
                * Tips: 当有预测结果时，可以点击 Image Result 显示/隐藏标签。
            </p>
        </div>

        <!-- 主要内容区域 -->
        <div class="content-wrapper">
            <div class="content-container">
                <!-- 原始图片区域 -->
                <div class="image-box">
                    <div class="box-header">
                        <h3>Origin Image</h3>
                    </div>
                    <div class="upload-area" @drop.prevent="handleDrop" @dragover.prevent @click="triggerFileInput">
                        <input type="file" ref="fileInput" style="display: none" @change="handleFileChange"
                            accept="image/*">
                        <div v-if="!originalImage && !isLoading" class="upload-placeholder">
                            <i class="fas fa-cloud-upload-alt"></i>
                            <p>点击或拖拽上传图片</p>
                            <span class="upload-hint">支持 jpg、png 格式</span>
                        </div>
                        <img v-else-if="originalImage" :src="originalImage" alt="Original Image">
                        <div v-else class="loading"></div>
                    </div>
                </div>

                <!-- 操作按钮区域 -->
                <div class="actions-column">
                    <button class="action-btn upload" @click="handleUpload">
                        <i class="fas fa-upload"></i>
                        上传
                    </button>
                    <button class="action-btn clear" @click="handleClear">
                        <i class="fas fa-trash-alt"></i>
                        清除
                    </button>
                    <button class="action-btn predict" @click="handlePredict" :disabled="isLoading">
                        <i class="fas fa-play"></i>
                        预测
                    </button>
                </div>

                <!-- 预测结果图片区域 -->
                <div class="image-box">
                    <div class="box-header">
                        <h3>Image Result</h3>
                    </div>
                    <div class="result-area" @click="toggleMask">
                        <img v-if="resultImage" :src="resultImage" alt="Result Image">
                        <div v-else-if="!isLoading" class="result-placeholder">
                            <i class="fas fa-image"></i>
                            <p>预测结果将在这里显示</p>
                        </div>
                        <div v-else class="loading"></div>
                        <!-- 遮罩和标签 -->
                        <div v-if="showMask" class="mask">
                            <div class="label">{{ predictionLabel }}</div>
                        </div>
                    </div>
                </div>

                <!-- 结果列表区域 -->
                <div class="result-list">
                    <div class="box-header">
                        <h3>Result List</h3>
                    </div>
                    <div class="list-container">
                        <table>
                            <thead>
                                <tr>
                                    <th>参数</th>
                                    <th>数值</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr v-for="(item, index) in resultList" :key="index">
                                    <td>{{ item.parameter }}</td>
                                    <td>{{ item.value }}</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>

<script>
// 添加 axios 导入
import axios from 'axios';

export default {
    name: 'ImagingPredictView',
    data() {
        return {
            originalImage: null,
            resultImage: null,
            resultList: Array(75).fill().map(() => ({
                parameter: '待测参数',
                value: '-'
            })),
            isLoading: false,  // 添加加载状态
            showMask: false,  // 控制遮罩显示
            predictionLabel: ''  // 预测标签
        }
    },
    methods: {
        triggerFileInput() {
            this.$refs.fileInput.click()
        },
        handleFileChange(event) {
            const file = event.target.files[0];
            if (file) {
                // 验证文件类型
                if (!file.type.startsWith('image/')) {
                    alert('请上传图片文件！');
                    return;
                }

                // 验证文件大小（例如限制为 5MB）
                if (file.size > 5 * 1024 * 1024) {
                    alert('文件大小不能超过 5MB！');
                    return;
                }

                this.loadImage(file);

                // 可选：清空文件输入框的值，这样用户可以重复上传同一个文件
                event.target.value = '';
            }
        },
        handleDrop(event) {
            const file = event.dataTransfer.files[0]
            if (file && file.type.startsWith('image/')) {
                this.loadImage(file)
            }
        },
        loadImage(file) {
            const reader = new FileReader()
            reader.onload = (e) => {
                this.originalImage = e.target.result
            }
            reader.onerror = () => {
                alert('图片加载失败，请重试！');
            };
            reader.readAsDataURL(file)
        },
        handleUpload() {
            // 实现上传逻辑
            this.$refs.fileInput.click();
        },
        handleClear() {
            this.originalImage = null
            this.resultImage = null
            this.resultList = Array(75).fill().map(() => ({
                parameter: '待测参数',
                value: '-'
            }))
            this.showMask = false;
            this.predictionLabel = '';
        },
        async handlePredict() {
            if (!this.originalImage) {
                alert('请先上传图片！');
                return;
            }
            try {
                this.resultImage = null
                this.resultList = Array(75).fill().map(() => ({
                    parameter: '待测参数',
                    value: '-'
                }))
                this.isLoading = true;  // 开始预测时设置加载状态
                this.predictionLabel = '';
                this.showMask = false;
                const base64Data = this.originalImage.split(',')[1];
                const blob = await fetch(`data:image/jpeg;base64,${base64Data}`).then(res => res.blob());
                const formData = new FormData();
                formData.append('image', blob, 'image.jpg');
                const response = await axios.post('http://127.0.0.1:16020/api/v1/mic-predict', formData, {
                    headers: {
                        'Content-Type': 'multipart/form-data'
                    }
                });
                if (response.data) {
                    this.resultImage = `data:image/jpeg;base64,${response.data.image}`;
                    this.predictionLabel = response.data.label;  // 获取预测标签
                    this.showMask = true;  // 显示遮罩和标签
                    console.log(response.data);
                    if (response.data.parameters) {
                        this.resultList = response.data.parameters.map(param => {
                            const [paramName, value] = Object.entries(param)[0];
                            return {
                                parameter: paramName,
                                value: value.toString()
                            };
                        });
                    }
                }
            } catch (error) {
                console.error('预测失败:', error);
                alert('预测失败，请重试！');
            } finally {
                this.isLoading = false;  // 预测完成后结束加载状态
            }
        },
        toggleMask() {
            // this.showMask = !this.showMask;  // 切换遮罩显示状态
            if (this.predictionLabel != '') {   // 当有预测结果时, 才能够切换遮罩显示状态
                this.showMask = !this.showMask; // 切换遮罩显示状态
            }
        }
    }
}
</script>

<style scoped>
.imaging-predict-container {
    padding: 1rem 2rem;
    background-color: #f8fafc;
    height: auto;
}

.section-header {
    text-align: left;
    margin-bottom: 1.5rem;
}

.section-header h2 {
    font-size: 1.8rem;
    color: #1a1a1a;
    margin-bottom: 0.25rem;
    font-weight: 600;
}

.description {
    color: #666;
    font-size: 1rem;
    font-family: 'Times New Roman', Times, serif;
    margin: 0;
}

.content-wrapper {
    background: #fff;
    border-radius: 12px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    padding: 1.5rem 2rem;
    margin-bottom: 1rem;
}

.content-container {
    display: flex;
    gap: 1.5rem;
    align-items: flex-start;
    flex-wrap: wrap;
}

.box-header {
    margin-bottom: 1rem;
    text-align: left;
}

.box-header h3 {
    font-size: 1.2rem;
    color: #2c3e50;
    font-weight: 600;
    margin: 0;
}

.image-box {
    flex: 1;
    max-width: 400px;
    min-width: 300px;
}

.upload-area,
.result-area {
    width: 100%;
    height: 400px;
    border: 2px dashed #e2e8f0;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    background: #f8fafc;
    position: relative;  /* 添加相对定位 */
}

.upload-area:hover {
    border-color: #2563eb;
    background: #f1f5f9;
}

.upload-area img,
.result-area img {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
}

.upload-placeholder,
.result-placeholder {
    text-align: center;
    color: #64748b;
}

.upload-placeholder i,
.result-placeholder i {
    font-size: 3rem;
    margin-bottom: 1rem;
    color: #94a3b8;
}

.upload-hint {
    font-size: 0.875rem;
    color: #94a3b8;
    margin-top: 0.5rem;
    display: block;
}

.actions-column {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    padding: 1rem 0;
    margin-top: 8rem;
    align-self: center;
}

.action-btn {
    padding: 0.75rem 1.5rem;
    border: none;
    border-radius: 8px;
    color: white;
    cursor: pointer;
    transition: all 0.3s ease;
    font-weight: 500;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    min-width: 120px;
}

.action-btn i {
    font-size: 1rem;
}

.action-btn.upload {
    background-color: #2563eb;
}

.action-btn.upload:hover {
    background-color: #1d4ed8;
}

.action-btn.clear {
    background-color: #dc2626;
}

.action-btn.clear:hover {
    background-color: #b91c1c;
}

.action-btn.predict {
    background-color: #059669;
}

.action-btn.predict:hover {
    background-color: #047857;
}

.result-list {
    flex: 1;
    max-width: 400px;
    min-width: 300px;
}

.list-container {
    height: 400px;
    overflow-y: auto;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    background: #fff;
}

table {
    width: 100%;
    border-collapse: collapse;
}

th,
td {
    padding: 0.75rem 1rem;
    text-align: left;
    border-bottom: 1px solid #e2e8f0;
    font-size: 0.875rem;
}

th {
    background-color: #f8fafc;
    font-weight: 600;
    color: #1a1a1a;
    position: sticky;
    top: 0;
}

td {
    color: #4b5563;
}

/* 添加加载状态样式 */
.loading {
    position: relative;
    pointer-events: none;
    opacity: 0.7;
}

.loading::after {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 30px;
    height: 30px;
    border: 3px solid #f3f3f3;
    border-top: 3px solid #3498db;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    0% {
        transform: translate(-50%, -50%) rotate(0deg);
    }

    100% {
        transform: translate(-50%, -50%) rotate(360deg);
    }
}

/* 预测按钮禁用状态 */
.action-btn.predict:disabled {
    background-color: #9ca3af;
    cursor: not-allowed;
}

/* 遮罩和标签样式 */
.mask {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.5);
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 1.2rem;
    font-weight: 600;
}

@media (max-width: 1280px) {
    .content-container {
        justify-content: center;
    }

    .actions-column {
        flex-direction: row;
        width: 100%;
        justify-content: center;
        margin-top: 1rem;
        order: 3;
    }

    .image-box,
    .result-list {
        flex: 0 1 400px;
    }
}

@media (max-width: 768px) {
    .imaging-predict-container {
        padding: 0.5rem 1rem;
    }

    .content-wrapper {
        padding: 1rem;
    }

    .image-box,
    .result-list {
        max-width: 100%;
    }
}
</style>