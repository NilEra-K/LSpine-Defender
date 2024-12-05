<template>
    <div class="stage-one-container">
        <!-- 标题和介绍部分 -->
        <div class="section-header">
            <h2>一阶段分析</h2>
            <p class="description">通过上传腰椎图像，系统将自动分析并生成相关参数指标</p>
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
                        <div v-if="!originalImage" class="upload-placeholder">
                            <i class="fas fa-cloud-upload-alt"></i>
                            <p>点击或拖拽上传图片</p>
                            <span class="upload-hint">支持 jpg、png 格式</span>
                        </div>
                        <img v-else :src="originalImage" alt="Original Image">
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
                    <button class="action-btn predict" @click="handlePredict">
                        <i class="fas fa-play"></i>
                        预测
                    </button>
                </div>

                <!-- 预测结果图片区域 -->
                <div class="image-box">
                    <div class="box-header">
                        <h3>Image Result</h3>
                    </div>
                    <div class="result-area">
                        <img v-if="resultImage" :src="resultImage" alt="Result Image">
                        <div v-else class="result-placeholder">
                            <i class="fas fa-image"></i>
                            <p>预测结果将在这里显示</p>
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
export default {
    name: 'StageOneView',
    data() {
        return {
            originalImage: null,
            resultImage: null,
            resultList: Array(75).fill().map((_, i) => ({
                parameter: `参数 ${i + 1}`,
                value: '0'
            }))
        }
    },
    methods: {
        triggerFileInput() {
            this.$refs.fileInput.click()
        },
        handleFileChange(event) {
            const file = event.target.files[0]
            if (file) {
                this.loadImage(file)
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
            reader.readAsDataURL(file)
        },
        handleUpload() {
            // 实现上传逻辑
        },
        handleClear() {
            this.originalImage = null
            this.resultImage = null
            this.resultList = Array(75).fill().map((_, i) => ({
                parameter: `参数 ${i + 1}`,
                value: '0'
            }))
        },
        handlePredict() {
            // 实现预测逻辑
        }
    }
}
</script>

<style scoped>
.stage-one-container {
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
    .stage-one-container {
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