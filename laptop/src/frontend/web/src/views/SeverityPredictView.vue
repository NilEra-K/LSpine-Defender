<template>
  <div class="severity-predict-container">
    <!-- 标题和介绍部分 -->
    <div class="severity-predict-header">
      <h2 style="font-size: large; font-family: 'Times New Roman', Times, serif;">🔥 疾病严重性预测</h2>
      <p class="description">
        * 通过上传压缩包，系统将自动分析并生成相关参数指标。<br>
        * 预测结果会显示在 Result List 部分，为一个包含 row_id、normal_mild、moderate、severe 四列的表格。<br>
        * 压缩包需要按照一定的格式进行组织，🔗请点击此处下载：<a href="/resource/test_images.zip">压缩包格式示例</a> 或 🖱️点击此处查看<a href="#"
          @click.prevent="toggleFormatCard">压缩包格式示例</a><br>
        * 算法包括 One-Stage 和 Multi-Stage 两种，其中 One-Stage 为 基于 KAN 和 CA 优化的 DenseNet 腰椎退行性病变分类预测，Multi-Stage 为基于多模型优化的腰椎退行性病变分类模型。
      </p>
    </div>
    <!-- 格式卡片 -->
    <div class="format-card" v-if="isFormatCardVisible">
      <div class="card-content">
        <h3>压缩包格式</h3>
        <pre>
file.zip/rar/tar/tar.gz
├─ test_images
│   └─ 44036939
│        ├─ 2828203845
│        │       1.dcm
│        │       2.dcm
│        │       3.dcm
│        │       ...
│        │      
│        ├─ 3481971518
│        │       1.dcm
│        │       2.dcm
│        │       3.dcm
│        │       ...
│        │      
│        └─ 3844393089
│                 1.dcm
│                 2.dcm
│                 3.dcm
│                 ...
└─ test_series_descriptions.csv
        </pre>
        <button @click="toggleFormatCard">关闭</button>
      </div>
    </div>

    <!-- 主要内容区域 -->
    <div class="severity-predict-content">
      <div class="severity-predict-box">
        <!-- 原始压缩包区域 -->
        <div class="predict-box">
          <div class="box-header">
            <h3>压缩包上传</h3>
          </div>
          <div class="upload-area" @drop.prevent="handleDrop" @dragover.prevent @click="triggerFileInput">
            <input type="file" ref="fileInput" style="display: none" @change="handleFileChange"
              accept=".zip, .tar, .gz">
            <div v-if="!originalFile" class="upload-placeholder">
              <i class="fas fa-cloud-upload-alt"></i>
              <p>点击或拖拽上传压缩包</p>
              <span class="upload-hint">仅支持 zip、tar、gz 格式</span>
            </div>
            <div v-else class="upload-placeholder">
              <i class="fas fa-file-archive"></i>
              <p>{{ originalFileName }}</p>
            </div>
          </div>
        </div>

        <!-- 操作按钮区域 -->
        <div class="predict-actions">
          <button class="action-btn upload" @click="handleUpload">
            <i class="fas fa-upload"></i>
            上传
          </button>
          <button class="action-btn clear" @click="handleClear">
            <i class="fas fa-trash-alt"></i>
            清除
          </button>
          <button class="action-btn predict" @click="handlePredict" :disabled="!selectedAlgorithm">
            <i class="fas fa-play"></i>
            预测
          </button>

          <!-- 算法选择下拉框 -->
          <div class="algorithm-select">
            <select id="algorithm-select" v-model="selectedAlgorithm">
              <option value="">请选择算法</option>
              <option value="algorithm1">One-Stage</option>
              <option value="algorithm2">Multi-Stage</option>
            </select>
          </div>
        </div>

        <!-- 结果列表区域 -->
        <div class="predict-result-list">
          <div class="box-header">
            <h3>Result List</h3>
          </div>
          <div class="list-container">
            <table>
              <thead>
                <tr>
                  <th>row_id</th>
                  <th>normal_mild</th>
                  <th>moderate</th>
                  <th>severe</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="(item, index) in resultList" :key="index">
                  <td>{{ item.row_id }}</td>
                  <td>{{ item.normal_mild }}</td>
                  <td>{{ item.moderate }}</td>
                  <td>{{ item.severe }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>

    <!-- 预测卡片 -->
    <div class="predict-loading" v-if="isPredicting">
      <div class="loading-card">
        <div class="loading-spinner"></div>
        <p>正在预测...</p>
      </div>
    </div>
  </div>
</template>

<script>
import axios from 'axios';

export default {
  name: 'SeverityPredictView',
  data() {
    return {
      originalFile: null,
      originalFileName: '',
      resultList: Array(75).fill().map(() => ({
        row_id: '-',
        normal_mild: '-',
        moderate: '-',
        severe: '-'
      })),
      selectedAlgorithm: '',
      isPredicting: false,        // 添加预测状态变量
      isFormatCardVisible: false  // 格式卡片状态变量
    }
  },
  methods: {
    triggerFileInput() {
      this.$refs.fileInput.click()
    },
    toggleFormatCard() {
      this.isFormatCardVisible = !this.isFormatCardVisible;
    },
    handleFileChange(event) {
      const file = event.target.files[0];
      if (file) {
        // 验证文件类型
        // console.log(file.type); // Edge 上传 Zip 文件显示 application/x-zip-compressed
        // Windows7 上传 Zip 文件显示 application/octet-stream
        const fileTypes = ['application/zip', 'application/x-tar', 'application/gzip', 'application/x-zip-compressed', 'application/octet-stream'];
        if (!fileTypes.includes(file.type)) {
          alert('请上传压缩包文件！');
          return;
        }

        // 验证文件大小（例如限制为 50MB）
        if (file.size > 50 * 1024 * 1024) {
          alert('文件大小不能超过 50MB！');
          return;
        }

        this.originalFile = file;
        this.originalFileName = file.name;

        // 可选：清空文件输入框的值，这样用户可以重复上传同一个文件
        event.target.value = '';
      }
    },
    handleDrop(event) {
      const file = event.dataTransfer.files[0]
      if (file && ['application/zip', 'application/x-tar', 'application/gzip', 'application/x-zip-compressed', 'application/octet-stream'].includes(file.type)) {
        this.originalFile = file;
        this.originalFileName = file.name;
      }
    },
    handleUpload() {
      // 实现上传逻辑
      this.$refs.fileInput.click();
    },
    handleClear() {
      this.originalFile = null
      this.originalFileName = ''
      this.resultList = Array(75).fill().map(() => ({
        row_id: '-',
        normal_mild: '-',
        moderate: '-',
        severe: '-'
      }))
    },
    async handlePredict() {
      // console.log("调用了handlePredict");
      // 实现预测逻辑
      if (!this.originalFile) {
        alert('请先上传压缩包！');
        return;
      }
      console.log(this.selectedAlgorithm);
      if (!this.selectedAlgorithm) {
        alert('请选择算法！');
        return;
      }

      this.isPredicting = true;  // 显示预测卡片

      try {
        // 创建 FormData
        const formData = new FormData();
        formData.append('file', this.originalFile);
        formData.append('algorithm', this.selectedAlgorithm);

        // 发送请求到后端
        const response = await axios.post('http://127.0.0.1:16022/api/v1/predict', formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        });

        // 处理响应
        if (response.data) {
          // 更新处理参数列表的逻辑
          if (response.data.parameters) {
            this.resultList = response.data.parameters.map(param => ({
              row_id: param.row_id,
              normal_mild: param.normal_mild,
              moderate: param.moderate,
              severe: param.severe
            }));
          }
        }
      } catch (error) {
        console.error('预测失败:', error);
        alert('预测失败，请重试！');
      } finally {
        this.isPredicting = false;  // 隐藏预测卡片
      }
    }
  }
}
</script>

<style scoped>
.format-card {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
}

.card-content {
  background: #fff;
  box-sizing: border-box;
  padding-left: 1rem;
  padding-right: 1rem;
  padding-bottom: 1.5rem;
  border-radius: 12px;
  box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
  text-align: left;
  width: 80%;
  max-width: 600px;
}

.card-content h3 {
  font-size: 1.5rem;
  margin-bottom: 1rem;
}

.card-content pre {
  background: #f8fafc;
  box-sizing: border-box;
  padding: 1rem;
  border-radius: 8px;
  font-size: 0.9rem;
  white-space: pre-wrap;
  word-wrap: break-word;
}

.card-content button {
  padding: 0.75rem 1.5rem;
  border: none;
  border-radius: 8px;
  color: white;
  cursor: pointer;
  transition: all 0.3s ease;
  font-weight: 500;
  display: inline-block;
  margin-top: 1rem;
  background-color: #2563eb;
}

.card-content button:hover {
  background-color: #1d4ed8;
}

.severity-predict-container {
  padding: 1rem 2rem;
  background-color: #f8fafc;
  height: auto;
}

.severity-predict-header {
  text-align: left;
  margin-bottom: 1.5rem;
}

.severity-predict-header h2 {
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

.severity-predict-content {
  background: #fff;
  border-radius: 12px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  padding: 1.5rem 2rem;
  margin-bottom: 1rem;
}

.severity-predict-box {
  display: flex;
  gap: 1.5rem;
  align-items: flex-start;
  flex-wrap: wrap;
}

.predict-box,
.predict-result-list {
  flex: 1;
  max-width: 800px;
  /* 保证最大宽度要超过容器宽度 */
  min-width: 300px;
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

.upload-area {
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

.upload-area img {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
}

.upload-placeholder {
  text-align: center;
  color: #64748b;
}

.upload-placeholder i {
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

.predict-actions {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  padding: 1rem 0;
  margin-top: 1rem;
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

.select-container {
  width: 200px;
  /* 根据需要调整宽度 */
  margin-top: 1rem;
}

#algorithm-select {
  width: 100%;
  padding: 8px 12px;
  font-size: 16px;
  border: 1px solid #ccc;
  border-radius: 4px;
  background-color: #fff;
  color: #333;
  cursor: pointer;
  outline: none;
  transition: border-color 0.3s, box-shadow 0.3s;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
  background-image: url('data:image/svg+xml;utf8,<svg fill="#666" height="30" viewBox="0 0 24 24" width="30" xmlns="http://www.w3.org/2000/svg"><path d="M7 10l5 5 5-5z"/><path d="M0 0h24v24H0z" fill="none"/></svg>');
  background-repeat: no-repeat;
  background-position: right 12px center;
  background-size: 16px 16px;
  text-align: center;
}

#algorithm-select:hover {
  border-color: #66af9a;
}

#algorithm-select:focus {
  border-color: #2563eb;
  box-shadow: 0 0 0 3px #66afea;
}

#algorithm-select option {
  padding: 8px 12px;
  color: #333;
}

#algorithm-select option:hover {
  background-color: #f1f5f9;
}

.predict-result-list .list-container {
  height: 400px;
  overflow-y: auto;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  background: #fff;
  width: 100%;
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
  .severity-predict-box {
    justify-content: center;
  }

  .predict-actions {
    flex-direction: row;
    width: 100%;
    justify-content: center;
    margin-top: 1rem;
    order: 3;
  }

  .predict-box,
  .predict-result-list {
    flex: 0 1 400px;
  }
}

@media (max-width: 768px) {
  .severity-predict-container {
    padding: 0.5rem 1rem;
  }

  .severity-predict-content {
    padding: 1rem;
  }

  .predict-box,
  .predict-result-list {
    max-width: 100%;
  }
}

/* 预测卡片样式 */
.predict-loading {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
}

.loading-card {
  background: #fff;
  padding: 2rem;
  border-radius: 12px;
  box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
  text-align: center;
}

.loading-spinner {
  border: 4px solid rgba(0, 0, 0, 0.1);
  border-top: 4px solid #2563eb;
  border-radius: 50%;
  width: 40px;
  height: 40px;
  animation: spin 1s linear infinite;
  margin: 0 auto 1rem;
}

@keyframes spin {
  0% {
    transform: rotate(0deg);
  }

  100% {
    transform: rotate(360deg);
  }
}
</style>
