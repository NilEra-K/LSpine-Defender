<template>
  <div class="severity-predict-container">
    <!-- 标题和介绍部分 -->
    <div class="severity-predict-header">
      <h2>疾病严重性预测</h2>
      <p class="description">通过上传压缩包，系统将自动分析并生成相关参数指标</p>
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
              <option value="algorithm3">算法3</option>
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
  </div>
</template>

<script>
// 添加 axios 导入
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
      selectedAlgorithm: ''  // 添加算法选择
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
            this.resultList = response.data.parameters.map(param => {
              return {
                row_id: param.row_id,
                normal_mild: param.normal_mild,
                moderate: param.moderate,
                severe: param.severe
              };
            });
          }
        }
      } catch (error) {
        console.error('预测失败:', error);
        alert('预测失败，请重试！');
      }
    }
  }
}
</script>

<style scoped>
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
  /* 确保列表容器占满可用宽度 */
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
</style>