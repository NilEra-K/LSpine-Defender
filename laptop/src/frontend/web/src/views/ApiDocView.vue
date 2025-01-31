<template>
  <div class="api-docs">
    <div v-for="(api, index) in apis" :key="index" class="api-container">
      <div @click="toggleDetails(index)" class="api-header">
        <span :class="{'api-version': true, 'v1': api.version === 'V1', 'v2': api.version === 'V2'}">
          {{ api.version }}
        </span>
        <span class="api-spacer"></span>
        <span :class="{'api-method': true, 'get': api.method === 'GET', 'post': api.method === 'POST'}">
          {{ api.method }}
        </span>
        <span class="api-name">{{ api.name }}</span>
        <button class="toggle-button" :aria-expanded="!api.expanded">
          <i :class="api.expanded ? 'fas fa-chevron-up' : 'fas fa-chevron-down'" :style="{ transform: api.expanded ? 'rotate(360deg)' : 'rotate(0deg)' }"></i>
        </button>
      </div>
      <transition name="fade">
        <div v-if="api.expanded" class="api-details">
          <div v-for="(section, sectionIndex) in api.description" :key="sectionIndex" class="description-section">
            <p v-if="section.type === 'text'">{{ section.content }}</p>
            <div v-if="section.type === 'code'" class="code-block">
              <pre><code>{{ section.content }}</code></pre>
              <button class="copy-button" @click="copyCode(section.content)">复制</button>
            </div>
          </div>
        </div>
      </transition>
    </div>
    <!-- 修改后的提示框 -->
    <div v-if="toast.visible" class="copy-message" :class="{ visible: toast.visible }">
      <img :src="checkIcon" alt="Check Icon" class="check-icon">
      <span>{{ toast.message }}</span>
    </div>
  </div>
</template>

<script>
export default {
  name: 'ApiDocView',
  data() {
    return {
      apis: [
        {
          version: 'V1',
          method: 'GET',
          name: '成像方式预测 http://127.0.0.1:16020/api/v1/mic-predict',
          description: [
            { type: 'text', content: '成像方式预测 API' },
            { type: 'code', content: 
`try{
  const base64Data = this.originalImage.split(',')[1];
  const blob = await fetch(\`data:image/jpeg;base64,\${base64Data}\`).then(res => res.blob());
  const formData = new FormData();
  formData.append('image', blob, 'image.jpg');
  const response = await axios.post('http://127.0.0.1:16020/api/v1/mic-predict', formData, {
      headers: {
          'Content-Type': 'multipart/form-data'
      }
  });
  if (response.data) {
      this.resultImage = \`data:image/jpeg;base64,\${response.data.image}\`;
      this.predictionLabel = response.data.label;  // 获取预测标签
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
  // 处理其他逻辑
}` 
            },
            { type: 'text', content: '具体使用方式需要根据您的代码做出相应的更改' }
          ],
          expanded: false
        },
        {
          version: 'V1',
          method: 'GET',
          name: '疾病严重性预测 http://127.0.0.1:16022/api/v1/predict',
          description: [
            { type: 'text', content: '疾病严重性预测 API' },
            { type: 'code', content:
`async handlePredict() {
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
    // 处理其他逻辑
  }
}
` 
            },
            { type: 'text', content: '具体使用方式需要根据您的代码做出相应的更改' }
          ],
          expanded: false
        },
        {
          version: 'V2',
          method: 'POST',
          name: 'API 3',
          description: [
            { type: 'text', content: 'This API is used to submit form data.' },
            { type: 'code', content: `// Here Is The Code
`
            },
            { type: 'text', content: '具体使用方式需要根据您的代码做出相应的更改' }
          ],
          expanded: false
        },
        {
          version: 'V1',
          method: 'GET',
          name: 'API 4',
          description: [
            { type: 'text', content: 'This API is used to submit form data.' },
            { type: 'code', content: `// Here Is The Code
`
            },
            { type: 'text', content: '具体使用方式需要根据您的代码做出相应的更改' }
          ],
          expanded: false
        },
        {
          version: 'V1',
          method: 'GET',
          name: 'API 5',
          description: [
            { type: 'text', content: 'This API is used to submit form data.' },
            { type: 'code', content: `// Here Is The Code1` },
            { type: 'text', content: '具体使用方式需要根据您的代码做出相应的更改' },
            { type: 'code', content: `// Here Is The Code2` },
          ],
          expanded: false
        }
      ],
      toast: {
        visible: false,
        message: '',
        duration: 2000 // 默认显示时间
      },
      checkIcon: require('@/assets/icons/check-icon.png'),
      rapidClickCount: 0,
      rapidClickTimeout: null
    };
  },
  methods: {
    toggleDetails(index) {
      this.apis[index].expanded = !this.apis[index].expanded;
    },
    copyCode(code) {
      clearTimeout(this.rapidClickTimeout);
      this.rapidClickCount++;
      this.rapidClickTimeout = setTimeout(() => {
        this.rapidClickCount = 0;
      }, 500); // 设置500ms的防抖时间

      if (this.rapidClickCount > 3) {
        this.showToast('点击速度过快，请稍后再试', 2000);
        return;
      }

      navigator.clipboard.writeText(code)
        .then(() => {
          this.showToast('复制成功', 2000);
        })
        .catch(err => {
          this.showToast('复制失败，请检查您的浏览器设置', 3000);
          console.error('复制失败:', err);
        });
    },
    showToast(message, duration = 2000) {
      // 如果提示框正在显示，先隐藏它
      if (this.toast.visible) {
        clearTimeout(this.toast.timeoutId); // 清除之前的定时器
        this.toast.visible = false;
        setTimeout(() => {
          this.showToast(message, duration); // 再次调用
        }, 500); // 等待当前提示框隐藏
        return;
      }

      this.toast.message = message;
      this.toast.duration = duration;
      this.toast.visible = true;

      // 设置新的定时器
      this.toast.timeoutId = setTimeout(() => {
        this.toast.visible = false;
      }, duration);
    }
  }
};
</script>

<style scoped>
.api-docs {
  margin: 20px auto 0 !important;
  padding: 20px;
  position: relative;
  z-index: 10;
}

.api-container {
  border: 1px solid #e0e0e0;
  margin-bottom: 10px;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  background-color: white;
}

.api-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 20px;
  background-color: white;
  cursor: pointer;
  border-bottom: 1px solid #e0e0e0;
}

.api-version {
  padding: 5px 10px;
  border-radius: 5px;
  color: white;
  font-family: Georgia, 'Times New Roman', Times, serif;
  font-size: 12px;
}

.v1 {
  background-color: #007bff;
}

.v2 {
  background-color: #dc3545;
}

.api-spacer {
  flex-grow: 0;
  width: 10px;
}

.api-method {
  display: inline-block;
  width: 60px;
  text-align: center;
  padding: 5px 0;
  border-radius: 5px;
  color: white;
  font-family: Georgia, 'Times New Roman', Times, serif;
  font-size: 12px;
}

.get {
  background-color: #28a745;
}

.post {
  background-color: #fd7e14;
}

.api-name {
  flex-grow: 1;
  margin: 0 10px;
  font-weight: bold;
  font-family:'Times New Roman', Times, serif;
  text-align: left;
}

.toggle-button {
  background: none;
  border: none;
  font-size: 16px;
  cursor: pointer;
  color: #007bff;
  transition: color 0.3s ease;
}

.toggle-button i {
  transition: transform 0.3s ease;
}

.toggle-button:hover {
  color: #0056b3;
}

.api-details {
  box-sizing: border-box;
  padding: 20px;
  padding-top: 5px;
  /* padding-bottom: 10px; */
  background-color: white;
  font-family: 'Times New Roman', Times, serif;  
}

.api-details p{
  text-align: left;
  margin: 10px;
}


.code-block {
  background-color: #f5f5f5;
  padding: 10px;
  border: 1px solid #ddd;
  border-radius: 5px;
  overflow-x: auto;
  position: relative;
  text-align: left;
  font-family: 'Courier New', Courier, monospace;
  font-size: 14px;
  line-height: 1.4;
  white-space: pre-wrap;
}

.copy-button {
  position: absolute;
  top: 10px;
  right: 10px;
  background-color: #007bff;
  color: white;
  border: none;
  border-radius: 5px;
  padding: 5px 10px;
  cursor: pointer;
  font-size: 14px;
}

.copy-button:hover {
  background-color: #0056b3;
}

.copy-message {
  position: fixed;
  bottom: 20px;
  left: 50%;
  transform: translateX(-50%);
  background-color: rgba(255, 255, 255, 0.9);
  border: 1px solid #ddd;
  border-radius: 10px;
  padding: 15px 25px;
  font-size: 16px;
  z-index: 1000;
  display: flex;
  align-items: center;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
  opacity: 0;
  transition: opacity 0.5s, transform 0.5s;
  pointer-events: none; /* 防止点击事件穿透 */
}

.copy-message.visible {
  opacity: 1;
  transform: translateX(-50%) translateY(0);
  pointer-events: auto;
}

.check-icon {
  width: 24px;
  height: 24px;
  margin-right: 15px;
}

@keyframes fadeInOut {
  0%, 100% {
    opacity: 0;
    transform: translateY(20px);
  }
  50% {
    opacity: 1;
    transform: translateY(0);
  }
}
</style>