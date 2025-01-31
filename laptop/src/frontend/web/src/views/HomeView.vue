<template>
  <div class="home-view">
    <div class="welcome-section">
      <h1>欢迎使用脊柱姿态评估系统</h1>
      <p class="subtitle">基于计算机视觉的智能评估解决方案</p>
    </div>
    
    <!-- 新增的轮播图和介绍区域 -->
    <div class="carousel-intro">
      <div class="carousel" @mouseenter="showControls = true" @mouseleave="showControls = false">
        <div class="carousel-inner">
          <div class="carousel-item" v-for="(image, index) in images" :key="index" :class="{ active: index === currentIndex }">
            <img :src="image" alt="Lumbur Image" class="carousel-image">
          </div>
        </div>
        <a class="carousel-control-prev" v-show="showControls" @click="prevSlide">
          <img class="carousel-control-prev-icon" src="@/assets/icons/left.png" />
        </a>
        <a class="carousel-control-next" v-show="showControls" @click="nextSlide">
          <img class="carousel-control-next-icon" src="@/assets/icons/right.png" />
        </a>
      </div>
      <div class="intro-card">
        <h3>系统介绍</h3>
        <p>LSpine-Defender，全称 Lumbar-Spine Defender，脊柱卫士。</p>
        <p>这是一款腰椎退行性病变分类系统，本系统基于多种算法和模型进行开发，包括 ResNet、DenseNet、CenterNet、EfficientNet等一系列模型。</p>
        <p>本系统致力于使用前沿技术，做技术创新，对现有模型进行了改进。</p>
        <p>上传腰椎图像，系统将自动分析并生成相关参数指标，帮助医生进行更准确的诊断。</p>
      </div>
    </div>

    <div class="features-grid">
      <div class="feature-card">
        <i class="fas fa-camera"></i>
        <h3>成像方式预测</h3>
        <p>实时捕捉和分析脊柱姿态</p>
      </div>
      <div class="feature-card">
        <i class="fas fa-chart-line"></i>
        <h3>数据分析</h3>
        <p>深入分析评估结果</p>
      </div>
      <div class="feature-card">
        <i class="fas fa-file-medical"></i>
        <h3>报告生成</h3>
        <p>生成专业评估报告</p>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'HomeView',
  data() {
    return {
      images: [
        require('@/assets/images/Sagittal_T1_00007.png'),
        require('@/assets/images/Sagittal_T1_00007_keypoint.png'),
        require('@/assets/images/Sagittal_T2_STIR_00007.png'),
        require('@/assets/images/Sagittal_T2_STIR_00007_keypoint.png'),
        require('@/assets/images/Axial_T2_00000.png'),
        require('@/assets/images/Axial_T2_00000_keypoint.png'),
      ],
      currentIndex: 0,
      intervalId: null, // 添加定时器ID
      showControls: false, // 控制按钮显示的布尔值
    };
  },
  methods: {
    nextSlide() {
      this.currentIndex = (this.currentIndex + 1) % this.images.length;
    },
    prevSlide() {
      this.currentIndex = (this.currentIndex - 1 + this.images.length) % this.images.length;
    },
    startAutoPlay() {
      this.intervalId = setInterval(() => {
        this.nextSlide();
      }, 5000); // 设置轮播间隔为5秒
    },
    stopAutoPlay() {
      clearInterval(this.intervalId);
    }
  },
  mounted() {
    this.startAutoPlay(); // 在组件挂载时启动自动播放
  },
  beforeUnmount() {
    this.stopAutoPlay(); // 在组件销毁前停止自动播放
  }
};
</script>

<style scoped>
.home-view {
  padding: 2rem;
}

.welcome-section {
  text-align: left;
  margin-bottom: 3rem;
}

.welcome-section h1 {
  font-size: 2.5rem;
  color: #2c3e50;
  margin-bottom: 1rem;
}

.subtitle {
  font-size: 1.2rem;
  color: #666;
}

/* 新增的轮播图和介绍区域样式 */
.carousel-intro {
  display: flex;
  align-items: center;
  gap: 1rem;
  margin-bottom: 2rem;
}

.carousel {
  position: relative;
  width: 300px; /* 设置轮播图的宽度 */
  height: 300px; /* 设置轮播图的高度 */
  min-width: 300px; /* 设置轮播图的最小宽度 */
  min-height: 300px; /* 设置轮播图的最小高度 */
  overflow: hidden; /* 添加此行 */
}

.carousel-inner {
  position: relative;
  width: 100%;
  height: 100%;
}

.carousel-item {
  position: absolute;
  width: 100%;
  height: 100%; /* 确保高度正确 */
  opacity: 0;
  transition: opacity 0.5s ease, transform 0.5s ease;
  z-index: 0;
}

.carousel-item.active {
  opacity: 1;
  transform: translateX(0);
  z-index: 1;
}

.carousel-image {
  width: 100%;
  height: 100%;
  object-fit: cover; /* 添加此行 */
  border-radius: 8px;
}

.carousel-control-prev,
.carousel-control-next {
  position: absolute;
  top: 50%;
  transform: translateY(-50%);
  background-color: transparent; /* 修改背景颜色 */
  color: white;
  padding: 0;
  cursor: pointer;
  z-index: 2; /* 添加此行 */
  font-size: 2rem; /* 增加箭头大小 */
  width: 30px; /* 设置宽度 */
  height: 30px; /* 设置高度 */
  display: flex;
  align-items: center;
  justify-content: center;
}

.carousel-control-prev {
  left: 10px; /* 修改位置 */
}

.carousel-control-next {
  right: 10px; /* 修改位置 */
}

.carousel-control-prev-icon,
.carousel-control-next-icon {
  color: white; /* 设置箭头颜色 */
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: transparent;
}

.intro-card {
  background: white;
  padding: 2rem;
  box-sizing: border-box;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  text-align: center;
  transition: transform 0.3s ease;
  height: 300px; /* 设置介绍卡片的高度与轮播图一致 */
  display: flex;
  flex-direction: column;
  justify-content: center;
}

.intro-card:hover {
  transform: translateY(-5px);
}

.intro-card h3 {
  font-size: 1.2rem;
  color: #1a1a1a;
  margin: 0 0 0.5rem;
  font-weight: 600;
}

.intro-card p {
  font-size: 1rem;
  color: #666;
  margin: 0;
  text-align: left;
}

.features-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 2rem;
  margin-top: 2rem;
}

.feature-card {
  background: white;
  padding: 2rem;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  text-align: center;
  transition: transform 0.3s ease;
}

.feature-card:hover {
  transform: translateY(-5px);
}

.feature-card i {
  font-size: 2.5rem;
  color: #2563eb;
  margin-bottom: 1rem;
}

.feature-card h3 {
  font-size: 1.25rem;
  color: #1a1a1a;
  margin-bottom: 0.5rem;
}

.feature-card p {
  color: #666;
}

@media (max-width: 768px) {
  .welcome-section h1 {
    font-size: 2rem;
  }
  
  .features-grid {
    grid-template-columns: 1fr;
  }

  .carousel-intro {
    flex-direction: column;
    align-items: center;
  }

  .carousel {
    width: 100%; /* 调整轮播图的宽度 */
    height: 300px; /* 设置轮播图的高度 */
    min-width: 300px; /* 设置轮播图的最小宽度 */
    min-height: 300px; /* 设置轮播图的最小高度 */
  }

  .intro-card {
    height: 300px; /* 设置介绍卡片的高度与轮播图一致 */
    text-align: center;
  }
}
</style>