<template>
  <div class="home-view">
    <div class="welcome-section">
      <h1>欢迎使用脊柱姿态评估系统</h1>
      <p class="subtitle">基于计算机视觉的智能评估解决方案</p>
    </div>
    <div class="features-grid">
      <div class="feature-card">
        <i class="fas fa-camera"></i>
        <h3>姿态检测</h3>
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
    name: 'HomeView'
}
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
}
</style>