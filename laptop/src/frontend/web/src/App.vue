<template>
  <div id="app">
    <side-bar @toggle="handleSidebarToggle" />
    <div class="main-wrapper" :class="{ 'sidebar-collapsed': isSidebarCollapsed }">
      <header-view />
      <div class="main-content">
        <router-view />
      </div>
    </div>
  </div>
</template>

<script>
import HeaderView from '@/components/HeaderView.vue'
import SideBar from '@/components/SideBar.vue'

export default {
  name: 'App',
  components: {
    SideBar,
    HeaderView
  },
  data() {
    return {
      isSidebarCollapsed: false
    }
  },
  methods: {
    handleSidebarToggle(collapsed) {
      this.isSidebarCollapsed = collapsed;
    }
  }
}
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
}

.container {
  display: flex;
}

body {
    margin: 0;
    padding: 0;
    background: #f8fafc;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
}

.main-wrapper {
    margin-left: 200px; /* 对应新的sidebar宽度 */
    transition: all 0.3s ease;
}

.main-wrapper.sidebar-collapsed {
    margin-left: 60px; /* 对应新的折叠宽度 */
}

.main-content {
    padding: 2rem;
    margin-top: 5px;
}

@media (max-width: 768px) {
    .main-wrapper {
        margin-left: 64px;
    }
    
    /* 当侧边栏展开时，隐藏主内容 */
    .sidebar:not(.collapsed) + .main-wrapper {
        display: none;
    }
}

</style>