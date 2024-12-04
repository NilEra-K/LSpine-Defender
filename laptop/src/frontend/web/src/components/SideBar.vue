<template>
    <div class="sidebar" :class="{ 'collapsed': isCollapsed }">
        <div class="toggle-btn" @click="toggleSidebar">
            <i :class="isCollapsed ? 'fas fa-angle-right' : 'fas fa-angle-left'"></i>
        </div>
        <div class="logo-container">
            <h2 class="logo">LSpine</h2>
        </div>
        <nav class="nav-menu">
            <!-- 为每个导航项添加点击事件 -->
            <router-link to="/" class="nav-item" active-class="active" @click="handleNavClick">
                <i class="fas fa-home"></i>
                <span>首页</span>
            </router-link>
            <router-link to="/feature-one" class="nav-item" active-class="active" @click="handleNavClick">
                <i class="fas fa-chart-line"></i>
                <span>姿态检测</span>
            </router-link>
            <router-link to="/feature-two" class="nav-item" active-class="active" @click="handleNavClick">
                <i class="fas fa-chart-bar"></i>
                <span>数据分析</span>
            </router-link>
            <router-link to="/feature-three" class="nav-item" active-class="active" @click="handleNavClick">
                <i class="fas fa-file-alt"></i>
                <span>报告生成</span>
            </router-link>
        </nav>
        <div class="toggle-btn" @click="toggleSidebar">
            <i :class="isCollapsed ? 'fas fa-angle-right' : 'fas fa-angle-left'"></i>
        </div>
    </div>
</template>

<script>
export default {
    name: 'SideBar',
    data() {
        return {
            isCollapsed: false,
            windowWidth: window.innerWidth
        }
    },
    mounted() {
        window.addEventListener('resize', this.handleResize);
        this.handleResize(); // 初始化时检查
    },
    beforeUnmount() {
        window.removeEventListener('resize', this.handleResize);
    },
    methods: {
        toggleSidebar() {
            this.isCollapsed = !this.isCollapsed;
            this.$emit('toggle', this.isCollapsed);
        },
        handleResize() {
            this.windowWidth = window.innerWidth;
            if (this.windowWidth <= 768) {
                this.isCollapsed = true;
                this.$emit('toggle', true);
            }
        },
        handleNavClick() {
            if (this.windowWidth <= 768) {
                this.isCollapsed = true;
                this.$emit('toggle', true);
            }
        }
    }
}
</script>

<style scoped>
.sidebar {
    width: 200px;
    /* 减小默认宽度 */
    height: 100vh;
    background: #FFFFFF;
    position: fixed;
    left: 0;
    top: 0;
    box-shadow: 0 0 20px rgba(0, 0, 0, 0.05);
    padding: 2rem 1rem;
    transition: all 0.3s ease;
    z-index: 1000;
    display: flex;
    flex-direction: column;
}

.sidebar.collapsed {
    width: 60px;
    /* 减小折叠时的宽度 */
}

.toggle-btn {
    position: absolute;
    right: -16px;
    top: 80%;
    transform: translateY(-50%);
    width: 32px;
    height: 32px;
    background: #fff;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    transition: all 0.3s ease;
    z-index: 1001;
}

.toggle-btn:hover {
    background: #f8f9fa;
}

.toggle-btn i {
    font-size: 1rem;
    color: #666;
    transition: transform 0.3s ease;
}

/* .sidebar.collapsed .toggle-btn i {
    transform: rotate(180deg);
} */

.logo-container {
    padding: 1rem 0 2rem 0;
    text-align: center;
    overflow: hidden;
}

.logo {
    font-size: 28px;
    font-weight: 700;
    color: #1a1a1a;
    letter-spacing: 1px;
    margin: 0;
    white-space: nowrap;
}

.nav-menu {
    flex: 1;
    /* 让导航菜单占据剩余空间 */
    margin-bottom: 60px;
    /* 为底部按钮留出空间 */
}

.nav-item {
    display: flex;
    align-items: center;
    padding: 1rem 1.5rem;
    color: #666666;
    text-decoration: none;
    border-radius: 12px;
    margin-bottom: 0.5rem;
    transition: all 0.3s ease;
    font-weight: 500;
    white-space: nowrap;
}

.nav-item:hover {
    background: #f8f9fa;
    color: #000000;
}

.nav-item.active {
    background: #f1f5f9;
    color: #2563eb;
    font-weight: 600;
}

.nav-item i {
    margin-right: 1rem;
    font-size: 1.2rem;
    opacity: 0.8;
}

.sidebar.collapsed .nav-item span {
    display: none;
}

.sidebar.collapsed .logo {
    font-size: 20px;
}

.sidebar.collapsed .nav-item {
    padding: 1rem;
    justify-content: center;
}

.sidebar.collapsed .nav-item i {
    margin: 0;
}

@media (max-width: 768px) {
    .sidebar {
        width: 64px;
    }

    .sidebar.collapsed {
        width: 64px;
    }

    /* 移动端展开时的样式 */
    .sidebar:not(.collapsed) {
        width: 100vw;
        padding: 2rem;
    }

    .nav-item span {
        display: none;
    }

    .sidebar:not(.collapsed) .nav-item span {
        display: inline;
        /* 展开时显示文字 */
    }

    .logo {
        font-size: 20px;
    }

    /* 移除隐藏按钮的样式 */
    .toggle-btn {
        display: block;
    }

    .toggle-btn {
        right: -12px;
        width: 24px;
        height: 24px;
    }

    .toggle-btn i {
        font-size: 0.9rem;
    }
}
</style>