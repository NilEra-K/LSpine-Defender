<template>
    <div class="header" :class="{ 'sidebar-collapsed': isSidebarCollapsed }">
        <div class="header-content">
            <div class="page-title">
                <h1 style="font-family: 'Times New Roman', Times, serif;">LSPINE-DEFENDER 腰椎卫士——腰椎退行性病变分类系统</h1>
            </div>
            <div class="header-actions">
                <div class="search-box">
                    <i class="fas fa-search"></i>
                    <input type="text" placeholder="搜索...">
                </div>
                <div class="user-profile">
                    <i class="fas fa-user-circle"></i>
                </div>
            </div>
        </div>
    </div>
</template>

<script>
export default {
    name: 'HeaderView',
    computed: {
        isSidebarCollapsed() {
            return this.$parent.isSidebarCollapsed;
        }
    }
}
</script>

<style scoped>
.header {
    background-color: #fff;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    padding: 0 2rem;
    position: fixed;
    top: 0;
    right: 0;
    left: 220px;
    height: 60px;
    z-index: 900;
    transition: all 0.3s ease;
}

.header.sidebar-collapsed {
    left: 80px;
}

.header-content {
    max-width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.page-title h1 {
    font-size: 1.25rem;
    color: #1a1a1a;
    margin: 0;
    font-weight: 600;
    white-space: nowrap;
}

.header-actions {
    display: flex;
    align-items: center;
    gap: 1.5rem;
    margin-left: auto;
}

.search-box {
    display: flex;
    align-items: center;
    background: #f1f5f9;
    padding: 0.5rem 1rem;
    border-radius: 8px;
    min-width: 200px;
}

.search-box input {
    border: none;
    background: none;
    outline: none;
    padding-left: 0.5rem;
    width: 100%;
    color: #666;
}

.search-box i {
    color: #666;
    flex-shrink: 0;
}

.user-profile {
    cursor: pointer;
    margin-left: 1rem;
    flex-shrink: 0;
}

.user-profile i {
    font-size: 1.5rem;
    color: #666;
}

@media (max-width: 768px) {
    .header {
        left: 64px;
        padding: 0 1.5rem;
    }
    
    .page-title h1 {
        font-size: 1rem;
        margin-left: 0.5rem;
    }
    
    .search-box {
        display: none;
    }
}
</style>