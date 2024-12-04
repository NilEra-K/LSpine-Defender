<template>
    <div class="header">
        <nav>
            <ul>
                <li><router-link to="/">Home</router-link></li>
                <li><router-link to="/about">About</router-link></li>
            </ul>
        </nav>
    </div>
</template>

<script>
export default {
    name: 'Header'
}
</script>

<style scoped>
.header {
    background-color: #333;
    color: white;
    padding: 10px;
    text-align: center;
}

ul {
    list-style: none;
    padding: 0;
}

li {
    display: inline;
    margin-right: 10px;
}
</style>