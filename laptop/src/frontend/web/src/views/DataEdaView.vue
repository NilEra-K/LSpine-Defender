<template>
    <div class="eda-container">
        <!-- 标题和介绍部分 -->
        <div class="eda-header">
            <h2 style="font-size: large; font-family: 'Times New Roman', Times, serif;">EDA</h2>
            <p class="description">
                展示EDA结果和分析思路
            </p>
        </div>
        <div class="chart-selector">
            <select v-model="selectedChart" @change="handleSelectChange" class="custom-select">
                <option v-for="(chart, index) in charts" :key="index" :value="index">
                    {{ chart.title }}
                </option>
            </select>
        </div>
        <div v-if="selectedChart !== null" class="chart-container">
            <div class="chart-item">
                <div ref="chartRef" class="chart"></div>
            </div>
        </div>
    </div>
</template>

<script>
import * as echarts from 'echarts';

export default {
    name: 'DataEdaView',
    data() {
        return {
            selectedChart: 0,
            charts: [
                {
                    title: 'EDA 结果 - 折线图',
                    option: {
                        title: {
                            text: '折线图'
                        },
                        tooltip: {},
                        xAxis: {
                            data: ["一月", "二月", "三月", "四月", "五月", "六月", "七月"]
                        },
                        yAxis: {},
                        series: [{
                            name: '销量',
                            type: 'line',
                            data: [120, 200, 150, 80, 70, 110, 130]
                        }]
                    }
                },
                {
                    title: 'EDA 结果 - 柱状图',
                    option: {
                        title: {
                            text: '柱状图'
                        },
                        tooltip: {},
                        xAxis: {
                            data: ["一月", "二月", "三月", "四月", "五月", "六月", "七月"]
                        },
                        yAxis: {},
                        series: [{
                            name: '销量',
                            type: 'bar',
                            data: [120, 200, 150, 80, 70, 110, 130]
                        }]
                    }
                },
                {
                    title: 'EDA 结果 - 饼图',
                    option: {
                        title: {
                            text: '饼图'
                        },
                        tooltip: {},
                        series: [{
                            name: '访问来源',
                            type: 'pie',
                            radius: '50%',
                            data: [
                                { value: 1048, name: '搜索引擎超链接' },
                                { value: 735, name: '直接访问' },
                                { value: 580, name: '邮件营销' },
                                { value: 484, name: '联盟广告' },
                                { value: 300, name: '视频广告' }
                            ]
                        }]
                    }
                }
            ]
        };
    },
    mounted() {
        this.initChart();
    },
    methods: {
        initChart() {
            const chartRef = this.$refs.chartRef;
            if (chartRef) {
                const myChart = echarts.init(chartRef);
                myChart.setOption(this.charts[this.selectedChart].option);
            }
        },
        handleSelectChange() {
            this.initChart();
        }
    }
};
</script>

<style scoped>
.eda-container {
    margin: 20px auto 0 !important;
    position: relative;
    z-index: 10;
    padding: 1rem 2rem;
    /* padding: 20px; */
    background-color: #f8fafc;
    height: auto;

}

.section-header {
    margin-bottom: 20px;
}

.section-header h2 {
    font-size: large;
    font-family: 'Times New Roman', Times, serif;
}

.section-header p {
    font-size: 14px;
    color: #666;
}

.eda-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
}

.chart-selector {
    margin-bottom: 20px;
}

.custom-select {
    padding: 10px;
    background-color: #ffffff;
    border: 1px solid #ddd;
    border-radius: 5px;
    font-family: 'Arial', sans-serif;
    font-size: 14px;
    width: 100%;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    appearance: none;
    -webkit-appearance: none;
    -moz-appearance: none;
    background-image: url('data:image/svg+xml;utf8,<svg fill="#000000" height="30" viewBox="0 0 24 24" width="30" xmlns="http://www.w3.org/2000/svg"><path d="M7 10l5 5 5-5z"/><path d="M0 0h24v24H0z" fill="none"/></svg>');
    background-repeat: no-repeat;
    background-position: right 10px top 50%;
    cursor: pointer;
}

.chart-container {
    padding: 10px;
    background-color: #ffffff;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.chart-item {
    width: 100%;
}

.chart {
    height: 400px;
}
</style>