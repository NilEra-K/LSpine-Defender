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
import { markRaw } from 'vue';
import * as echarts from 'echarts';

export default {
    name: 'DataEdaView',
    data() {
        return {
            selectedChart: 0,
            myChart: null, // 添加一个属性来存储 ECharts 实例
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
                    title: 'EDA 结果 - 热力图 - 不同节段不同疾病的相关性矩阵',
                    type: 'heatmap'
                },
                {
                    title: 'EDA 结果 - 柱状图 - 关节下狭窄在不同节段各程度的患病数',
                    type: 'bar',
                    url: 'http://127.0.0.1:16020/api/eda/subarticular_stenosis_counts'
                },
                {
                    title: 'EDA 结果 - 柱状图 - 左侧关节下狭窄在不同节段各程度的患病数',
                    type: 'bar',
                    url: 'http://127.0.0.1:16020/api/eda/left_subarticular_stenosis_counts'
                },
                {
                    title: 'EDA 结果 - 柱状图 - 右侧关节下狭窄在不同节段各程度的患病数',
                    type: 'bar',
                    url: 'http://127.0.0.1:16020/api/eda/right_subarticular_stenosis_counts'
                },
            ]
        };
    },
    mounted() {
        this.initChart();
    },
    methods: {
        async fetchHeatmapData() {
            try {
                const response = await fetch('http://127.0.0.1:16020/api/eda/heatmap');
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                const data = await response.json();
                return data;
            } catch (error) {
                console.error('Error fetching heatmap data:', error);
                return null;
            }
        },
        async fetchBarData(url) {
            try {
                const response = await fetch(url);
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                const data = await response.json();
                console.log(this.processBarData(data));
                return this.processBarData(data);
            } catch (error) {
                console.error('Error fetching chart data:', error);
                return null;
            }
        },
        processBarData(data) {
            const levels = [];
            const normalCounts = [];
            const moderateCounts = [];
            const severeCounts = [];

            data.forEach(item => {
                const level = item.level;
                if (!levels.includes(level)) {
                    levels.push(level);
                }
                switch (item.subarticular_stenosis) {
                    case 'Normal/Mild':
                        normalCounts.push(item.count);
                        break;
                    case 'Moderate':
                        moderateCounts.push(item.count);
                        break;
                    case 'Severe':
                        severeCounts.push(item.count);
                        break;
                }
            });

            return { levels, normalCounts, moderateCounts, severeCounts };
        },
        async initChart() {
            const chartRef = this.$refs.chartRef;
            if (chartRef) {
                if (this.myChart) {
                    this.myChart.dispose(); // 销毁当前的 ECharts 实例
                }
                this.myChart = echarts.init(chartRef); // 初始化一个新的 ECharts 实例

                const selectedChart = this.charts[this.selectedChart];
                console.log(selectedChart.type);
                if (selectedChart.type === 'heatmap') {
                    const heatmapData = await this.fetchHeatmapData();
                    if (heatmapData) {
                        const option = {
                            title: {
                                text: heatmapData.title,
                                textStyle: {
                                    fontSize: 20
                                },
                                left: 'center'
                            },
                            tooltip: {
                                position: 'inside',
                                formatter: '{c}'
                            },
                            xAxis: {
                                type: 'category',
                                data: heatmapData.xAxis,
                                axisLabel: {
                                    rotate: -20,
                                    fontSize: 8
                                }
                            },
                            yAxis: {
                                type: 'category',
                                data: heatmapData.yAxis,
                                axisLabel: {
                                    fontSize: 8
                                }
                            },
                            visualMap: {
                                min: -0.08708446110266807,
                                max: 1,
                                orient: 'vertical',
                                left: 'right',
                                top: 'middle',
                                text: ['高', '低'],
                                inRange: {
                                    color: ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026']
                                },
                                is_piecewise: false
                            },
                            series: [
                                {
                                    name: '相关性',
                                    type: 'heatmap',
                                    data: heatmapData.data,
                                    label: {
                                        show: true,
                                    }
                                }
                            ]
                        };
                        this.myChart.setOption(option);
                    }
                } else if(selectedChart.type === 'bar') {
                    const url = selectedChart.url;
                    const chartData = markRaw(await this.fetchBarData(url));

                    if (chartData) {
                        // const levels = chartData.map(item => item.level);
                        // const normalCounts = chartData.map(item => item.count[0]);
                        // const moderateCounts = chartData.map(item => item.count[1]);
                        // const severeCounts = chartData.map(item => item.count[2]);
                        console.log(chartData);
                        const option = {
                            title: {
                                text: selectedChart.title.split(' - ')[2],
                                textStyle: {
                                    fontSize: 20
                                },
                                left: 'center'
                            },
                            tooltip: {
                                trigger: 'axis',
                                axisPointer: {
                                    type: 'shadow'
                                }
                            },
                            legend: {
                                data: ['Normal/Mild', 'Moderate', 'Severe'],
                                top: '2.5%'
                            },
                            xAxis: {
                                type: 'category',
                                data: chartData.levels,
                                axisLabel: {
                                    rotate: 0,
                                    fontSize: 15
                                }
                            },
                            yAxis: {
                                type: 'value',
                                name: 'Count',
                                axisLabel: {
                                    fontSize: 15
                                }
                            },
                            series: [
                                {
                                    name: 'Normal/Mild',
                                    type: 'bar',
                                    data: chartData.normalCounts
                                },
                                {
                                    name: 'Moderate',
                                    type: 'bar',
                                    data: chartData.moderateCounts
                                },
                                {
                                    name: 'Severe',
                                    type: 'bar',
                                    data: chartData.severeCounts
                                }
                            ]
                        };
                        this.myChart.setOption(option);
                    }
                } else {
                    this.myChart.setOption(selectedChart.option);
                }
            }
        },
        handleSelectChange() {
            this.initChart();
        }
    },
    beforeUnmount() {
        window.removeEventListener('resize', this.onResize);
        if (this.myChart) {
            this.myChart.dispose(); // 组件销毁前，销毁 ECharts 实例
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
    background-color: #f8fafc;
    height: auto;
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
    height: 800px;
}
</style>