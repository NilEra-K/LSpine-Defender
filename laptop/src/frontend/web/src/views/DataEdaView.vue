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
import { isRef, markRaw } from 'vue';
import * as echarts from 'echarts';

export default {
    name: 'DataEdaView',
    data() {
        return {
            selectedChart: 0,
            myChart: null, // 添加一个属性来存储 ECharts 实例
            charts: [
                {
                    title: 'EDA 结果 - 散点图 - MRI影像患病坐标总览',
                    type: 'scatter',
                    url: 'http://127.0.0.1:16020/api/eda/mri_image_coordinates_overview'
                },
                {
                    title: 'EDA 结果 - 散点图 - 疾病在MRI影像中的患病坐标',
                    type: 'condition_scatter',
                    url: 'http://127.0.0.1:16020/api/eda/mri_image_coordinates_condition'
                },
                {
                    title: 'EDA 结果 - 散点图 - 不同节段和患病坐标的关系',
                    type: 'scatter',
                    url: 'http://127.0.0.1:16020/api/eda/mri_image_coordinates_level'
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
                {
                    title: 'EDA 结果 - 柱状图 - 神经孔狭窄在不同节段各程度的患病数',
                    type: 'bar',
                    url: 'http://127.0.0.1:16020/api/eda/neural_foraminal_narrowing_counts'
                },
                {
                    title: 'EDA 结果 - 柱状图 - 左侧神经孔狭窄在不同节段各程度的患病数',
                    type: 'bar',
                    url: 'http://127.0.0.1:16020/api/eda/left_neural_foraminal_narrowing_counts'
                },
                {
                    title: 'EDA 结果 - 柱状图 - 右侧神经孔狭窄在不同节段各程度的患病数',
                    type: 'bar',
                    url: 'http://127.0.0.1:16020/api/eda/right_neural_foraminal_narrowing_counts'
                },
                {
                    title: 'EDA 结果 - 柱状图 - 椎管狭窄狭窄在不同节段各程度的患病数',
                    type: 'bar',
                    url: 'http://127.0.0.1:16020/api/eda/spinal_canal_stenosis_counts'
                },
                {
                    title: 'EDA 结果 - 柱状图 - 每个 Study 中 Series 计数的频率分布',
                    type: 'bar_2',
                    url: 'http://127.0.0.1:16020/api/eda/series_counts'
                },
            ]
        };
    },
    mounted() {
        this.initChart();
    },
    methods: {
        async fetchscatterData(url) {
            try {
                const response = await fetch(url);
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                const data = await response.json();
                return data;
            } catch (error) {
                console.error('Error fetching chart data:', error);
                return null;
            }
        },
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
        async fetchBarData_2(url) {
            try {
                const response = await fetch(url);
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                const data = await response.json();
                return data;
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
                // 关节下狭窄
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
                // 左侧关节下狭窄
                switch (item.left_subarticular_stenosis) {
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
                // 右侧关节下狭窄
                switch (item.right_subarticular_stenosis) {
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
                // 神经孔狭窄
                switch (item.neural_foraminal_narrowing) {
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
                // 左侧神经孔狭窄
                switch (item.left_neural_foraminal_narrowing) {
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
                // 右侧神经孔狭窄
                switch (item.right_neural_foraminal_narrowing) {
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
                // 椎管狭窄
                switch (item.spinal_canal_stenosis) {
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
                this.myChart = markRaw(echarts.init(chartRef)); // 初始化一个新的 ECharts 实例

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
                } else if (selectedChart.type === 'bar') {
                    const url = selectedChart.url;
                    const chartData = markRaw(await this.fetchBarData(url));
                    console.log('是否为REF', isRef(chartData));

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
                } else if (selectedChart.type === 'bar_2') {
                    const url = selectedChart.url;
                    const chartData = markRaw(await this.fetchBarData_2(url));
                    // const xAxis = chartData.xAxis;
                    // const yAxis = chartData.yAxis;
                    // console.log('是否为REF', isRef(chartData));
                    console.log("bar_2, chartData: ", chartData.xAxis);
                    if (chartData) {
                        // const levels = chartData.map(item => item.level);
                        // const normalCounts = chartData.map(item => item.count[0]);
                        // const moderateCounts = chartData.map(item => item.count[1]);
                        // const severeCounts = chartData.map(item => item.count[2]);
                        console.log(chartData);
                        const option = {
                            title: {
                                text: '每个 Study 中 Series 计数的频率分布',
                                subtext: '',
                                left: 'center',
                                top: '1%'
                            },
                            tooltip: {
                                trigger: 'axis',
                                axisPointer: {
                                    type: 'shadow'
                                }
                            },
                            xAxis: {
                                type: 'category',
                                data: chartData.xAxis,
                                axisLabel: {
                                    rotate: 0
                                }
                            },
                            yAxis: {
                                type: 'value',
                                name: '频率'
                            },
                            series: [
                                {
                                    name: '频率',
                                    type: 'bar',
                                    data: chartData.yAxis,
                                    label: {
                                        show: true,
                                        position: 'top'
                                    },
                                    markLine: {
                                        data: [
                                            {
                                                type: 'average',
                                                name: 'Average'
                                            }
                                        ]
                                    },
                                    itemStyle: {
                                        color: '#4e97a7'
                                    }
                                }
                            ],
                            visualMap: {
                                // max: Math.max(...this.chartData.yAxis),
                                max: Math.max(...chartData.yAxis),
                                isShow: false
                            },
                            toolbox: {
                                feature: {
                                    saveAsImage: {
                                        type: 'jpeg'
                                    },
                                    dataView: {},
                                    magicType: {
                                        type: ['line', 'bar']
                                    },
                                    restore: {}
                                },
                                top: '1%',
                                right: '5%'
                            },
                            legend: {
                                top: '5%'
                            }
                        };
                        this.myChart.setOption(option);
                    }
                } else if (selectedChart.type === 'scatter') {
                    const url = selectedChart.url;
                    const chartData = markRaw(await this.fetchscatterData(url));
                    if (chartData) {
                        const option = {
                            title: {
                                text: selectedChart.title.split(' - ')[2],
                                left: 'center',
                                top: '1%',
                                textStyle: {
                                    fontSize: 22
                                }
                            },
                            tooltip: {
                                trigger: 'item',
                                formatter: function (params) {
                                    return `X: ${params.value[0]}, Y: ${params.value[1]}`;
                                }
                            },
                            xAxis: {
                                type: 'value',
                                max: Math.max(...chartData.xAxis)
                            },
                            yAxis: {
                                type: 'value',
                                max: Math.max(...chartData.yAxis)
                            },
                            series: [
                                {
                                    name: '',
                                    type: 'scatter', // 散点图类型
                                    symbolSize: 8,
                                    color: 'black',
                                    data: chartData.xAxis.map((x, index) => [x, chartData.yAxis[index]]), // 将 x 和 y 数据组合成数组
                                    itemStyle: {
                                        opacity: 0.4,
                                        borderColor: 'white',
                                        borderWidth: 1
                                    },
                                    label: {
                                        show: false
                                    }
                                }
                            ],
                            toolbox: {
                                feature: {
                                    saveAsImage: {},
                                    dataView: {},
                                    magicType: {
                                        type: ['line', 'bar']
                                    },
                                    restore: {}
                                },
                                top: '0%',
                                right: '5%'
                            }
                        };
                        this.myChart.setOption(option);
                    }
                } else if (selectedChart.type === 'condition_scatter') {
                    const url = selectedChart.url;
                    const chartData = markRaw(await this.fetchscatterData(url));
                    console.log(chartData);
                    const conditionColors = {
                        'Spinal Canal Stenosis': '#4600c0',
                        'Right Neural Foraminal Narrowing': '#00a419',
                        'Left Neural Foraminal Narrowing': '#111111',
                        'Left Subarticular Stenosis': '#ffff00',
                        'Right Subarticular Stenosis': '#ff0000'
                    };
                    if (this.myChart) {
                        this.myChart.dispose();
                    }
                    this.myChart = echarts.init(this.$refs.chartRef);

                    if (chartData) {
                        const series = chartData.map(condition => ({
                            name: condition.name,
                            type: 'scatter',
                            symbolSize: 8,
                            data: condition.data,
                            itemStyle: {
                                color: conditionColors[condition.name],
                                opacity: 0.7,
                                borderColor: 'white',
                                borderWidth: 1
                            },
                            label: {
                                show: false
                            }
                        }));

                        const option = {
                            title: {
                                text: '疾病在MRI影像中的患病坐标',
                                left: 'center',
                                top: '1%',
                                textStyle: {
                                    fontSize: 22
                                }
                            },
                            tooltip: {
                                trigger: 'item',
                                formatter: function (params) {
                                    return `X: ${params.value[0]}, Y: ${params.value[1]}`;
                                }
                            },
                            xAxis: {
                                type: 'value',
                                max: chartData.reduce((max, condition) => Math.max(max, ...condition.data.map(d => d[0])), 0)
                            },
                            yAxis: {
                                type: 'value',
                                max: chartData.reduce((max, condition) => Math.max(max, ...condition.data.map(d => d[1])), 0)
                            },
                            series: series,
                            legend: {
                                data: chartData.map(condition => condition.name),
                                left: '10%',
                                top: '7.65%',
                                orient: 'vertical',
                                itemHeight: 10
                            },
                            toolbox: {
                                feature: {
                                    saveAsImage: {},
                                    dataView: {},
                                    magicType: {
                                        type: ['line', 'bar']
                                    },
                                    restore: {}
                                },
                                top: '1%',
                                right: '5%'
                            }
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