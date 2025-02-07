from flask import Blueprint, render_template, request, redirect, url_for
from flask import Blueprint, request, jsonify
import os
from PIL import Image
import base64
import io
from flask_cors import CORS
import torch
import importlib
import pandas as pd

# 使用 sys.path.append 配合 importlib 导入相关的包
import sys
sys.path.append("workstation/competition/med-img-classify")
mic_predict = importlib.import_module("mic-predict")

# 数据加载
train_data_df = pd.read_csv('E:/RSNA-Dataset/train.csv')
train_label_df = pd.read_csv('E:/RSNA-Dataset/train_label_coordinates.csv')
train_desc_df = pd.read_csv('E:/RSNA-Dataset/train_series_descriptions.csv')

df_train_step_1 = pd.merge(left=train_label_df, right=train_data_df, how='left', on='study_id').reset_index(drop=True)
df_train_step_1.head()

df_train = pd.merge(left=df_train_step_1, right=train_desc_df, how='left', on=['study_id', 'series_id']).reset_index(drop=True)
df_train.head()

# 关节下狭窄数据加载
subarticular_stenosis_columns = [column for column in df_train.columns if 'subarticular_stenosis' in column]
df_train_subarticular_stenosis = []

for column in subarticular_stenosis_columns:
    df = df_train[[column]].copy(deep=True)
    df['level'] = '_'.join(column.split('_')[-2:])
    df = df.rename(columns={column: 'subarticular_stenosis'})
    df_train_subarticular_stenosis.append(df)
    
df_train_subarticular_stenosis = pd.concat(df_train_subarticular_stenosis, axis=0).reset_index(drop=True)

df_train_subarticular_stenosis_counts = df_train_subarticular_stenosis.groupby('level').value_counts().reset_index()
df_train_subarticular_stenosis_counts['severity'] = df_train_subarticular_stenosis_counts['subarticular_stenosis'].map({
    'Normal/Mild': 0,
    'Moderate': 1,
    'Severe': 2
})
df_train_subarticular_stenosis_counts = df_train_subarticular_stenosis_counts.sort_values(by=['level', 'severity'], ascending=True)
df_train_subarticular_stenosis_counts['percentage'] = df_train_subarticular_stenosis_counts['count'] / df_train_subarticular_stenosis_counts.groupby('level')['count'].transform('sum') * 100

left_subarticular_stenosis_columns = [column for column in df_train.columns if column.startswith('left_subarticular_stenosis')]
df_train_left_subarticular_stenosis = []

for column in left_subarticular_stenosis_columns:
    df = df_train[[column]].copy(deep=True)
    df['level'] = '_'.join(column.split('_')[-2:])
    df = df.rename(columns={column: 'left_subarticular_stenosis'})
    df_train_left_subarticular_stenosis.append(df)
    
df_train_left_subarticular_stenosis = pd.concat(df_train_left_subarticular_stenosis, axis=0).reset_index(drop=True)

df_train_left_subarticular_stenosis_counts = df_train_left_subarticular_stenosis.groupby('level').value_counts().reset_index()
df_train_left_subarticular_stenosis_counts['severity'] = df_train_left_subarticular_stenosis_counts['left_subarticular_stenosis'].map({
    'Normal/Mild': 0,
    'Moderate': 1,
    'Severe': 2
})
df_train_left_subarticular_stenosis_counts = df_train_left_subarticular_stenosis_counts.sort_values(by=['level', 'severity'], ascending=True)
df_train_left_subarticular_stenosis_counts['percentage'] = df_train_left_subarticular_stenosis_counts['count'] / df_train_left_subarticular_stenosis_counts.groupby('level')['count'].transform('sum') * 100

right_subarticular_stenosis_columns = [column for column in df_train.columns if column.startswith('right_subarticular_stenosis')]
df_train_right_subarticular_stenosis = []

for column in right_subarticular_stenosis_columns:
    df = df_train[[column]].copy(deep=True)
    df['level'] = '_'.join(column.split('_')[-2:])
    df = df.rename(columns={column: 'right_subarticular_stenosis'})
    df_train_right_subarticular_stenosis.append(df)
    
df_train_right_subarticular_stenosis = pd.concat(df_train_right_subarticular_stenosis, axis=0).reset_index(drop=True)

df_train_right_subarticular_stenosis_counts = df_train_right_subarticular_stenosis.groupby('level').value_counts().reset_index()
df_train_right_subarticular_stenosis_counts['severity'] = df_train_right_subarticular_stenosis_counts['right_subarticular_stenosis'].map({
    'Normal/Mild': 0,
    'Moderate': 1,
    'Severe': 2
})
df_train_right_subarticular_stenosis_counts = df_train_right_subarticular_stenosis_counts.sort_values(by=['level', 'severity'], ascending=True)
df_train_right_subarticular_stenosis_counts['percentage'] = df_train_right_subarticular_stenosis_counts['count'] / df_train_right_subarticular_stenosis_counts.groupby('level')['count'].transform('sum') * 100

# 神经孔狭窄数据加载
neural_foraminal_narrowing_columns = [column for column in df_train.columns if 'neural_foraminal_narrowing' in column]
df_train_neural_foraminal_narrowing = []

for column in neural_foraminal_narrowing_columns:
    df = df_train[[column]].copy(deep=True)
    df['level'] = '_'.join(column.split('_')[-2:])
    df = df.rename(columns={column: 'neural_foraminal_narrowing'})
    df_train_neural_foraminal_narrowing.append(df)
    
df_train_neural_foraminal_narrowing = pd.concat(df_train_neural_foraminal_narrowing, axis=0).reset_index(drop=True)

df_train_neural_foraminal_narrowing_counts = df_train_neural_foraminal_narrowing.groupby('level').value_counts().reset_index()
df_train_neural_foraminal_narrowing_counts['severity'] = df_train_neural_foraminal_narrowing_counts['neural_foraminal_narrowing'].map({
    'Normal/Mild': 0,
    'Moderate': 1,
    'Severe': 2
})
df_train_neural_foraminal_narrowing_counts = df_train_neural_foraminal_narrowing_counts.sort_values(by=['level', 'severity'], ascending=True)
df_train_neural_foraminal_narrowing_counts['percentage'] = df_train_neural_foraminal_narrowing_counts['count'] / df_train_neural_foraminal_narrowing_counts.groupby('level')['count'].transform('sum') * 100

left_neural_foraminal_narrowing_columns = [column for column in df_train.columns if column.startswith('left_neural_foraminal_narrowing')]
df_train_left_neural_foraminal_narrowing = []

for column in left_neural_foraminal_narrowing_columns:
    df = df_train[[column]].copy(deep=True)
    df['level'] = '_'.join(column.split('_')[-2:])
    df = df.rename(columns={column: 'left_neural_foraminal_narrowing'})
    df_train_left_neural_foraminal_narrowing.append(df)
    
df_train_left_neural_foraminal_narrowing = pd.concat(df_train_left_neural_foraminal_narrowing, axis=0).reset_index(drop=True)

df_train_left_neural_foraminal_narrowing_counts = df_train_left_neural_foraminal_narrowing.groupby('level').value_counts().reset_index()
df_train_left_neural_foraminal_narrowing_counts['severity'] = df_train_left_neural_foraminal_narrowing_counts['left_neural_foraminal_narrowing'].map({
    'Normal/Mild': 0,
    'Moderate': 1,
    'Severe': 2
})
df_train_left_neural_foraminal_narrowing_counts = df_train_left_neural_foraminal_narrowing_counts.sort_values(by=['level', 'severity'], ascending=True)
df_train_left_neural_foraminal_narrowing_counts['percentage'] = df_train_left_neural_foraminal_narrowing_counts['count'] / df_train_left_neural_foraminal_narrowing_counts.groupby('level')['count'].transform('sum') * 100

right_neural_foraminal_narrowing_columns = [column for column in df_train.columns if column.startswith('right_neural_foraminal_narrowing')]
df_train_right_neural_foraminal_narrowing = []

for column in right_neural_foraminal_narrowing_columns:
    df = df_train[[column]].copy(deep=True)
    df['level'] = '_'.join(column.split('_')[-2:])
    df = df.rename(columns={column: 'right_neural_foraminal_narrowing'})
    df_train_right_neural_foraminal_narrowing.append(df)
    
df_train_right_neural_foraminal_narrowing = pd.concat(df_train_right_neural_foraminal_narrowing, axis=0).reset_index(drop=True)

df_train_right_neural_foraminal_narrowing_counts = df_train_right_neural_foraminal_narrowing.groupby('level').value_counts().reset_index()
df_train_right_neural_foraminal_narrowing_counts['severity'] = df_train_right_neural_foraminal_narrowing_counts['right_neural_foraminal_narrowing'].map({
    'Normal/Mild': 0,
    'Moderate': 1,
    'Severe': 2
})
df_train_right_neural_foraminal_narrowing_counts = df_train_right_neural_foraminal_narrowing_counts.sort_values(by=['level', 'severity'], ascending=True)
df_train_right_neural_foraminal_narrowing_counts['percentage'] = df_train_right_neural_foraminal_narrowing_counts['count'] / df_train_right_neural_foraminal_narrowing_counts.groupby('level')['count'].transform('sum') * 100

# 椎管狭窄数据
spinal_canal_stenosis_columns = [column for column in df_train.columns if column.startswith('spinal_canal_stenosis')]
df_train_spinal_canal_stenosis = []

for column in spinal_canal_stenosis_columns:
    df = df_train[[column]].copy(deep=True)
    df['level'] = '_'.join(column.split('_')[-2:])
    df = df.rename(columns={column: 'spinal_canal_stenosis'})
    df_train_spinal_canal_stenosis.append(df)
    
df_train_spinal_canal_stenosis = pd.concat(df_train_spinal_canal_stenosis, axis=0).reset_index(drop=True)

df_train_spinal_canal_stenosis_counts = df_train_spinal_canal_stenosis.groupby('level').value_counts().reset_index()
df_train_spinal_canal_stenosis_counts['severity'] = df_train_spinal_canal_stenosis_counts['spinal_canal_stenosis'].map({
    'Normal/Mild': 0,
    'Moderate': 1,
    'Severe': 2
})
df_train_spinal_canal_stenosis_counts = df_train_spinal_canal_stenosis_counts.sort_values(by=['level', 'severity'], ascending=True)
df_train_spinal_canal_stenosis_counts['percentage'] = df_train_spinal_canal_stenosis_counts['count'] / df_train_spinal_canal_stenosis_counts.groupby('level')['count'].transform('sum') * 100

df_train_spinal_canal_stenosis_counts

# 后端
api = Blueprint('api', __name__)
CORS(api, resources={
    r"/v1/*": {
        "origins": ["http://localhost:8080"],
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

@api.route('/v1/mic-predict', methods=['POST'])
def predict():
    print("predict")
    try:
        # 检查是否有文件
        if 'image' not in request.files:
            return jsonify({'error': '没有找到图片文件'}), 400
            
        file = request.files['image']
        
        # 检查文件是否为空
        if file.filename == '':
            return jsonify({'error': '文件名为空'}), 400
            
        # 确保目录存在
        save_dir = 'D:/lib'
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存图片
        img = Image.open(file.stream).convert('RGB')
        
        # # 转换为PNG格式
        # if img.format != 'PNG':
        #     img = img.convert('RGB')
        #     print("\n" + img.format + "\n")
        
        class_names = ["Axial_T2", "Sagittal_T1", "Sagittal_T2_STIR"]
        # 设置设备
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 加载模型
        model_path = 'workstation/competition/med-img-classify/mic-results/resnet50_model.pth'
        model = mic_predict.load_model(model_path, class_names, device)

        # 预测准确度
        predicted_label, probabilities = mic_predict.predict_image_with_prob_imageLoaded(img, model=model, class_names=class_names, device=device)
        print(predicted_label, probabilities)

        # 生成文件名（这里用时间戳作为示例）
        import time
        filename = f"image_{int(time.time())}.png"
        save_path = os.path.join(save_dir, filename)
        img.save(save_path, 'PNG')
        
        # 将处理后的图片转为base64（模拟返回处理结果）
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # 修改返回数据的格式: 可以返回更多参数, 只需按照这种方式组织数据就可以
        parameters = [{param_name: param_value} for param_name, param_value in probabilities]
        # parameters = [
        #     {'参数 1': 23.5},
        #     {'参数 2': 25.1},
        #     {'参数 3': 22.8},
        #     {'参数 4': 23.5},
        #     {'参数 5': 25.1},
        #     {'参数 6': 22.8},
        #     {'参数 7': 23.5},
        #     {'参数 8': 25.1},
        #     {'参数 9': 22.8},
        #     {'参数10': 23.5},
        #     # ... 其他参数
        # ]
        
        response = {
            'image': img_str,
            'label': predicted_label,
            'parameters': parameters
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"错误: {str(e)}")
        return jsonify({'error': str(e)}), 500
    

@api.route('/eda/heatmap', methods=['GET'])
def get_heatmap_data():
    cols = train_data_df.columns[1:]
    df_temp = train_data_df.copy()
    for column in cols:
        df_temp[column] = df_temp[column].astype('category').cat.codes

    # 计算相关性矩阵
    correlation_matrix = df_temp.corr()

    # 准备热图的数据
    x_axis = correlation_matrix.columns.tolist()
    y_axis = correlation_matrix.index.tolist()

    # 将相关性矩阵转换为 pyecharts 需要的二维列表格式
    data = []
    for i in range(len(y_axis)):
        for j in range(len(x_axis)):
            data.append([y_axis[i], x_axis[j], round(correlation_matrix.iloc[i, j], 2)])
            # data.append([i, j, round(correlation_matrix.iloc[i, j], 2)])

    print(data)
    # cols = train_data_df.columns[1:]
    # df_temp = train_data_df.copy()
    # for column in cols:
    #     df_temp[column] = df_temp[column].astype('category').cat.codes

    # # 计算相关性矩阵
    # correlation_matrix = df_temp.corr()

    # # 准备热图的数据
    # x_axis = correlation_matrix.columns.tolist()
    # y_axis = correlation_matrix.index.tolist()

    # # 将相关性矩阵转换为前端需要的二维列表格式
    # data = []
    # for i in range(len(y_axis)):
    #     for j in range(len(x_axis)):
    #         data.append([j, i, round(correlation_matrix.iloc[i, j], 2)])  # 注意：前端需要 [x, y, value] 格式
    
    print(data)

    # 返回数据
    return jsonify({
        "title": "不同节段不同疾病的相关性矩阵",
        "xAxis": x_axis,
        "yAxis": y_axis,
        "data": data
    })

@api.route('/eda/subarticular_stenosis_counts')
def get_subarticular_stenosis_counts():
    data = df_train_subarticular_stenosis_counts.to_dict(orient='records')
    return jsonify(data)

@api.route('/eda/left_subarticular_stenosis_counts')
def get_left_subarticular_stenosis_counts():
    data = df_train_left_subarticular_stenosis_counts.to_dict(orient='records')
    return jsonify(data)

@api.route('/eda/right_subarticular_stenosis_counts')
def get_right_subarticular_stenosis_counts():
    data = df_train_right_subarticular_stenosis_counts.to_dict(orient='records')
    return jsonify(data)

@api.route('/eda/neural_foraminal_narrowing_counts')
def get_neural_foraminal_narrowing_counts():
    data = df_train_neural_foraminal_narrowing_counts.to_dict(orient='records')
    return jsonify(data)

@api.route('/eda/left_neural_foraminal_narrowing_counts')
def get_left_neural_foraminal_narrowing_counts():
    data = df_train_left_neural_foraminal_narrowing_counts.to_dict(orient='records')
    return jsonify(data)

@api.route('/eda/right_neural_foraminal_narrowing_counts')
def get_right_neural_foraminal_narrowing_counts():
    data = df_train_right_neural_foraminal_narrowing_counts.to_dict(orient='records')
    return jsonify(data)

@api.route('/eda/spinal_canal_stenosis_counts')
def get_():
    data = df_train_spinal_canal_stenosis_counts.to_dict(orient='records')
    return jsonify(data)

@api.route('/eda/series_counts', methods=['GET'])
def get_series_counts():
    # Grouping by study_id and counting series_id occurrences
    series_counts = train_desc_df.groupby('study_id')['series_id'].count().value_counts().sort_index()
    
    # Prepare data for the frontend
    x_data = series_counts.index.astype(str).tolist()  # X轴数据
    y_data = series_counts.values.tolist()  # Y轴数据
    
    return jsonify({
        'xAxis': x_data,
        'yAxis': y_data
    })

@api.route('/eda/mri_image_coordinates_overview', methods=['GET'])
def get_scatter_data_overview():
    # Prepare data for the frontend
    x_data = df_train['x'].tolist()  # X轴数据
    y_data = df_train['y'].tolist()  # Y轴数据
    
    return jsonify({
        'xAxis': x_data,
        'yAxis': y_data
    })

@api.route('/eda/mri_image_coordinates_condition', methods=['GET'])
def get_scatter_data_condition():
    # Prepare data for the frontend
    conditions = df_train['condition'].unique().tolist()
    scatter_data = []
    for condition in conditions:
        condition_data = df_train[df_train['condition'] == condition][['x', 'y']].values.tolist()
        scatter_data.append({
            'name': condition,
            'data': condition_data
        })
    return jsonify(scatter_data)

@api.route('/eda/mri_image_coordinates_level', methods=['GET'])
def get_scatter_data_level():
    # Prepare data for the frontend
    x_data = df_train['x'].tolist()  # X轴数据
    y_data = df_train['y'].tolist()  # Y轴数据
    
    return jsonify({
        'xAxis': x_data,
        'yAxis': y_data
    })


@api.route('/v1/test', methods=['GET'])
def test():
    print("测试连接")
    return jsonify({'message': 'connection successful'})

