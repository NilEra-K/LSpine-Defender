def transform_prediction_data(prediction_data):
    """
    将预测数据从嵌套字典格式转换为列表格式。

    参数:
        prediction_data (dict): 原始预测数据，包含嵌套的字典。

    返回:
        list: 转换后的预测数据列表，每个元素是一个字典，包含 row_id 和预测值。
    """
    result = []
    row_ids = prediction_data.get("row_id", {})
    normal_mild = prediction_data.get("normal_mild", {})
    moderate = prediction_data.get("moderate", {})
    severe = prediction_data.get("severe", {})

    # 遍历 row_id 的键值对
    for key in row_ids.keys():
        row_id = row_ids.get(key)
        nm = normal_mild.get(key, 0)  # 如果没有值，默认为 0
        md = moderate.get(key, 0)
        sv = severe.get(key, 0)

        result.append({
            "row_id": row_id,
            "normal_mild": nm,
            "moderate": md,
            "severe": sv
        })

    return result