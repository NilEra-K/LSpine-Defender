from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import shutil
import os
import sys
import importlib
import json
import time

from utils import data_transform
from utils import file_operation
sys.path.append("workstation/competition/one-stage")

rsna_predict = importlib.import_module("rsna-predict")

app = FastAPI()

# 配置 CORS
origins = [
    "http://localhost:8081",  # 允许前端应用的地址
    "http://127.0.0.1:8081",  # 如果您使用 127.0.0.1 作为前端地址
    "http://localhost:3000",  # 如果您使用其他端口
    "http://127.0.0.1:3000",
    "http://localhost",       # 其他可能的地址
    "http://127.0.0.1"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有 HTTP 方法
    allow_headers=["*"],  # 允许所有头部
)


@app.get("/api/v1/test")
async def read_test():
    return {"message": "This is a test endpoint"}


@app.get("/api/v1/multi-models-predict")
async def read_test():
    return {"message": "/api/v1/multi-models-predict 测试接口"}

# 定义返回的参数模型
class PredictionResult(BaseModel):
    row_id: str
    normal_mild: float
    moderate: float
    severe: float

@app.post("/api/v1/predict")
async def severityPredict(file: UploadFile, algorithm: str = Form(...)):
    """
    接收前端上传的文件和算法选择，返回预测结果。
    """
    # 检查文件类型
    if file.content_type not in ["application/zip", "application/x-tar", "application/gzip", "application/x-zip-compressed", "application/octet-stream"]:
        raise HTTPException(status_code=400, detail="仅支持 zip、tar、gz 格式的文件！")

    print(file)
    print(algorithm)

    # 保存上传的文件到本地
    local_data_dir = "workstation/backend/uploaded"
    try:
        # 确保 data 目录存在
        if not os.path.exists(local_data_dir):
            os.makedirs(local_data_dir)

        # 定义保存路径
        save_path = os.path.join(local_data_dir, file.filename)

        # 将文件保存到指定路径
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f">>> [LOGS] 文件已成功保存到 {save_path}")
        with open(file.filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        file_operation.extract_file(local_data_dir + "/" + file.filename, local_data_dir + "/" + file_operation.remove_extension(file.filename))
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件保存失败: {e}")
    
    if algorithm == "algorithm1":
        print(">>> [LOGS] algorithm 1")
        # predict_result = {"row_id":{"0":"44036939_spinal_canal_stenosis_l1_l2","1":"44036939_spinal_canal_stenosis_l2_l3","2":"44036939_spinal_canal_stenosis_l3_l4","3":"44036939_spinal_canal_stenosis_l4_l5","4":"44036939_spinal_canal_stenosis_l5_s1","5":"44036939_left_neural_foraminal_narrowing_l1_l2","6":"44036939_left_neural_foraminal_narrowing_l2_l3","7":"44036939_left_neural_foraminal_narrowing_l3_l4","8":"44036939_left_neural_foraminal_narrowing_l4_l5","9":"44036939_left_neural_foraminal_narrowing_l5_s1","10":"44036939_right_neural_foraminal_narrowing_l1_l2","11":"44036939_right_neural_foraminal_narrowing_l2_l3","12":"44036939_right_neural_foraminal_narrowing_l3_l4","13":"44036939_right_neural_foraminal_narrowing_l4_l5","14":"44036939_right_neural_foraminal_narrowing_l5_s1","15":"44036939_left_subarticular_stenosis_l1_l2","16":"44036939_left_subarticular_stenosis_l2_l3","17":"44036939_left_subarticular_stenosis_l3_l4","18":"44036939_left_subarticular_stenosis_l4_l5","19":"44036939_left_subarticular_stenosis_l5_s1","20":"44036939_right_subarticular_stenosis_l1_l2","21":"44036939_right_subarticular_stenosis_l2_l3","22":"44036939_right_subarticular_stenosis_l3_l4","23":"44036939_right_subarticular_stenosis_l4_l5","24":"44036939_right_subarticular_stenosis_l5_s1"},"normal_mild":{"0":0.567779094,"1":0.2940558121,"2":0.177681433,"3":0.2102441899,"4":0.6839972064,"5":0.637311101,"6":0.3963948563,"7":0.2488650419,"8":0.1603668761,"9":0.2138342895,"10":0.6142781675,"11":0.4411704838,"12":0.2192274407,"13":0.1554939896,"14":0.2277899086,"15":0.4572613649,"16":0.1911141369,"17":0.0927640609,"18":0.0789845893,"19":0.2823046297,"20":0.4451997504,"21":0.2116134558,"22":0.0944291679,"23":0.0775610348,"24":0.2774010636},"moderate":{"0":0.2687550038,"1":0.3797519654,"2":0.328002084,"3":0.2359720953,"4":0.1766417958,"5":0.2986501977,"6":0.4965074286,"7":0.5180035904,"8":0.4286295623,"9":0.3646043316,"10":0.2648667544,"11":0.4735741317,"12":0.53430406,"13":0.3950768113,"14":0.341448836,"15":0.3431036435,"16":0.3863891959,"17":0.3359911591,"18":0.2490087114,"19":0.3213812113,"20":0.368171066,"21":0.3967567757,"22":0.3244051486,"23":0.2613239437,"24":0.3315833025},"severe":{"0":0.1634659432,"1":0.3261922449,"2":0.4943164438,"3":0.5537837371,"4":0.1393610034,"5":0.0640387041,"6":0.1070977254,"7":0.2331313714,"8":0.4110035524,"9":0.4215613827,"10":0.120855093,"11":0.0852553817,"12":0.2464685272,"13":0.4494292066,"14":0.4307612628,"15":0.1996349767,"16":0.4224966615,"17":0.5712447986,"18":0.6720067263,"19":0.3963141218,"20":0.1866291985,"21":0.3916297406,"22":0.5811656639,"23":0.6611149982,"24":0.3910156377}}
        row_names, y_preds = rsna_predict.inference_model_with_zip(local_data_dir + "/" + file_operation.remove_extension(file.filename))
        predict_result = rsna_predict.output_result(row_names=row_names, y_preds=y_preds)
        # print(type(predict_result))
        result = data_transform.transform_prediction_data(json.loads(predict_result))
        print(result)

        # 模拟预测逻辑
        # 在实际应用中，您可以在这里调用您的模型或处理逻辑
        # result = [
        #     {"row_id": "1", "normal_mild": 0.1, "moderate": 0.3, "severe": 0.6},
        #     {"row_id": "2", "normal_mild": 0.2, "moderate": 0.4, "severe": 0.4},
        #     # 添加更多模拟结果
        # ]
        # print(result)
        file_operation.delete_folder(local_data_dir + "/" + file_operation.remove_extension(file.filename))
        file_operation.delete_file(local_data_dir + "/" + file.filename)

        # 返回预测结果
        return JSONResponse(content={"parameters": result}, status_code=200)
    elif algorithm == "algorithm2":
        print(">>> [LOGS] algorithm 1") 



# @app.post("/api/v1/predict")
# async def severityPredict():
#     return {"message": "/api/v1/multi-models-predict 测试接口"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=16022)
