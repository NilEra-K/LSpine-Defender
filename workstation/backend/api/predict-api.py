from fastapi import FastAPI

app = FastAPI()

@app.get("/api/v1/test")
async def read_test():
    return {"message": "This is a test endpoint"}

@app.get("/api/v1/one-stage-predict")
async def oneStagePredict(image):
    return {"message": "/api/v1/one-stage-predict 测试接口"}

@app.get("/api/v1/multi-models-predict")
async def read_test():
    return {"message": "/api/v1/multi-models-predict 测试接口"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=16022)
