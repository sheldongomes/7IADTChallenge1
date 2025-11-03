from fastapi import FastAPI, HTTPException, Request
from api.model import predict_paciente
from api.schemas import PacienteInput
import uvicorn

app = FastAPI(
    title="Breast Cancer Diagnostic System",
    description="API that receives patient data and return a prediction if tumor is Malignant/Benign.",
    version="1.0.0"
)

@app.get("/")
def home():
    return {"mensagem": "Breast Cancer Diagnostic API", "status": "online"}

@app.post("/predict", response_model=dict)
@app.post("/predict/best", response_model=dict)
@app.post("/predict/logistic_regression", response_model=dict)
@app.post("/predict/random_forest", response_model=dict)
@app.post("/predict/svm", response_model=dict)
def predict(paciente: PacienteInput, request: Request):
    model = None if request.url.path.count('/') == 1 else request.url.path.split('/')[2]
    print(model)
    try:
        resultado = predict_paciente(paciente.model_dump(), model)
        return resultado
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Execute with: uvicorn api.main:app --reload
if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)