# api/main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from api.model import predict_paciente
from api.schemas import PacienteInput
import uvicorn

app = FastAPI(
    title="Sistema de Diagnóstico de Câncer de Mama",
    description="API que recebe dados do paciente e retorna predição de tumor maligno/benigno.",
    version="1.0.0"
)

@app.get("/")
def home():
    return {"mensagem": "API de Diagnóstico de Câncer de Mama", "status": "online"}

@app.post("/predict", response_model=dict)
def predict(paciente: PacienteInput):
    try:
        resultado = predict_paciente(paciente.model_dump())
        return resultado
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Rodar com: uvicorn api.main:app --reload
if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)