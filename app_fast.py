# main.py

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from model import MitralPlusRAG

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

templates = Jinja2Templates(directory="templates")
runner = MitralPlusRAG()

class InputModel(BaseModel):
    input_text: str
    use_rag: bool

@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process")
def process_input(data: InputModel):
    if data.use_rag:
        result = runner._return_with_RAG(data.input_text)
    else:
        result = runner._return_without_RAG(data.input_text)
    return {"result": result}