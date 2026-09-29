import json
import os
from datetime import datetime
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")  # fallback: .env na raiz do projeto

MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
LITURGIA_URL = "https://liturgia.up.railway.app/v2/"
FRONTEND_FILE = BASE_DIR.parent / "frontend" / "index.html"

# Mensagens definidas pelo FastAPI (altere aqui)
WELCOME_MESSAGE = "Bem-vindo ao Free GPT! Use POST /send_message para conversar."
DEFAULT_MESSAGE = "Olá! Me dê as boas-vindas em uma frase."

MAX_TOOL_ROUNDS = 5
MAX_TOOL_OUTPUT = 12000

app = FastAPI(title="Free GPT", description=WELCOME_MESSAGE)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class MessageRequest(BaseModel):
    message: str = Field(
        default=DEFAULT_MESSAGE,
        description="Mensagem enviada ao modelo (padrão vem do FastAPI)",
        examples=[DEFAULT_MESSAGE],
    )


class MessageResponse(BaseModel):
    reply: str


# ---------- Ferramentas de internet ----------

def pesquisar_web(consulta: str, max_resultados: int = 5) -> list[dict]:
    resultados = DDGS().text(consulta, region="br-pt", max_results=max_resultados)
    return [
        {"titulo": r.get("title"), "url": r.get("href"), "resumo": r.get("body")}
        for r in resultados
    ]


def ler_pagina(url: str) -> str:
    response = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0 (FreeGPT)"},
        timeout=20,
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()
    texto = "\n".join(linha.strip() for linha in soup.get_text("\n").splitlines() if linha.strip())
    return texto


def liturgia_diaria(data: str | None = None) -> dict:
    params = {}
    if data:
        dia, mes, ano = data.split("/")
        params = {"dia": dia, "mes": mes, "ano": ano}
    response = requests.get(LITURGIA_URL, params=params, timeout=20)
    response.raise_for_status()
    return response.json()


TOOL_FUNCTIONS = {
    "pesquisar_web": pesquisar_web,
    "ler_pagina": ler_pagina,
    "liturgia_diaria": liturgia_diaria,
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "pesquisar_web",
            "description": "Pesquisa na internet (DuckDuckGo) e retorna títulos, links e resumos. "
            "Use para notícias, fatos recentes ou qualquer informação que você não saiba.",
            "parameters": {
                "type": "object",
                "properties": {
                    "consulta": {"type": "string", "description": "Termos da pesquisa"},
                },
                "required": ["consulta"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ler_pagina",
            "description": "Abre uma página da internet e retorna o texto dela. "
            "Use depois de pesquisar_web quando o resumo não for suficiente.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Endereço completo da página"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "liturgia_diaria",
            "description": "Retorna a liturgia diária católica (leituras, salmo, evangelho, "
            "orações e cor litúrgica). Sem data, retorna a de hoje.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {"type": "string", "description": "Data no formato DD/MM/AAAA (opcional)"},
                },
            },
        },
    },
]


def run_tool(name: str, arguments: str) -> str:
    func = TOOL_FUNCTIONS.get(name)
    if not func:
        return f"Ferramenta desconhecida: {name}"
    try:
        args = json.loads(arguments or "{}")
        result = func(**args)
    except Exception as e:
        return f"Erro ao executar {name}: {e}"
    if not isinstance(result, str):
        result = json.dumps(result, ensure_ascii=False)
    return result[:MAX_TOOL_OUTPUT]


def system_prompt() -> str:
    hoje = datetime.now().strftime("%d/%m/%Y %H:%M")
    return (
        f"Você é o Free GPT, um assistente que responde em português. Agora é {hoje}. "
        "Você tem acesso à internet pelas ferramentas disponíveis: use-as sempre que a pergunta "
        "envolver informações atuais, notícias ou a liturgia do dia. Cite as fontes (links) "
        "quando usar a pesquisa na web."
    )


# ---------- Rotas ----------

@app.get("/", include_in_schema=False)
def home():
    """Página do chat."""
    return FileResponse(FRONTEND_FILE)


@app.get("/health")
def health():
    """Mensagem de boas-vindas (vem do FastAPI)."""
    return {"message": WELCOME_MESSAGE}


def call_openrouter(api_key: str, messages: list[dict]) -> dict:
    response = requests.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": MODEL, "messages": messages, "tools": TOOLS},
        timeout=90,
    )

    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=response.text)

    data = response.json()
    if "choices" not in data:
        error = data.get("error", {})
        message = error.get("message") if isinstance(error, dict) else str(error)
        raise HTTPException(status_code=502, detail=f"OpenRouter: {message or data}")

    return data["choices"][0]["message"]


@app.post("/send_message", response_model=MessageResponse)
def send_message(body: MessageRequest):
    api_key = os.getenv("openai_api_key")
    if not api_key:
        raise HTTPException(status_code=500, detail="API key não configurada")

    messages = [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": body.message},
    ]

    for _ in range(MAX_TOOL_ROUNDS):
        message = call_openrouter(api_key, messages)
        tool_calls = message.get("tool_calls")
        if not tool_calls:
            return MessageResponse(reply=message.get("content") or "")

        messages.append({"role": "assistant", "content": message.get("content"), "tool_calls": tool_calls})
        for call in tool_calls:
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": run_tool(call["function"]["name"], call["function"].get("arguments")),
            })

    raise HTTPException(status_code=502, detail="O modelo excedeu o limite de pesquisas.")
