import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
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
TIMEZONE = ZoneInfo("America/Sao_Paulo")

# Palavras que indicam que o usuário quer informação da internet
SEARCH_HINTS = (
    "pesquis", "busque", "buscar", "procure", "internet", "google", "site",
    "hoje", "agora", "atual", "recente", "notícia", "noticia", "últim", "ultim",
    "liturgia", "evangelho", "cotação", "cotacao", "previsão", "previsao",
    "imagem", "imagens", "foto", "figura",
)
# Frases que indicam que o modelo desistiu de pesquisar
REFUSAL_HINTS = (
    "não tenho acesso", "nao tenho acesso", "não consigo acessar", "nao consigo acessar",
    "não posso acessar", "nao posso acessar", "não posso navegar", "não consigo navegar",
    "tempo real", "recomendo consultar", "recomendo acessar", "sugiro consultar",
    "sugiro acessar", "você pode consultar", "voce pode consultar", "consulte o site",
)
NUDGE_MESSAGE = (
    "Você TEM acesso à internet. Não sugira sites: use agora a ferramenta pesquisar_web "
    "(ou buscar_imagens, se pedi imagens; e ler_pagina se precisar) e responda com o que encontrar."
)

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


def buscar_imagens(consulta: str, max_resultados: int = 4) -> list[dict]:
    resultados = DDGS().images(consulta, region="br-pt", max_results=max_resultados)
    return [
        {
            "titulo": r.get("title"),
            "pagina": r.get("url"),
            "markdown": f"[![{(r.get('title') or 'imagem').replace('[', '').replace(']', '')}]"
            f"({(r.get('thumbnail') or '').replace('+', '%2B')})]({r.get('image')})",
        }
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
    data = data or datetime.now(TIMEZONE).strftime("%d/%m/%Y")
    dia, mes, ano = data.split("/")
    params = {"dia": dia, "mes": mes, "ano": ano}
    response = requests.get(LITURGIA_URL, params=params, timeout=20)
    response.raise_for_status()
    return response.json()


TOOL_FUNCTIONS = {
    "pesquisar_web": pesquisar_web,
    "buscar_imagens": buscar_imagens,
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
            "name": "buscar_imagens",
            "description": "Busca imagens/fotos na internet. Cada resultado traz um campo 'markdown' "
            "pronto: copie-o exatamente na resposta para a imagem aparecer para o usuário.",
            "parameters": {
                "type": "object",
                "properties": {
                    "consulta": {"type": "string", "description": "O que buscar"},
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
    hoje = datetime.now(TIMEZONE).strftime("%d/%m/%Y %H:%M")
    return (
        f"Você é o Free GPT, um assistente que responde em português. Agora é {hoje} "
        "(horário de Brasília). Você TEM acesso à internet pelas ferramentas pesquisar_web, "
        "buscar_imagens, ler_pagina e liturgia_diaria. Quando o usuário pedir imagem, foto ou "
        "figura, use buscar_imagens e coloque na resposta o campo 'markdown' de cada resultado, "
        "sem alterar. Sempre que o usuário pedir uma pesquisa ou a pergunta "
        "envolver fatos, pessoas, lugares, história, notícias ou informações atuais, use "
        "pesquisar_web e, se os resumos forem curtos, ler_pagina nos melhores links. "
        "Nunca diga que não tem acesso à internet e nunca responda apenas sugerindo sites: "
        "leia as fontes e responda com o conteúdo encontrado, citando os links no final."
    )


def wants_search(text: str) -> bool:
    text = text.lower()
    return any(hint in text for hint in SEARCH_HINTS)


def looks_like_refusal(text: str) -> bool:
    text = text.lower()
    return any(hint in text for hint in REFUSAL_HINTS)


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

    used_tools = False
    nudged = False
    asked_search = wants_search(body.message)

    for _ in range(MAX_TOOL_ROUNDS):
        message = call_openrouter(api_key, messages)
        tool_calls = message.get("tool_calls")
        if not tool_calls:
            content = message.get("content") or ""
            if not used_tools and not nudged and (asked_search or looks_like_refusal(content)):
                nudged = True
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": NUDGE_MESSAGE})
                continue
            return MessageResponse(reply=content)

        used_tools = True

        messages.append({"role": "assistant", "content": message.get("content"), "tool_calls": tool_calls})
        for call in tool_calls:
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": run_tool(call["function"]["name"], call["function"].get("arguments")),
            })

    raise HTTPException(status_code=502, detail="O modelo excedeu o limite de pesquisas.")
