"""
Demo: agente que entrevista o usuário, decompõe uma ROTINA em TAREFAS e salva um JSON estruturado.
(LangChain v1 + Streamlit)

Middlewares:
  - TodoListMiddleware        (opcional via toggle) -> plano dinâmico: 1 item por tarefa identificada
  - SummarizationMiddleware   -> rede de segurança contra estouro de contexto
  - HumanInTheLoopMiddleware  -> usuário confirma/edita/rejeita o JSON antes de salvar

Instalação:
  pip install streamlit "langchain>=1.0" langgraph langchain-groq pydantic

Variáveis de ambiente:
  GROQ_API_KEY
  MODEL (opcional, ex.: "groq:openai/gpt-oss-120b"; o prefixo "groq:" é adicionado
         automaticamente se omitido. Confira os ids disponíveis na sua conta da Groq)

Rodar:
  streamlit run app.py
  # no Replit: streamlit run app.py --server.address 0.0.0.0 --server.port 8000
"""

import json
import os
import re
import uuid

import streamlit as st
from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    SummarizationMiddleware,
    TodoListMiddleware,
)
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel, Field

st.set_page_config(page_title="Mapeador de rotinas", page_icon="🗂️", layout="wide")

SAVE_DIR = "saved_routines"


# ---------------------------------------------------------------------------
# Schema do JSON final (usado como argumento da tool -> o LLM é forçado a segui-lo)
# ---------------------------------------------------------------------------
class Task(BaseModel):
    id: str = Field(description="Identificador curto em snake_case, ex.: coleta_assinatura")
    name: str = Field(description="Nome curto da tarefa")
    description: str = Field(description="Descrição simples do que a tarefa faz e por quê")
    platform: str = Field(description="Plataforma/ferramenta principal usada nesta tarefa")
    responsible: str = Field(description="Quem executa a tarefa (pessoa ou papel)")
    steps: list[str] = Field(
        description="Passos principais em ordem, em nível de ação (não precisa detalhar cada clique)"
    )
    inputs: list[str] = Field(default_factory=list, description="Dados/documentos necessários para começar")
    outputs: list[str] = Field(default_factory=list, description="Resultado/entregável ao terminar")
    exceptions: list[str] = Field(default_factory=list, description="Exceções e casos especiais")
    notes: str = Field(
        default="",
        description="Observação opcional, ex.: 'normalmente vem depois de X'. "
        "A tarefa deve poder ser executada/registrada de forma independente.",
    )


class Routine(BaseModel):
    name: str = Field(description="Nome da rotina")
    description: str = Field(description="Descrição geral da rotina, de ponta a ponta")
    trigger: str = Field(description="O que dispara a rotina e com que frequência")
    completion_criteria: str = Field(description="Como saber que a rotina terminou com sucesso")
    tasks: list[Task] = Field(description="Tarefas menores em que a rotina foi dividida, em ordem")


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
BASE_PROMPT = """Você é um analista de processos. Conversando com o usuário, você deve entender uma
ROTINA de trabalho inteira e dividi-la em TAREFAS menores e independentes, cada uma capaz de ser
registrada ou explicada em um momento diferente por quem a conhece. Depois, gere uma descrição de
cada tarefa e da rotina como um todo. A rotina pode ser de qualquer área; não assuma um tipo.

Conduza a conversa em fases:

FASE 1 - Entender a rotina
- Peça que o usuário conte a rotina do começo ao fim, com as palavras dele, sem interromper muito.
- Para cada trecho da rotina, descubra: plataforma/ferramenta usada, quem executa, passos
  principais, o que entra e o que sai, e exceções. Faça no máximo 1 a 2 perguntas por mensagem.
- Você precisa desse contexto (plataforma e passos) para decidir se um trecho é uma tarefa
  separada. Não decida a divisão sem ele.
- Se a resposta for vaga, peça o detalhe concreto.

FASE 2 - Dividir e confirmar
- Proponha a divisão em tarefas como lista numerada e peça confirmação ou ajustes.
- Abra uma nova tarefa quando mudar o objetivo, a plataforma/ferramenta ou o responsável, ou
  quando houver um ponto de espera (ex.: aguardar assinatura).
- Cada tarefa tem um objetivo único e deve poder ser feita sozinha, mesmo que normalmente venha
  depois de outra. Nesse caso, registre isso no campo notes.

FASE 3 - Salvar
- Com a divisão confirmada e cada tarefa com plataforma, responsável e passos principais, chame
  save_routine com a rotina completa (descrição geral + tarefas).
- O usuário precisará aprovar. Não diga que salvou antes de a ferramenta retornar sucesso. Se for
  rejeitada com feedback, ajuste e chame a ferramenta novamente.

Converse de forma natural e em português do Brasil. Nunca despeje todas as perguntas de uma vez."""

TODO_PROMPT = """

Gestão do plano (write_todos):
- Na primeira resposta, chame write_todos com: "Entender a rotina de ponta a ponta" (in_progress),
  "Propor divisão em tarefas e confirmar com o usuário" e "Gerar descrições e salvar" (pending).
- Conforme identificar os trechos da rotina, acrescente à lista um item
  "Confirmar plataforma e passos: <trecho>" para CADA trecho.
- Ao marcar um desses itens como completed, reescreva o texto do item incluindo um resumo curto do
  que foi apurado, ex.: "Envio do documento: DocuSign, feito pelo gestor, após validação do RG".
  Isso preserva a informação no estado, mesmo que o histórico seja sumarizado.
- Só proponha a divisão final quando todos os itens de trechos estiverem completed, e só chame
  save_routine depois que o usuário confirmar a divisão."""


def build_system_prompt(use_todo: bool) -> str:
    return BASE_PROMPT + (TODO_PROMPT if use_todo else "")


# ---------------------------------------------------------------------------
# Construção do agente
# ---------------------------------------------------------------------------
def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "rotina"


def build_agent(model: str, use_todo: bool, trigger_tokens: int, keep_messages: int, store: dict):
    @tool
    def save_routine(routine: Routine) -> str:
        """Salva a rotina completa (descrição geral + tarefas) como JSON estruturado.
        Só chame depois que o usuário confirmou a divisão em tarefas e todas foram detalhadas.
        Requer aprovação humana antes de executar."""
        data = routine.model_dump()
        os.makedirs(SAVE_DIR, exist_ok=True)
        path = os.path.join(SAVE_DIR, f"{slugify(data['name'])}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        store["routine"] = data  # <- a "variável" salva após o HITL
        store["path"] = path
        return f"Rotina salva com sucesso em {path}."

    middleware = []
    if use_todo:
        middleware.append(TodoListMiddleware())
    middleware.append(
        SummarizationMiddleware(
            model=model,
            trigger=("tokens", trigger_tokens),
            keep=("messages", keep_messages),
        )
    )
    middleware.append(
        HumanInTheLoopMiddleware(
            interrupt_on={
                "save_routine": {"allowed_decisions": ["approve", "edit", "reject"]},
            },
            description_prefix="Confirme a rotina antes de salvar",
        )
    )

    return create_agent(
        model=model,
        tools=[save_routine],
        system_prompt=build_system_prompt(use_todo),
        middleware=middleware,
        checkpointer=InMemorySaver(),  # obrigatório para o HITL poder pausar/retomar
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def to_text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)


def reset_session(cfg: tuple):
    st.session_state.cfg = cfg
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.store = {}
    st.session_state.chat = []
    st.session_state.pending = None
    st.session_state.nonce = 0
    model, use_todo, trigger_tokens, keep_messages = cfg
    st.session_state.agent = build_agent(
        model, use_todo, trigger_tokens, keep_messages, st.session_state.store
    )


def thread_config() -> dict:
    return {"configurable": {"thread_id": st.session_state.thread_id}}


def run_agent(payload):
    agent = st.session_state.agent
    try:
        with st.spinner("Pensando..."):
            result = agent.invoke(payload, config=thread_config())
    except Exception as e:  # noqa: BLE001
        st.error(f"Erro ao chamar o agente: {e}")
        return

    last_ai_text = ""
    for m in reversed(result["messages"]):
        if m.type == "ai":
            last_ai_text = to_text(m.content).strip()
            break
    if last_ai_text:
        st.session_state.chat.append({"role": "assistant", "content": last_ai_text})

    interrupts = result.get("__interrupt__")
    if interrupts:
        st.session_state.pending = interrupts[0].value["action_requests"][0]
    else:
        st.session_state.pending = None


def render_routine(routine: dict):
    st.markdown(f"### {routine.get('name', '(sem nome)')}")
    st.write(routine.get("description", ""))
    st.caption(
        f"**Gatilho:** {routine.get('trigger', '-')}  \n"
        f"**Conclusão:** {routine.get('completion_criteria', '-')}"
    )
    for i, t in enumerate(routine.get("tasks", []), 1):
        with st.expander(f"{i}. {t.get('name')} · {t.get('platform')}"):
            st.write(t.get("description", ""))
            st.markdown(f"**Responsável:** {t.get('responsible', '-')}")
            st.markdown("**Passos**")
            for j, s in enumerate(t.get("steps", []), 1):
                st.markdown(f"{j}. {s}")
            if t.get("inputs"):
                st.markdown("**Entradas:** " + "; ".join(t["inputs"]))
            if t.get("outputs"):
                st.markdown("**Saídas:** " + "; ".join(t["outputs"]))
            if t.get("exceptions"):
                st.markdown("**Exceções:** " + "; ".join(t["exceptions"]))
            if t.get("notes"):
                st.markdown("**Obs.:** " + t["notes"])


# ---------------------------------------------------------------------------
# Sidebar: configuração
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuração")
    model = st.text_input("Modelo", value=os.getenv("MODEL", "groq:openai/gpt-oss-120b"))
    use_todo = st.toggle("Usar TodoListMiddleware", value=True)
    st.caption("Mudar qualquer opção abaixo reinicia a conversa.")
    trigger_tokens = st.slider(
        "Sumarizar a partir de (tokens)", 500, 20000, 6000, step=500,
        help="Reduza para ver a sumarização acontecer na demo.",
    )
    keep_messages = st.slider("Mensagens recentes mantidas", 2, 20, 8)
    if st.button("🔄 Nova conversa", use_container_width=True):
        st.session_state.pop("cfg", None)
    st.divider()
    status_box = st.container()

model = model.strip()
if ":" not in model:  # aceita "openai/gpt-oss-120b" e completa com o provedor
    model = f"groq:{model}"

cfg = (model, use_todo, trigger_tokens, keep_messages)
if st.session_state.get("cfg") != cfg:
    reset_session(cfg)

# ---------------------------------------------------------------------------
# Corpo principal
# ---------------------------------------------------------------------------
st.title("🗂️ Mapeador de rotinas")
st.caption(
    f"Modo: **{'COM' if use_todo else 'SEM'} TodoList** · "
    "HITL na tool `save_routine` · Summarization ativo"
)

pending = st.session_state.pending
prompt = st.chat_input(
    "Conte como funciona a rotina que você quer documentar..." if not pending
    else "Aguardando sua confirmação abaixo",
    disabled=pending is not None,
)
if prompt:
    st.session_state.chat.append({"role": "user", "content": prompt})
    run_agent({"messages": [{"role": "user", "content": prompt}]})
    pending = st.session_state.pending

if not st.session_state.chat:
    st.info(
        "Comece contando uma rotina, ex.: "
        '"Quero explicar como fazemos a abertura de conta de um novo cliente no banco".'
    )

for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Card de aprovação (Human-in-the-loop)
# ---------------------------------------------------------------------------
if pending:
    args = pending["args"]
    original = args.get("routine", args)
    key = f"{st.session_state.thread_id}_{st.session_state.nonce}"
    with st.container(border=True):
        st.subheader("✋ Confirmação necessária")
        st.write("O agente quer salvar esta rotina. Confirme, edite o JSON ou peça ajustes.")
        render_routine(original)
        with st.expander("✏️ Editar JSON"):
            edited_text = st.text_area(
                "JSON da rotina",
                value=json.dumps(original, ensure_ascii=False, indent=2),
                height=400,
                key=f"json_{key}",
            )
        feedback = st.text_input("Se quiser pedir ajustes, diga o que mudar:", key=f"fb_{key}")
        c1, c2 = st.columns(2)
        confirm = c1.button("✅ Confirmar e salvar", type="primary", use_container_width=True)
        reject = c2.button("↩️ Pedir ajustes", use_container_width=True)

    decision, note = None, ""
    if confirm:
        try:
            edited = json.loads(edited_text)
            Routine.model_validate(edited)  # valida o schema antes de seguir
        except Exception as e:  # noqa: BLE001
            st.error(f"JSON inválido: {e}")
        else:
            if edited != original:
                decision = {
                    "type": "edit",
                    "edited_action": {"name": pending["name"], "args": {"routine": edited}},
                }
                note = "✏️ Editei o JSON e confirmei."
            else:
                decision = {"type": "approve"}
                note = "✅ Confirmado."
    elif reject:
        fb = feedback.strip() or "O usuário não aprovou. Pergunte o que deve ser ajustado."
        decision = {"type": "reject", "message": fb}
        note = f"↩️ Pedi ajustes: {fb}"

    if decision:
        st.session_state.chat.append({"role": "user", "content": note})
        st.session_state.pending = None
        st.session_state.nonce += 1
        run_agent(Command(resume={"decisions": [decision]}))
        st.rerun()

# ---------------------------------------------------------------------------
# Resultado salvo
# ---------------------------------------------------------------------------
saved = st.session_state.store.get("routine")
if saved:
    st.success(f"Rotina salva em `{st.session_state.store.get('path')}` e em `store['routine']` ✔")
    with st.expander("Ver JSON salvo", expanded=True):
        st.json(saved)
    st.download_button(
        "⬇️ Baixar JSON",
        json.dumps(saved, ensure_ascii=False, indent=2),
        file_name=f"{slugify(saved['name'])}.json",
        mime="application/json",
    )

# ---------------------------------------------------------------------------
# Sidebar: estado do agente (todos + sumarização)
# ---------------------------------------------------------------------------
with status_box:
    try:
        values = st.session_state.agent.get_state(thread_config()).values
    except Exception:  # noqa: BLE001
        values = {}

    st.subheader("📋 Plano (todos)")
    if not use_todo:
        st.caption("TodoListMiddleware desativado: o agente conduz só pelo prompt.")
    else:
        todos = values.get("todos") or []
        if not todos:
            st.caption("Ainda não criado pelo agente.")
        icons = {"completed": "✅", "in_progress": "🔄", "pending": "⬜"}
        for t in todos:
            st.write(f"{icons.get(t.get('status'), '⬜')} {t.get('content')}")
        if todos:
            done = sum(t.get("status") == "completed" for t in todos)
            st.progress(done / len(todos), text=f"{done}/{len(todos)} itens")

    st.subheader("🧠 Contexto")
    msgs = values.get("messages", [])
    summarized = any(
        m.type == "human" and to_text(m.content).lstrip().lower().startswith("here is a summary")
        for m in msgs
    )
    st.write(f"Mensagens no estado do agente: **{len(msgs)}**")
    st.write("Histórico sumarizado: " + ("**sim** ✅" if summarized else "ainda não"))