# Mapeador de rotinas

Demo de um agente conversacional que entrevista uma pessoa sobre uma rotina de trabalho, divide essa rotina em tarefas menores e independentes, e salva o resultado em um JSON estruturado depois de uma confirmação humana.

Feito com LangChain (agentes + middlewares), Groq (LLM) e Streamlit (interface).

## Visão geral

O fluxo da conversa tem três fases:

1. **Entender a rotina:** a pessoa conta como a rotina funciona. O agente faz perguntas pontuais para descobrir, em cada trecho, a plataforma usada, quem executa e os passos principais.
2. **Dividir e confirmar:** o agente propõe a divisão em tarefas (nova tarefa quando muda o objetivo, a plataforma ou o responsável, ou quando há uma espera) e pede o ok da pessoa.
3. **Salvar:** o agente monta a descrição geral e a de cada tarefa e chama a tool `save_routine`. A execução pausa para o usuário aprovar, editar ou rejeitar. Só depois disso o JSON é gravado em `saved_routines/` e guardado em `store["routine"]`.

O JSON final segue um schema Pydantic (`Routine` com uma lista de `Task`), que é usado como argumento da própria tool. Isso força o modelo a gerar o formato certo e permite validar qualquer edição manual antes de salvar.

## Middlewares usados

### TodoListMiddleware

**O que faz:** dá ao agente a tool `write_todos`, que mantém uma lista de tarefas com status (`pending`, `in_progress`, `completed`) no estado do agente.

**Por que usamos:** a rotina pode ser de qualquer área, então não existe um checklist fixo. O agente cria o plano dinamicamente: um item para cada trecho da rotina que precisa de confirmação de plataforma e passos. Isso o ajuda a não esquecer de perguntar nada e dá visibilidade do progresso na barra lateral. Ao concluir um item, o agente reescreve o texto com um resumo do que apurou, para que a informação fique no estado mesmo que o histórico seja sumarizado.

Na interface há um toggle para ligar e desligar este middleware, para comparar o comportamento do agente com e sem o plano.

### SummarizationMiddleware

**O que faz:** quando o histórico passa de um limite de tokens, resume as mensagens antigas e mantém só as mais recentes.

**Por que usamos:** como rede de segurança contra estouro da janela de contexto em conversas longas. O limite (6000 tokens por padrão) e a quantidade de mensagens mantidas são ajustáveis na barra lateral; para ver o efeito na demo, reduza o limite.

### HumanInTheLoopMiddleware

**O que faz:** pausa a execução antes de uma tool específica e espera uma decisão humana: aprovar, editar os argumentos ou rejeitar com feedback.

**Por que usamos:** é a etapa de confirmação. Antes de salvar, a pessoa vê a rotina renderizada (e o JSON editável) e decide se é isso mesmo. Se rejeitar com um comentário, o agente ajusta e tenta de novo. Também funciona como validação final: se o agente deixou algo passar, a revisão humana pega. Este middleware exige um checkpointer (`InMemorySaver`) para poder pausar e retomar a conversa.

## Como rodar

```powershell
pip install streamlit "langchain>=1.0" langgraph langchain-groq pydantic
$env:GROQ_API_KEY="sua_chave_aqui"
streamlit run app.py
```

O modelo padrão é `groq:openai/gpt-oss-120b` e pode ser trocado na barra lateral (o prefixo `groq:` é adicionado automaticamente se omitido). Escolha um modelo com bom suporte a tool calling.

## Estrutura

```
app.py              # agente, middlewares, schema e interface Streamlit
saved_routines/     # JSONs salvos após a aprovação (criada automaticamente)
```

## Limitações

- Os dados da conversa ficam em memória (`InMemorySaver`); reiniciar o app apaga o estado.
- O status dos itens do todo é decidido pelo próprio modelo, então a confirmação humana continua sendo a validação final.
- Projeto de demonstração: não há autenticação, persistência em banco nem tratamento de erros avançado para chamadas de tool inválidas.