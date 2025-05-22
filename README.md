# Projeto de Análise de Logs de Rede com RAG e Mistral AI

## Sumário

  * [1. Visão Geral do Projeto](https://www.google.com/search?q=%231-vis%C3%A3o-geral-do-projeto)
  * [2. O Que É RAG (Retrieval-Augmented Generation)?](https://www.google.com/search?q=%232-o-que-%C3%A9-rag-retrieval-augmented-generation)
  * [3. Objetivos Deste Protótipo](https://www.google.com/search?q=%233-objetivos-deste-prot%C3%B3tipo)
  * [4. Arquitetura do Sistema](https://www.google.com/search?q=%234-arquitetura-do-sistema)
  * [5. Tecnologias Utilizadas](https://www.google.com/search?q=%235-tecnologias-utilizadas)
      * [5.1. Python](https://www.google.com/search?q=%2351-python)
      * [5.2. Mistral AI](https://www.google.com/search?q=%2352-mistral-ai)
      * [5.3. LangChain](https://www.google.com/search?q=%2353-langchain)
      * [5.4. `sentence-transformers`](https://www.google.com/search?q=%2354-sentence-transformers)
      * [5.5. ChromaDB](https://www.google.com/search?q=%2355-chromadb)
      * [5.6. `python-dotenv`](https://www.google.com/search?q=%2356-python-dotenv)
  * [6. Configuração e Instalação](https://www.google.com/search?q=%236-configura%C3%A7%C3%A3o-e-instala%C3%A7%C3%A3o)
      * [6.1. Pré-requisitos](https://www.google.com/search?q=%2361-pr%C3%A9-requisitos)
      * [6.2. Clonar o Repositório](https://www.google.com/search?q=%2362-clonar-o-reposit%C3%B3rio)
      * [6.3. Configurar a Chave de API do Mistral AI](https://www.google.com/search?q=%2363-configurar-a-chave-de-api-do-mistral-ai)
      * [6.4. Instalar Dependências](https://www.google.com/search?q=%2364-instalar-depend%C3%AAncias)
  * [7. Como Executar o Protótipo](https://www.google.com/search?q=%237-como-executar-o-prot%C3%B3tipo)
      * [7.1. Estrutura dos Logs de Rede](https://www.google.com/search?q=%2371-estrutura-dos-logs-de-rede)
      * [7.2. Executando o Script](https://www.google.com/search?q=%2372-executando-o-script)
      * [7.3. Exemplo de Interação](https://www.google.com/search?q=%2373-exemplo-de-intera%C3%A7%C3%A3o)
  * [8. Detalhamento do Código (`rag_mistral_logs.py`)](https://www.google.com/search?q=%238-detalhamento-do-c%C3%B3digo-rag_mistral_logspy)
      * [8.1. Carregamento das Variáveis de Ambiente](https://www.google.com/search?q=%2381-carregamento-das-vari%C3%A1veis-de-ambiente)
      * [8.2. Carregamento dos Logs de Rede](https://www.google.com/search?q=%2382-carregamento-dos-logs-de-rede)
      * [8.3. Divisão dos Documentos em Chunks](https://www.google.com/search?q=%2383-divis%C3%A3o-dos-documentos-em-chunks)
      * [8.4. Geração de Embeddings](https://www.google.com/search?q=%2384-gera%C3%A7%C3%A3o-de-embeddings)
      * [8.5. Armazenamento em um Banco de Vetores (ChromaDB)](https://www.google.com/search?q=%2385-armazenamento-em-um-banco-de-vetores-chromadb)
      * [8.6. Configuração do Retriever](https://www.google.com/search?q=%2386-configura%C3%A7%C3%A3o-do-retriever)
      * [8.7. Configuração do Modelo Mistral AI](https://www.google.com/search?q=%2387-configura%C3%A7%C3%A3o-do-modelo-mistral-ai)
      * [8.8. Criação do Prompt Template](https://www.google.com/search?q=%2388-cria%C3%A7%C3%A3o-do-prompt-template)
      * [8.9. Construção da Cadeia RAG](https://www.google.com/search?q=%2389-constru%C3%A7%C3%A3o-da-cadeia-rag)
      * [8.10. Loop de Interação](https://www.google.com/search?q=%23810-loop-de-intera%C3%A7%C3%A3o)
  * [9. Próximos Passos e Possíveis Melhorias](https://www.google.com/search?q=%239-pr%C3%B3ximos-passos-e-poss%C3%ADveis-melhorias)
  * [10. Contribuições](https://www.google.com/search?q=%2310-contribui%C3%A7%C3%B5es)
  * [11. Licença](https://www.google.com/search?q=%2311-licen%C3%A7a)

-----

## 1\. Visão Geral do Projeto

Este projeto consiste em um protótipo para análise de logs de rede utilizando a arquitetura RAG (Retrieval-Augmented Generation) e o modelo de linguagem grande (LLM) Mistral AI. O objetivo principal é demonstrar como combinar a capacidade de um LLM de gerar texto coerente com a habilidade de recuperar informações específicas de uma base de conhecimento (neste caso, logs de rede), para fornecer respostas precisas e contextuais sobre eventos de rede.

Em ambientes de segurança e operações de rede, a quantidade de logs gerada é massiva. Analisá-los manualmente é inviável e ferramentas tradicionais podem não ser flexíveis o suficiente para responder a perguntas complexas ou inesperadas. Este projeto explora o RAG como uma solução inovadora para extrair insights valiosos e automatizar a análise de logs de rede de forma eficiente.

## 2\. O Que É RAG (Retrieval-Augmented Generation)?

RAG, ou Geração Aumentada por Recuperação, é uma técnica que aprimora a capacidade dos Grandes Modelos de Linguagem (LLMs) de gerar respostas informadas e factuais. Ao contrário dos LLMs tradicionais que respondem apenas com base em seus dados de treinamento internos, o RAG permite que o modelo:

1.  **Recupere (Retrieve):** Busque informações relevantes em uma base de conhecimento externa (documentos, bancos de dados, logs, etc.) que não foram incluídas em seu treinamento original.
2.  **Aumente (Augment):** Incorpore essas informações recuperadas ao prompt de entrada do LLM.
3.  **Gere (Generate):** Use tanto o conhecimento interno do LLM quanto o contexto fornecido pelos dados recuperados para produzir uma resposta mais precisa, atualizada e baseada em fatos.

No contexto da análise de logs de rede, isso significa que o Mistral AI não "alucinará" respostas ou dependerá apenas de conhecimentos gerais sobre rede, mas sim consultará os logs reais para extrair os detalhes específicos necessários para a consulta do usuário.

## 3\. Objetivos Deste Protótipo

Este protótipo tem como objetivo principal validar a viabilidade da aplicação de RAG com Mistral AI para análise de logs de rede. Especificamente, buscamos:

  * Demonstrar o fluxo completo de um pipeline RAG.
  * Processar e indexar logs de rede fictícios.
  * Permitir que usuários façam perguntas em linguagem natural sobre os logs.
  * Obter respostas relevantes e contextuais geradas pelo Mistral AI com base nos logs.
  * Servir como base para futuras expansões e testes em um ambiente de TCC.

## 4\. Arquitetura do Sistema

O sistema segue a arquitetura padrão de RAG, adaptada para logs de rede:

1.  **Logs de Rede (`network_logs.txt`):** Dados brutos, neste protótipo, um arquivo de texto simples com entradas de logs fictícias.
2.  **Módulo de Carregamento:** Carrega o conteúdo do arquivo de logs.
3.  **Chunking (Divisão em Pedaços):** Divide as entradas de logs em pedaços menores (chunks) para gerenciar o contexto e otimizar a busca.
4.  **Geração de Embeddings:** Converte cada chunk de texto em um vetor numérico de alta dimensão (embedding), que semanticamente representa o conteúdo do chunk.
5.  **Banco de Vetores (ChromaDB):** Armazena os embeddings e seus chunks de texto correspondentes, permitindo buscas rápidas por similaridade.
6.  **Módulo de Recuperação (Retriever):** Dada uma consulta do usuário, o retriever busca no banco de vetores os chunks de logs mais semanticamente relevantes.
7.  **Modelo de Linguagem (Mistral AI):** Recebe a consulta original do usuário e os chunks de logs recuperados como contexto. Ele então gera uma resposta em linguagem natural.
8.  **Interface de Usuário (Terminal):** O usuário interage com o sistema via linha de comando, fazendo perguntas e recebendo as respostas.

<!-- end list -->

```
+-------------------+      +-------------------+      +-------------------+
|   Logs de Rede    |      |   Pré-processamento   |      |   Geração de      |
| (`network_logs.txt`)| ---> |  (Chunking, Parsing)  | ---> |    Embeddings     |
+-------------------+      +-------------------+      +-------------------+
        |                                                              |
        |                                                              v
        |                    +-------------------+      +-------------------+
        |                    |   Banco de Vetores    | <--- |   (ChromaDB)      |
        |                    |      (Indexação)      |      |                   |
        |                    +-------------------+      +-------------------+
        |                                 ^
        |                                 |
        |                                 |
        |  +---------------------------+  |
        |  |  Módulo de Recuperação    |  |
        +--|     (Retriever)         |<----+ (Busca por Similaridade)
           +---------------------------+  |
                    | (Contexto)           |
                    |                      |
      +-------------+------------+         |
      |                            |         |
      |   Consulta do Usuário      | --------+
      | (Linguagem Natural)      |
      +----------------------------+
                    |
                    v
           +---------------------+      +-------------------+
           |    Mistral AI       | ---> | Resposta Gerada   |
           |    (Gerador)        |      | (Linguagem Natural)|
           +---------------------+      +-------------------+
```

## 5\. Tecnologias Utilizadas

Este projeto utiliza uma combinação de bibliotecas e serviços Python para construir o pipeline RAG.

### 5.1. Python

  * **Porquê:** Linguagem de programação amplamente utilizada em Machine Learning e desenvolvimento de IA, com um vasto ecossistema de bibliotecas.
  * **Função no Projeto:** Orquestra todo o pipeline, desde o carregamento de dados até a interação com o LLM e a apresentação da resposta.

### 5.2. Mistral AI

  * **Porquê:** Uma das empresas líderes em modelos de linguagem grandes, conhecida por modelos eficientes e de alto desempenho. O Mistral AI oferece uma API robusta e um *free tier* que facilita a experimentação.
  * **Função no Projeto:** Atua como o **Gerador** no pipeline RAG. Ele recebe a pergunta do usuário e os trechos de logs relevantes, e é responsável por sintetizar essas informações em uma resposta coerente e útil em linguagem natural. Utilizamos a API `mistral-small-latest` para este protótipo, que oferece um bom equilíbrio entre desempenho e custo.
  * **Acesso:** O acesso à IA do Mistral é feito remotamente, através de sua API. Seu código Python envia requisições HTTP para os servidores da Mistral AI, onde o modelo é executado e a resposta é gerada e enviada de volta.

### 5.3. LangChain

  * **Porquê:** Um framework de orquestração de LLMs que simplifica a construção de aplicações complexas baseadas em LLMs, como RAG. Ele abstrai muitas das complexidades de integração e encadeamento de componentes.
  * **Função no Projeto:** Facilita a interconexão de todos os módulos: o carregador de documentos (`TextLoader`), o divisor de texto (`RecursiveCharacterTextSplitter`), os embeddings (`HuggingFaceEmbeddings`), o banco de vetores (`Chroma`), o retriever e a integração com o Mistral AI. Ele também ajuda na criação e formatação de prompts.

### 5.4. `sentence-transformers` (via `langchain-huggingface`)

  * **Porquê:** Fornece acesso a modelos de embedding pré-treinados, que são excelentes para transformar texto em representações numéricas (vetores) que capturam o significado semântico.
  * **Função no Projeto:** Utiliza o modelo `all-MiniLM-L6-v2` para gerar os embeddings dos chunks de logs e da consulta do usuário. Esses embeddings são essenciais para que o banco de vetores possa encontrar chunks de logs semanticamente similares à pergunta.

### 5.5. ChromaDB (via `langchain-community`)

  * **Porquê:** Um banco de vetores leve e fácil de usar, ideal para prototipagem e desenvolvimento local. Ele não requer uma configuração complexa e se integra nativamente com o LangChain.
  * **Função no Projeto:** Armazena os embeddings gerados a partir dos logs de rede. Quando uma consulta é feita, o ChromaDB é consultado para encontrar os vetores (e, consequentemente, os chunks de log) mais próximos (mais semanticamente similares) ao embedding da consulta.

### 5.6. `python-dotenv`

  * **Porquê:** Ajuda a carregar variáveis de ambiente de um arquivo `.env`, mantendo informações sensíveis (como chaves de API) fora do código-fonte principal.
  * **Função no Projeto:** Carrega a `MISTRAL_API_KEY` do arquivo `.env` para que o script possa autenticar as requisições à API do Mistral AI de forma segura.

## 6\. Configuração e Instalação

Para configurar e executar o projeto em seu ambiente local, siga os passos abaixo.

### 6.1. Pré-requisitos

  * **Python 3.8+:** Certifique-se de ter uma versão compatível do Python instalada.
  * **Chave de API do Mistral AI:** Obtenha uma chave de API gratuita ou paga registrando-se em [La Plateforme Mistral AI](https://console.mistral.ai/api-keys).

### 6.2. Clonar o Repositório

Primeiro, clone este repositório para sua máquina local (se você já tiver o código, pule esta etapa):

```bash
git clone <URL_DO_SEU_REPOSITORIO_GITHUB>
cd <nome_da_pasta_do_projeto> # ex: cd rag_analise_logs
```

### 6.3. Configurar a Chave de API do Mistral AI

Crie um arquivo chamado `.env` na raiz do diretório do projeto (o mesmo diretório onde `rag_mistral_logs.py` está). Adicione sua chave de API nele no seguinte formato:

```
MISTRAL_API_KEY="SUA_CHAVE_AQUI"
```

**Importante:** Substitua `"SUA_CHAVE_AQUI"` pela chave de API real que você obteve do Mistral AI. Não inclua a chave diretamente no código-fonte por motivos de segurança.

### 6.4. Instalar Dependências

Instale todas as bibliotecas Python necessárias usando `pip`:

```bash
pip install langchain langchain-community langchain-mistralai langchain-huggingface sentence-transformers chromadb python-dotenv
```

## 7\. Como Executar o Protótipo

### 7.1. Estrutura dos Logs de Rede

O protótipo utiliza um arquivo de texto simples chamado `network_logs.txt` para simular logs de rede. Crie este arquivo no mesmo diretório do seu script Python (`rag_mistral_logs.py`) e preencha-o com logs fictícios. Um exemplo de log maior e mais diversificado é fornecido abaixo para melhor teste:

```
# Conteúdo de network_logs.txt
2023-10-26 10:00:01 INFO User 'admin' logged in from 192.166.1.10. Session ID: A1B2C3D4.
2023-10-26 10:00:05 WARN Failed login attempt for user 'guest' from 192.166.1.20. Reason: Invalid password.
2023-10-26 10:01:10 ERROR Network connection dropped to 10.0.0.5 on port 80. Protocol: TCP.
2023-10-26 10:02:00 INFO Traffic spike detected from 203.0.113.40 to 192.166.1.100. Bytes transferred: 50MB.
2023-10-26 10:03:15 ALERT Potential DDoS attack from multiple IPs to 192.166.1.100. Source IPs: 1.1.1.1, 2.2.2.2, 3.3.3.3.
# ... (adicione o restante do log maior aqui) ...
```

**(Cole aqui o conteúdo completo do `network_logs_maior.txt` que te forneci anteriormente)**

### 7.2. Executando o Script

Com todas as dependências instaladas e o `.env` configurado, você pode executar o script Python:

```bash
python3 rag_mistral_logs.py
```

### 7.3. Exemplo de Interação

O script inicializará o pipeline RAG e então entrará em um loop, solicitando que você digite perguntas.

```
1. Carregando logs de rede...
Total de documentos carregados: 1
2. Dividindo documentos em chunks...
Total de chunks criados: X (número de chunks dependerá do tamanho total do log)
3. Criando embeddings dos chunks...
4. Armazenando embeddings no ChromaDB...
Embeddings armazenados no ChromaDB.
5. Configurando o retriever...
Retriever configurado.
6. Configurando o Mistral AI...
7. Criando o prompt template...
Prompt template criado.
8. Construindo a cadeia RAG...
Cadeia RAG construída. Agora você pode fazer perguntas!

Faça uma pergunta sobre os logs (ou 'sair' para encerrar): Qual o status de disk usage no servidor 'web-01'?

Processando sua pergunta: 'Qual o status de disk usage no servidor 'web-01'?'...

Resposta do Mistral AI:
Com base nos logs, o uso de disco no servidor 'web-01' atingiu 95% no caminho /var/log. É necessária uma ação imediata.

Faça uma pergunta sobre os logs (ou 'sair' para encerrar): Algum ataque DDoS foi detectado?

Processando sua pergunta: 'Algum ataque DDoS foi detectado?'...

Resposta do Mistral AI:
Sim, um alerta de potencial ataque DDoS foi detectado contra 192.166.1.100, com IPs de origem 1.1.1.1, 2.2.2.2 e 3.3.3.3.

Faça uma pergunta sobre os logs (ou 'sair' para encerrar): sair
```

## 8\. Detalhamento do Código (`rag_mistral_logs.py`)

Este script Python implementa o pipeline RAG passo a passo.

```python
import os
from dotenv import load_dotenv

# Importações dos módulos LangChain
from langchain_community.document_loaders import TextLoader # Para carregar o TXT
from langchain.text_splitter import RecursiveCharacterTextSplitter # Para dividir em chunks
from langchain_huggingface.embeddings import HuggingFaceEmbeddings # Para gerar embeddings
from langchain_community.vectorstores import Chroma # O banco de vetores
from langchain_core.prompts import ChatPromptTemplate # Para criar o prompt para o LLM
from langchain_core.output_parsers import StrOutputParser # Para formatar a saída do LLM
from langchain_core.runnables import RunnablePassthrough # Para passagem de dados na cadeia
from langchain_mistralai.chat_models import ChatMistralAI # Para interagir com o Mistral AI

# 8.1. Carregamento das Variáveis de Ambiente
# Carrega a MISTRAL_API_KEY do arquivo .env. Isso mantém a chave segura e fora do código-fonte.
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# Verifica se a chave foi carregada. Se não, levanta um erro.
if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY não encontrada. Crie um arquivo .env com sua chave.")

# --- 8.2. Carregamento dos Logs de Rede ---
# Usa TextLoader do LangChain para ler o arquivo de logs.
# Cada linha ou conjunto de linhas é tratado como um "documento".
print("1. Carregando logs de rede...")
loader = TextLoader("network_logs.txt")
documents = loader.load()
print(f"Total de documentos carregados: {len(documents)}")

# --- 8.3. Divisão dos Documentos em Chunks ---
# Divide os documentos maiores em pedaços menores (chunks).
# Isso é crucial porque os LLMs têm um limite de tokens de entrada (context window).
# `chunk_size`: tamanho máximo de cada pedaço em caracteres.
# `chunk_overlap`: sobreposição entre pedaços consecutivos para manter o contexto.
print("2. Dividindo documentos em chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)
print(f"Total de chunks criados: {len(chunks)}")

# --- 8.4. Geração de Embeddings ---
# Converte cada chunk de texto em um vetor numérico (embedding).
# O modelo "sentence-transformers/all-MiniLM-L6-v2" é um modelo de embedding leve e eficiente.
# Os embeddings são usados para medir a similaridade semântica entre textos.
print("3. Criando embeddings dos chunks...")
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- 8.5. Armazenamento em um Banco de Vetores (ChromaDB) ---
# O ChromaDB armazena os chunks e seus embeddings.
# Ele permite buscar rapidamente por chunks semanticamente semelhantes a uma consulta.
# Neste protótipo, o ChromaDB é iniciado em memória ou em um diretório temporário.
print("4. Armazenando embeddings no ChromaDB...")
vectorstore = Chroma.from_documents(chunks, embeddings_model)
print("Embeddings armazenados no ChromaDB.")

# --- 8.6. Configuração do Retriever ---
# O retriever é a interface que consulta o banco de vetores.
# `search_kwargs={"k": 3}`: instrui o retriever a retornar os 3 chunks mais relevantes
# para a consulta do usuário.
print("5. Configurando o retriever...")
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("Retriever configurado.")

# --- 8.7. Configuração do Modelo Mistral AI ---
# Inicializa o modelo de chat do Mistral AI usando a chave de API fornecida.
# 'mistral-small-latest' é o modelo escolhido para este protótipo,
# oferecendo bom desempenho e eficiência.
print("6. Configurando o Mistral AI...")
llm = ChatMistralAI(api_key=MISTRAL_API_KEY, model="mistral-small-latest")
print("Mistral AI configurado.")

# --- 8.8. Criação do Prompt Template ---
# Define o formato do prompt que será enviado ao Mistral AI.
# O `{context}` será preenchido com os chunks de logs recuperados.
# O `{question}` será a pergunta original do usuário.
# As instruções ("Você é um assistente útil...", "Use os seguintes trechos...")
# guiam o LLM a como usar o contexto e responder.
print("7. Criando o prompt template...")
template = """
Você é um assistente útil para análise de logs de rede.
Use os seguintes trechos de logs de rede recuperados para responder à pergunta.
Se você não souber a resposta, apenas diga que não tem informações suficientes nos logs.
Mantenha a resposta concisa e relevante para o contexto dos logs.

Contexto dos Logs:
{context}

Pergunta: {question}

Resposta:
"""
prompt = ChatPromptTemplate.from_template(template)
print("Prompt template criado.")

# --- 8.9. Construção da Cadeia RAG ---
# A cadeia RAG orquestra o fluxo de dados:
# 1. A pergunta do usuário (RunnablePassthrough) é passada.
# 2. O retriever usa a pergunta para buscar o contexto relevante dos logs.
# 3. O prompt é preenchido com a pergunta e o contexto.
# 4. O prompt completo é enviado ao LLM (Mistral AI).
# 5. A resposta do LLM é convertida para uma string simples (StrOutputParser).
print("8. Construindo a cadeia RAG...")
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
print("Cadeia RAG construída. Agora você pode fazer perguntas!")

# --- 8.10. Loop de Interação ---
# Permite que o usuário faça múltiplas perguntas interativamente.
while True:
    user_query = input("\nFaça uma pergunta sobre os logs (ou 'sair' para encerrar): ")
    if user_query.lower() == 'sair':
        break

    print(f"\nProcessando sua pergunta: '{user_query}'...")
    response = rag_chain.invoke(user_query)
    print(f"\nResposta do Mistral AI:\n{response}")
```

## 9\. Próximos Passos e Possíveis Melhorias

  * **Processamento de Logs Mais Robusto:**
      * Utilizar bibliotecas de parsing de logs (ex: `loguru`, `structlog`, ou custom parsers) para extrair campos estruturados (timestamps, IPs, níveis, IDs de eventos).
      * Tratar diferentes formatos de logs (JSON, Syslog, CSV).
      * Anonimização de dados sensíveis.
  * **Aumento do Volume de Dados:** Testar com datasets de logs maiores, o que pode exigir bancos de vetores mais escaláveis (ex: Pinecone, Weaviate, Qdrant, Milvus) e estratégias de chunking mais avançadas.
  * **Avaliação de Desempenho:**
      * Definir métricas para avaliar a qualidade das respostas do RAG (relevância, precisão, completude).
      * Realizar testes de benchmark com diferentes consultas.
  * **Interface de Usuário:** Desenvolver uma interface web (ex: Flask, FastAPI, Streamlit) para uma experiência de usuário mais amigável.
  * **Otimização de Prompt Engineering:** Experimentar diferentes prompts para extrair respostas mais específicas e relevantes do Mistral AI.
  * **Filtragem de Metadados:** Se os logs forem estruturados com metadados (timestamp, severidade, IP de origem), o banco de vetores pode ser usado para filtrar resultados antes da busca por similaridade.
  * **Uso de Contexto Histórico:** Implementar uma forma de o LLM lembrar do histórico da conversa para perguntas de acompanhamento.
  * **Modelos Locais:** Para fins de pesquisa ou se houver restrições de custo/privacidade, explorar a execução de modelos open-source Mistral localmente (ex: com Ollama, vLLM) se o hardware permitir.
  * **Detecção de Anomalias:** Integrar o RAG com módulos de detecção de anomalias para identificar padrões incomuns nos logs e alertar o usuário.

## 10\. Contribuições

Contribuições são bem-vindas\! Sinta-se à vontade para abrir issues para reportar bugs, sugerir melhorias ou enviar pull requests.

