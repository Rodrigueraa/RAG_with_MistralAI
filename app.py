import os
import streamlit as st
from dotenv import load_dotenv

# Importações dos módulos LangChain
# TextLoader: Usado para carregar dados de arquivos de texto simples.
from langchain_community.document_loaders import TextLoader
# RecursiveCharacterTextSplitter: Divide documentos grandes em pedaços menores (chunks).
from langchain.text_splitter import RecursiveCharacterTextSplitter
# HuggingFaceEmbeddings: Gera embeddings (representações vetoriais numéricas) de texto.
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
# Chroma: Um banco de vetores leve e integrado que armazena os embeddings.
from langchain_community.vectorstores import Chroma
# ChatPromptTemplate: Ajuda a construir prompts estruturados para modelos de chat.
from langchain_core.prompts import ChatPromptTemplate
# StrOutputParser: Extrai a string de texto da resposta do modelo de linguagem.
from langchain_core.output_parsers import StrOutputParser
# RunnablePassthrough: Permite que uma entrada seja passada diretamente para o próximo passo.
from langchain_core.runnables import RunnablePassthrough
# ChatMistralAI: Classe para interagir com os modelos de chat da API do Mistral AI.
from langchain_mistralai.chat_models import ChatMistralAI

# --- Configuração da Página Streamlit ---
st.set_page_config(
    page_title="Análise de Logs de Rede com RAG e Mistral AI",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title("🔍 Análise de Logs de Rede com RAG e Mistral AI")
st.markdown(
    """
    Faça uma pergunta sobre os logs abaixo e obtenha insights instantâneos!
    """
)

# --- Carregamento de Variáveis de Ambiente ---
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    st.error("MISTRAL_API_KEY não encontrada. Crie um arquivo .env com sua chave na raiz do projeto.")
    st.stop() # Interrompe a execução do Streamlit se a chave não estiver configurada

# --- Função para Configurar o Pipeline RAG (Cacheada) ---
# O decorador st.cache_resource garante que esta função seja executada apenas uma vez
# quando o aplicativo Streamlit é iniciado, e seus resultados são armazenados em cache.
# Isso evita reprocessar os logs e recarregar o modelo a cada interação do usuário.
@st.cache_resource
def setup_rag_pipeline():
    """
    Configura e retorna a cadeia RAG completa.
    Esta função é cacheada para otimizar o desempenho.
    """
    st.spinner("Carregando e processando logs... Isso pode levar alguns segundos.")

    # 1. Carregamento dos Logs de Rede
    # Usa TextLoader para ler o arquivo de logs.
    loader = TextLoader("./logs/network_logs.txt")
    documents = loader.load()
#    st.success(f"Logs de rede carregados: {len(documents)} documento(s).")

    # 2. Divisão dos Documentos em Chunks
    # Divide os documentos maiores em pedaços menores (chunks).
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
#    st.success(f"Chunks criados: {len(chunks)} pedaços de log.")

    # 3. Geração de Embeddings
    # Gera embeddings para cada chunk usando um modelo pré-treinado.
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#    st.success("Modelo de embeddings carregado.")

    # 4. Armazenamento em um Banco de Vetores (ChromaDB)
    # Cria e popula um banco de vetores ChromaDB com os chunks e embeddings.
    # O ChromaDB é volátil por padrão aqui, mas pode ser persistente se configurado.
    vectorstore = Chroma.from_documents(chunks, embeddings_model)
#    st.success("Embeddings armazenados no ChromaDB.")

    # 5. Configuração do Retriever
    # Configura o retriever para buscar os 3 chunks mais relevantes.
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
#    st.success("Retriever configurado.")

    # 6. Configuração do Modelo Mistral AI
    # Inicializa o modelo de chat do Mistral AI.
    llm = ChatMistralAI(api_key=MISTRAL_API_KEY, model="mistral-small-latest")
#    st.success("Mistral AI configurado.")

    # 7. Criação do Prompt Template
    # Define o formato do prompt que será enviado ao Mistral AI.
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
#    st.success("Prompt template criado.")

    # 8. Construção da Cadeia RAG
    # A cadeia RAG orquestra o fluxo de dados: retriever -> prompt -> llm -> parser.
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
#    st.success("Cadeia RAG construída com sucesso!")
    return rag_chain

# --- Inicializa o pipeline RAG ---
# Esta chamada aciona a função setup_rag_pipeline() apenas uma vez.
rag_chain = setup_rag_pipeline()

# --- Interface de Interação ---
st.subheader("Faça sua pergunta sobre os logs:")

# st.text_input cria uma caixa de texto para a entrada do usuário.
user_query = st.text_input(
    "Digite sua pergunta aqui:",
    placeholder="Ex: Houve algum ataque DDoS recente?"
)

# st.button cria um botão para enviar a pergunta.
if st.button("Analisar Logs"):
    if user_query:
        with st.spinner("Processando sua pergunta e consultando os logs..."):
            # Invoca a cadeia RAG com a pergunta do usuário.
            response = rag_chain.invoke(user_query)
            st.markdown("---")
            st.subheader("Resposta do Mistral AI:")
            st.info(response) # Exibe a resposta em um bloco de informação
    else:
        st.warning("Por favor, digite uma pergunta para analisar.")

st.markdown("---")
st.markdown("Desenvolvido por Rodrigo Cauã para TCC de Análise de Logs de Rede com RAG e Mistral AI.")
