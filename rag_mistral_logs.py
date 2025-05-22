import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_mistralai.chat_models import ChatMistralAI # Importação específica para o MistralAI

# Carregar variáveis de ambiente do .env
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY não encontrada. Crie um arquivo .env com sua chave.")

# --- 1. Carregar os logs de rede ---
print("1. Carregando logs de rede...")
# Usamos TextLoader para ler o arquivo .txt
loader = TextLoader("network_logs.txt")
documents = loader.load()
print(f"Total de documentos carregados: {len(documents)}")

# --- 2. Dividir os documentos em chunks (pedaços) ---
print("2. Dividindo documentos em chunks...")
# RecursiveCharacterTextSplitter é bom para manter o contexto
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Tamanho máximo de cada pedaço
    chunk_overlap=50 # Overlap para manter o contexto entre pedaços
)
chunks = text_splitter.split_documents(documents)
print(f"Total de chunks criados: {len(chunks)}")
# print(f"Exemplo de chunk: {chunks[0].page_content[:100]}...")

# --- 3. Criar Embeddings ---
print("3. Criando embeddings dos chunks...")
# Usamos um modelo de embedding do Hugging Face. all-MiniLM-L6-v2 é leve e eficaz.
# Certifique-se de que o modelo seja baixado se não estiver em cache.
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- 4. Armazenar em um Vector Database (ChromaDB) ---
print("4. Armazenando embeddings no ChromaDB...")
# O ChromaDB é um banco de vetores leve que roda localmente
vectorstore = Chroma.from_documents(chunks, embeddings_model)
print("Embeddings armazenados no ChromaDB.")

# --- 5. Configurar o Retriever ---
# O retriever será responsável por buscar os chunks mais relevantes
print("5. Configurando o retriever...")
retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Recupera os 3 chunks mais relevantes
print("Retriever configurado.")

# --- 6. Configurar o Mistral AI para Geração de Respostas ---
print("6. Configurando o Mistral AI...")
# Inicializa o modelo de chat do Mistral AI
# Você pode escolher o modelo, 'mistral-large-latest' ou 'mistral-small-latest' são boas opções
llm = ChatMistralAI(api_key=MISTRAL_API_KEY, model="mistral-small-latest") # mistral-small-latest é geralmente mais rápido e econômico para protótipos

# --- 7. Criar o Prompt Template para o RAG ---
# Este template instrui o Mistral a usar as informações recuperadas
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

# --- 8. Construir a Cadeia RAG com LangChain ---
print("8. Construindo a cadeia RAG...")
# A cadeia RAG orquestra o fluxo:
# 1. Recuperar chunks relevantes
# 2. Formatar o prompt com os chunks recuperados e a pergunta
# 3. Enviar para o LLM (Mistral AI)
# 4. Parsear a saída do LLM
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
print("Cadeia RAG construída. Agora você pode fazer perguntas!")

# --- 9. Fazer Perguntas ---
while True:
    user_query = input("\nFaça uma pergunta sobre os logs (ou 'sair' para encerrar): ")
    if user_query.lower() == 'sair':
        break

    print(f"\nProcessando sua pergunta: '{user_query}'...")
    response = rag_chain.invoke(user_query)
    print(f"\nResposta do Mistral AI:\n{response}")

    # Opcional: Mostrar os documentos fonte que foram usados
    # Para isso, precisaríamos ajustar a cadeia para retornar também os source_documents.
    # Por simplicidade neste protótipo, estamos focando na resposta.
    # Se quiser ver os docs fonte, a cadeia precisaria ser algo como RetrievalQA.from_chain_type
    # que já retorna os source_documents.
    # Exemplo:
    # from langchain.chains import RetrievalQA
    # qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    # result = qa_chain.invoke({"query": user_query})
    # print(f"Documentos fonte utilizados: {[doc.page_content for doc in result['source_documents']]}")
