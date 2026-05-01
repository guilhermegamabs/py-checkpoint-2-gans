import json
import os
import tempfile
from typing import List

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict

load_dotenv()

SCHEMA_CONTRATO = {
    "comprador": {
        "nome_completo": "",
        "cpf": "",
        "rg": "",
        "nacionalidade": "",
        "estado_civil": "",
        "conjuge": {"nome": "", "cpf": "", "rg": ""},
        "profissao": "",
        "endereco": "",
        "email": ""
    },
    "vendedor": {
        "nome_completo": "",
        "cnpj": "",
        "endereco_sede": "",
        "representantes_legais": []
    },
    "imovel": {
        "endereco_completo": "",
        "matricula": "",
        "cartorio_registro": "",
        "inscricao_iptu": "",
        "area_total_m2": "",
        "areas_lazer": "",
        "comodos": "",
        "vagas_garagem": ""
    },
    "condicoes_financeiras": {
        "preco_total": "",
        "indice_reajuste": "",
        "periodicidade_reajuste": "",
        "juros_mora": "",
        "multa_atraso": "",
        "comissao_corretagem": ""
    }
}

QUERIES = {
    "comprador": "nome completo CPF RG nacionalidade estado civil cônjuge profissão endereço email comprador",
    "vendedor": "vendedor incorporadora CNPJ endereço sede representantes legais",
    "imovel": "imóvel matrícula cartório registro IPTU área total lazer cômodos vagas garagem endereço",
    "condicoes_financeiras": "preço total valor venda índice reajuste INCC IGPM IPCA juros multa atraso comissão corretagem"
}

PROMPT = ChatPromptTemplate.from_messages([
    ("human", """Você é um especialista em análise de contratos jurídicos brasileiros.

    Analise os trechos do contrato abaixo e extraia SOMENTE as informações do papel solicitado na questão.
    
    REGRAS DE VALIDAÇÃO DE PAPEL:
    - Se a questão pede dados do COMPRADOR: extraia apenas campos explicitamente atribuídos ao COMPRADOR no texto. Ignore qualquer dado do VENDEDOR/INCORPORADORA.
    - Se a questão pede dados do VENDEDOR/INCORPORADORA: extraia apenas campos explicitamente atribuídos ao VENDEDOR ou INCORPORADORA. Ignore dados do COMPRADOR.
    - Se a questão pede dados do IMÓVEL: extraia apenas características e identificação do imóvel. Ignore dados das partes.
    - Se a questão pede condições financeiras: extraia apenas valores, índices e penalidades contratuais. Ignore dados pessoais.
    - Extraia APENAS o que estiver explicitamente no texto. Não infira nem complete com suposições.
    - Para campos não encontrados, use string vazia "" ou lista vazia [].
    - Retorne SOMENTE um objeto JSON válido. Sem explicações fora do JSON.
    
    Schema esperado:
    {schema}
    
    Question: {question}
    Context: {context}
    Answer:""")
])


def build_vector_store(pdf_bytes: bytes):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vector_store = InMemoryVectorStore(embeddings)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
    finally:
        os.unlink(tmp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    all_splits = splitter.split_documents(docs)
    vector_store.add_documents(all_splits)
    return vector_store, all_splits


def _doc_to_trecho(doc: Document) -> dict:
    return {
        "trecho":  doc.page_content,
        "posicao": doc.metadata.get("start_index", "?"),
        "pagina":  doc.metadata.get("page", "?")
    }


def _search_terms(value) -> list[str]:
    if isinstance(value, dict):
        texts = [str(v) for v in value.values() if v and str(v).strip()]
    elif isinstance(value, list):
        texts = [str(v) for v in value if v and str(v).strip()]
    else:
        texts = [str(value)] if value else []

    tokens = set()
    for text in texts:
        for token in text.split():
            if len(token) > 3 or any(c.isdigit() for c in token):
                tokens.add(token.lower())
    return list(tokens)


def find_source_chunks(value, all_splits: list, k: int = 2) -> list[Document]:
    terms = _search_terms(value)
    if not terms:
        return []

    scored = []
    for split in all_splits:
        content_lower = split.page_content.lower()
        hits = sum(1 for t in terms if t in content_lower)
        if hits > 0:
            scored.append((hits, split))

    if not scored:
        return []

    scored.sort(key=lambda x: x[0], reverse=True)
    max_hits = scored[0][0]
    threshold = max(1, max_hits - 1)
    return [s for hits, s in scored[:k] if hits >= threshold]


def analyze_contract(pdf_bytes: bytes, progress_callback=None) -> dict:
    llm = init_chat_model("gpt-4o-mini", model_provider="openai")
    vector_store, all_splits = build_vector_store(pdf_bytes)

    class State(TypedDict):
        question: str
        schema: str
        context: List[Document]
        answer: str

    def retrieve(state: State):
        return {"context": vector_store.similarity_search(state["question"], k=4)}

    def generate(state: State):
        docs_content = ""
        for i, doc in enumerate(state["context"], 1):
            pos = doc.metadata.get("start_index", "?")
            docs_content += f"[Trecho {i} — posição {pos} no documento]\n{doc.page_content}\n\n"

        messages = PROMPT.invoke({
            "question": state["question"],
            "context": docs_content,
            "schema": state["schema"]
        })
        return {"answer": llm.invoke(messages).content}

    graph = StateGraph(State).add_sequence([retrieve, generate])
    graph.add_edge(START, "retrieve")
    compiled = graph.compile()

    resultado_final = {}
    sections = list(QUERIES.items())

    for idx, (secao, query) in enumerate(sections):
        if progress_callback:
            progress_callback(idx, len(sections), secao)

        result = compiled.invoke({
            "question": f"Extraia as informações de '{secao}' do contrato: {query}",
            "schema": json.dumps(SCHEMA_CONTRATO[secao], ensure_ascii=False, indent=2)
        })

        raw = result["answer"].replace("```json", "").replace("```", "").strip()
        try:
            dados = json.loads(raw)
        except json.JSONDecodeError:
            dados = raw

        trechos_por_campo: dict = {}
        if isinstance(dados, dict):
            for field, field_value in dados.items():
                source_docs = find_source_chunks(field_value, all_splits, k=2)
                trechos_por_campo[field] = [_doc_to_trecho(d) for d in source_docs]

        resultado_final[secao] = {
            "dados": dados,
            "trechos_por_campo": trechos_por_campo
        }

    if progress_callback:
        progress_callback(len(sections), len(sections), "concluído")

    return resultado_final
