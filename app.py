import html as html_lib
import os

import streamlit as st

from rag_engine import analyze_contract

st.set_page_config(page_title="Analisador de Contratos", layout="wide")

st.title("Analisador de Contratos")

with st.sidebar:
    st.header("Configuracao")
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="Ou defina OPENAI_API_KEY no arquivo .env"
    )
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

SECTION_LABELS = {
    "comprador": "Comprador",
    "vendedor": "Vendedor / Incorporadora",
    "imovel": "Imovel",
    "condicoes_financeiras": "Condicoes Financeiras"
}

FIELD_LABELS = {
    "nome_completo":         "Nome Completo",
    "cpf":                   "CPF",
    "rg":                    "RG",
    "nacionalidade":         "Nacionalidade",
    "estado_civil":          "Estado Civil",
    "conjuge":               "Conjuge",
    "profissao":             "Profissao",
    "endereco":              "Endereco",
    "email":                 "E-mail",
    "cnpj":                  "CNPJ",
    "endereco_sede":         "Endereco da Sede",
    "representantes_legais": "Representantes Legais",
    "endereco_completo":     "Endereco Completo",
    "matricula":             "Matricula",
    "cartorio_registro":     "Cartorio de Registro",
    "inscricao_iptu":        "Inscricao IPTU",
    "area_total_m2":         "Area Total (m2)",
    "areas_lazer":           "Areas de Lazer",
    "comodos":               "Comodos",
    "vagas_garagem":         "Vagas de Garagem",
    "preco_total":           "Preco Total",
    "indice_reajuste":       "Indice de Reajuste",
    "periodicidade_reajuste":"Periodicidade do Reajuste",
    "juros_mora":            "Juros de Mora",
    "multa_atraso":          "Multa por Atraso",
    "comissao_corretagem":   "Comissao de Corretagem"
}


def render_value(v) -> str:
    if isinstance(v, dict):
        parts = [f"{k}: {val}" for k, val in v.items() if val]
        return ", ".join(parts) if parts else ""
    if isinstance(v, list):
        return "; ".join(str(x) for x in v) if v else ""
    return str(v) if v else ""


def get_highlighted_window(value: str, trecho_text: str, context_chars: int = 200) -> str:
    if not value or not trecho_text:
        escaped = html_lib.escape(trecho_text[:400] if trecho_text else "")
        return f'<pre style="font-family:monospace;font-size:0.85em;white-space:pre-wrap;margin:0">{escaped}</pre>'

    text_lower = trecho_text.lower()
    search_term = value.strip()
    pos = text_lower.find(search_term.lower())

    if pos == -1:
        tokens = sorted(
            [t for t in search_term.split() if len(t) > 4 or any(c.isdigit() for c in t)],
            key=len, reverse=True
        )
        for token in tokens:
            p = text_lower.find(token.lower())
            if p != -1:
                pos = p
                search_term = token
                break

    if pos == -1:
        escaped = html_lib.escape(trecho_text[:400])
        return f'<pre style="font-family:monospace;font-size:0.85em;white-space:pre-wrap;margin:0">{escaped}</pre>'

    win_start = max(0, pos - context_chars)
    win_end = min(len(trecho_text), pos + len(search_term) + context_chars)
    window = trecho_text[win_start:win_end]

    prefix = "..." if win_start > 0 else ""
    suffix = "..." if win_end < len(trecho_text) else ""

    match_pos = window.lower().find(search_term.lower())
    if match_pos != -1:
        before  = html_lib.escape(window[:match_pos])
        matched = html_lib.escape(window[match_pos:match_pos + len(search_term)])
        after   = html_lib.escape(window[match_pos + len(search_term):])
        content = (
            f"{prefix}{before}"
            f'<mark style="background:#fef08a;padding:1px 4px;border-radius:3px;font-weight:700">'
            f"{matched}</mark>"
            f"{after}{suffix}"
        )
    else:
        content = html_lib.escape(window)

    return (
        f'<pre style="font-family:monospace;font-size:0.85em;'
        f'white-space:pre-wrap;line-height:1.6;margin:0">{content}</pre>'
    )


def render_section(dados: dict, trechos_por_campo: dict):
    header_col1, header_col2 = st.columns([1, 2])
    header_col1.markdown("**Campo**")
    header_col2.markdown("**Valor**")
    st.divider()

    for field, value in dados.items():
        label    = FIELD_LABELS.get(field, field)
        rendered = render_value(value)

        col1, col2 = st.columns([1, 2])
        col1.write(label)

        if rendered:
            col2.write(rendered)
            field_trechos = trechos_por_campo.get(field, [])
            if field_trechos:
                with col2.expander(f"Ver trecho ({len(field_trechos)} encontrado(s))"):
                    for i, t in enumerate(field_trechos, 1):
                        st.markdown(
                            f"**Trecho {i}** — Pagina `{t['pagina']}` · posicao `{t['posicao']}`"
                        )
                        highlighted = get_highlighted_window(rendered, t["trecho"])
                        st.markdown(highlighted, unsafe_allow_html=True)
                        if i < len(field_trechos):
                            st.divider()
        else:
            col2.markdown("*Nao encontrado*")

        st.divider()


uploaded_file = st.file_uploader("Selecione o contrato em PDF", type="pdf")

if uploaded_file:
    st.info(f"Arquivo carregado: **{uploaded_file.name}** ({uploaded_file.size // 1024} KB)")

    if st.button("Analisar Contrato", type="primary", use_container_width=True):
        if not os.getenv("OPENAI_API_KEY"):
            st.error("Informe a OpenAI API Key na barra lateral.")
        else:
            pdf_bytes = uploaded_file.read()
            progress_bar = st.progress(0, text="Iniciando analise...")
            status = st.empty()

            def on_progress(current, total, secao):
                pct = int((current / total) * 100)
                label = SECTION_LABELS.get(secao, secao)
                progress_bar.progress(pct, text=f"Analisando: {label}...")
                if secao != "concluido":
                    status.text(f"Secao {current + 1} de {total}: {label}")

            try:
                resultado = analyze_contract(pdf_bytes, progress_callback=on_progress)
                progress_bar.progress(100, text="Analise concluida!")
                status.empty()

                st.success("Contrato analisado com sucesso!")
                st.divider()

                tab_labels = [SECTION_LABELS[k] for k in resultado]
                tabs = st.tabs(tab_labels)

                for tab, (secao, conteudo) in zip(tabs, resultado.items()):
                    with tab:
                        dados            = conteudo["dados"]
                        trechos_por_campo = conteudo.get("trechos_por_campo", {})

                        if isinstance(dados, dict):
                            render_section(dados, trechos_por_campo)
                        else:
                            st.warning("Nao foi possivel estruturar o JSON. Resposta bruta:")
                            st.text(dados)

            except Exception as e:
                progress_bar.empty()
                status.empty()
                st.error(f"Erro na analise: {e}")
