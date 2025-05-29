import streamlit as st
import requests

# Configurar layout amplio
st.set_page_config(layout="wide")

# Estilos CSS personalizados
st.markdown("""
    <style>
        .scrollable-container {
            height: 85vh;
            overflow-y: auto;
            padding-right: 20px;
            border-left: 1px solid #ddd;
            margin-top: 1rem;
        }
        .stButton button {
            width: 100%;
        }
    </style>
""", unsafe_allow_html=True)

# URL de tu backend en Railway
base_url = "https://fastapi-agent-tool-langchain-production.up.railway.app"

def main():
    st.title("Agente Comparador de Contratos - Lawgent")

    left_col, right_col = st.columns(2)

    with left_col:
        pdf1 = st.file_uploader("Upload Contract 1", type=['pdf'])
        pdf2 = st.file_uploader("Upload Contract 2", type=['pdf'])

        explain_with_llm = not st.checkbox(
            "Mostrar diferencias técnicas (diff)",
            value=False,
            help="Por defecto verás una explicación legal en lenguaje sencillo. Marca para ver el diff técnico."
        )

        compare_clicked = st.button("Comparar PDFs")

    with right_col:
        if compare_clicked and pdf1 is not None and pdf2 is not None:
            st.warning("""
                ⚠️ **AVISO IMPORTANTE**: Este sistema opera con una precisión predictiva que puede ser inferior al 99%. 
                El usuario es totalmente responsable de verificar la veracidad y exactitud de toda la información proporcionada. 
                Los resultados deben ser validados manualmente antes de tomar cualquier decisión legal o contractual.
            """)

            files = {
                'pdf1': ('documento1.pdf', pdf1.getvalue(), 'application/pdf'),
                'pdf2': ('documento2.pdf', pdf2.getvalue(), 'application/pdf')
            }

            params = {'explain_with_llm': str(explain_with_llm).lower()}

            try:
                with st.spinner("Comparando documentos..."):
                    res = requests.post(f"{base_url}/compare-pdfs-diff", files=files, params=params)
                    res.raise_for_status()
                    result = res.json()

                st.success("¡Comparación completada!")

                if explain_with_llm:
                    st.subheader("🧾 Explicación de las diferencias:")
                    st.write(result.get("explanation", "Sin explicación disponible."))
                else:
                    st.subheader("🔍 Diferencias técnicas (diff):")
                    diff_lines = result.get("diff_lines", [])
                    if diff_lines:
                        st.code("\n".join(diff_lines), language="diff")
                    else:
                        st.info("No se encontraron diferencias técnicas.")

                    st.subheader("📊 Cambios detectados:")
                    st.json(result.get("changes", []))

                # Mostrar interpretación legal
                summary = result.get("summary", {})
                if "interpretation" in summary:
                    st.info(f"🧠 Interpretación: {summary['interpretation']}")
                else:
                    st.warning("No se pudo interpretar el resultado.")

            except Exception as e:
                st.error(f"Error durante la comparación: {str(e)}")

        else:
            st.markdown("""
                ### 👋 Bienvenido al Agente Comparador de Contratos - Lawgent

                #### Instrucciones:
                1. Sube dos contratos en formato PDF en la columna izquierda
                2. Decide si quieres ver las diferencias técnicas o explicaciones legales
                3. Haz clic en "Comparar PDFs" para ver los resultados

                ---
                ⚠️ **AVISO**: Este sistema puede tener una precisión menor al 99%. Verifica siempre los resultados antes de usarlos legalmente.
            """)

if __name__ == "__main__":
    main()
