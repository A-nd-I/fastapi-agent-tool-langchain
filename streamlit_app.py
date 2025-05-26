import streamlit as st
import requests

def main():
    st.title("PDF Comparison Tool")

    pdf1 = st.file_uploader("Upload first PDF", type=['pdf'])
    pdf2 = st.file_uploader("Upload second PDF", type=['pdf'])

    # Elige diff como opción por defecto (index=2)
    comparison_method = st.radio(
        "Select comparison method:",
        (
            "Embeddings global + por secciones",
            "FAISS CPU vector search",
            "Diff palabra a palabra (precisión máxima)"
        ),
        index=2  # ← ¡Esta línea hace que Diff sea la opción por defecto!
    )

    base_url = "http://localhost:8000"
    endpoints = {
        "Embeddings global + por secciones": "/compare-pdfs",
        "FAISS CPU vector search": "/compare-pdfs-faiss",
        "Diff palabra a palabra (precisión máxima)": "/compare-pdfs-diff"
    }

    # Checkbox solo si diff está seleccionado
    explain_with_llm = True
    if comparison_method == "Diff palabra a palabra (precisión máxima)":
        explain_with_llm = st.checkbox(
            "Mostrar diferencias técnicas (diff)", value=False,
            help="Por defecto verás una explicación en lenguaje sencillo. Marca para ver el diff técnico."
        )

    if st.button("Compare PDFs") and pdf1 is not None and pdf2 is not None:
        files = {
            'pdf1': ('documento1.pdf', pdf1.getvalue(), 'application/pdf'),
            'pdf2': ('documento2.pdf', pdf2.getvalue(), 'application/pdf')
        }
        endpoint = endpoints[comparison_method]
        params = {}
        if comparison_method == "Diff palabra a palabra (precisión máxima)":
            # Invertido porque explain_with_llm es "¿quieres ver el diff técnico?"
            params['explain_with_llm'] = str(not explain_with_llm).lower()

        try:
            with st.spinner('Comparing PDFs...'):
                compare_response = requests.post(f"{base_url}{endpoint}", files=files, params=params)
                compare_response.raise_for_status()
                response_json = compare_response.json()
                st.success("Comparison completed!")

                if comparison_method == "Diff palabra a palabra (precisión máxima)":
                    if explain_with_llm:
                        # Mostrar técnico
                        st.subheader("Diferencias línea por línea:")
                        diff_output = "\n".join(response_json.get("diff_lines", []))
                        st.code(diff_output, language="diff")
                        st.json(response_json.get("changes", []))
                        st.info(response_json["summary"]["interpretation"])
                    else:
                        # Mostrar explicación amigable
                        st.subheader("Explicación de las diferencias:")
                        st.write(response_json.get("explanation", "Sin explicación"))
                        st.info(response_json["summary"]["interpretation"])
                else:
                    st.json(response_json)
        except Exception as e:
            st.error(f"Error during comparison: {str(e)}")

    if st.button("Get Legal Phrase"):
        try:
            frase_response = requests.get(f"{base_url}/frase-legal")
            st.info(frase_response.json().get("frase", "No phrase received."))
        except Exception as e:
            st.error(f"Error getting legal phrase: {str(e)}")

if __name__ == "__main__":
    main()
