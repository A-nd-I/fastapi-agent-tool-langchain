import streamlit as st
import requests

# Añadir esta línea al inicio del archivo, antes de cualquier otro comando de st
st.set_page_config(layout="wide")

# Añadir estos estilos CSS al inicio del archivo, después de set_page_config
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
        .stSpinner {
            position: relative;
            top: 0;
            left: 0;
        }
    </style>
""", unsafe_allow_html=True)

def main():
    st.title("Agente Comparador de Contratos - Lawgent")

    # Crear dos columnas principales
    left_col, right_col = st.columns(2)

    # Columna izquierda: formulario y controles
    with left_col:
        pdf1 = st.file_uploader("Upload Contract 1", type=['pdf'])
        pdf2 = st.file_uploader("Upload Contract 2", type=['pdf'])

         # comparison_method = st.radio(
         #    "Select comparison method:",
         #    (
         #        "Embeddings global + por secciones",
         #        "FAISS CPU vector search",
         #        "Diff palabra a palabra (precisión máxima)"
         #    ),
         #    index=2
         # )
        # Simplificamos a solo una opción de comparación
        comparison_method = "Diff palabra a palabra (precisión máxima)"

        explain_with_llm = st.checkbox(
            "Mostrar diferencias técnicas (diff)", value=False,
            help="Por defecto verás una explicación en lenguaje sencillo. Marca para ver el diff técnico."
        )

        compare_clicked = st.button("Comparar PDFs")

    

    base_url = "http://localhost:8000"
    endpoints = {
        "Diff palabra a palabra (precisión máxima)": "/compare-pdfs-diff"
    }

    # Columna derecha: resultados
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
            endpoint = endpoints[comparison_method]
            params = {}
            if comparison_method == "Diff palabra a palabra (precisión máxima)":
                params['explain_with_llm'] = str(not explain_with_llm).lower()

            try:
                with st.spinner('Comparing PDFs...'):
                    compare_response = requests.post(f"{base_url}{endpoint}", files=files, params=params)
                    compare_response.raise_for_status()
                    result_data = compare_response.json()

                st.success("Comparison completed!")

                if comparison_method == "Diff palabra a palabra (precisión máxima)":
                    if explain_with_llm:
                        st.subheader("Diferencias línea por línea:")
                        diff_output = "\n".join(result_data.get("diff_lines", []))
                        st.code(diff_output, language="diff")
                        st.json(result_data.get("changes", []))
                        st.info(result_data["summary"]["interpretation"])
                    else:
                        st.subheader("Explicación de las diferencias:")
                        st.write(result_data.get("explanation", "Sin explicación"))
                        st.info(result_data["summary"]["interpretation"])
                else:
                    st.json(result_data)
            except Exception as e:
                st.error(f"Error during comparison: {str(e)}")
        else:
            st.markdown("""
                ### 👋 Bienvenido al Agente Comparador de Contratos - Lawgent
                
                #### Instrucciones:
                1. Sube dos contratos en formato PDF en la columna izquierda
                2. Decide si quieres ver las diferencias técnicas marcando la casilla correspondiente
                3. Haz clic en "Comparar PDFs" para ver los resultados
                
                #### Características:
                - Comparación detallada palabra por palabra
                - Resumen en lenguaje natural de las diferencias
                - Visualización técnica de cambios (opcional)
                
                *Los resultados aparecerán en este espacio una vez que se realice la comparación.*

                ---
                
                ⚠️ **AVISO DE RESPONSABILIDAD**:
                > Este sistema opera con una precisión predictiva que puede ser inferior o igual al 99%. El usuario es totalmente 
                > responsable de verificar la veracidad y exactitud de toda la información proporcionada. Los resultados 
                > deben ser validados manualmente antes de tomar cualquier decisión legal o contractual.
            """)
            
            # Añadir una imagen o icono placeholder (opcional)
            st.markdown("""
                <div style="text-align: center; padding: 2rem;">
                    <i class="fas fa-file-contract" style="font-size: 5rem; color: #f0f2f6;"></i>
                </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
