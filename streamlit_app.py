import streamlit as st
import requests
import base64

def main():
    st.title("PDF Comparison Tool")
    
    # File uploaders for PDFs
    pdf1 = st.file_uploader("Upload first PDF", type=['pdf'])
    pdf2 = st.file_uploader("Upload second PDF", type=['pdf'])
    
    # Método de comparación
    comparison_method = st.radio(
        "Select comparison method:",
        ("Embeddings global + por secciones", "FAISS CPU vector search"),
        index=0
    )
    
    # URL base
    base_url = "http://localhost:8000"
    
    if st.button("Compare PDFs") and pdf1 is not None and pdf2 is not None:
        
        # Preparar archivos
        files = {
            'pdf1': ('documento1.pdf', pdf1.getvalue(), 'application/pdf'),
            'pdf2': ('documento2.pdf', pdf2.getvalue(), 'application/pdf')
        }
        
        # Determinar endpoint
        endpoint = "/compare-pdfs" if comparison_method == "Embeddings global + por secciones" else "/compare-pdfs-faiss"
        
        try:
            with st.spinner('Comparing PDFs...'):
                compare_response = requests.post(f"{base_url}{endpoint}", files=files)
                compare_response.raise_for_status()
                
                st.success("Comparison completed!")
                st.json(compare_response.json())
                
        except Exception as e:
            st.error(f"Error during comparison: {str(e)}")
    
    # Frase legal
    if st.button("Get Legal Phrase"):
        try:
            frase_response = requests.get(f"{base_url}/frase-legal")
            st.info(frase_response.json().get("frase", "No phrase received."))
        except Exception as e:
            st.error(f"Error getting legal phrase: {str(e)}")

if __name__ == "__main__":
    main()
