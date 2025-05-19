import requests
import json
import base64

def test_compare_pdfs():
    # Leer los archivos PDF y convertirlos a base64
    with open('./1.pdf', 'rb') as f1, open('./2.pdf', 'rb') as f2:
        pdf1_base64 = base64.b64encode(f1.read()).decode('utf-8')
        pdf2_base64 = base64.b64encode(f2.read()).decode('utf-8')

    # Crear el payload en el formato que espera LangServe
    payload = {
        "input": {
            "input": "Compare estos PDFs",
            "pdf1_base64": pdf1_base64,
            "pdf2_base64": pdf2_base64
        }
    }
    
    try:
        response = requests.post(
            'http://localhost:8000/general-agent/invoke',
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            print("Éxito:", response.json())
        else:
            print("Error:", response.status_code)
            print("Respuesta:", response.text)
            
    except Exception as e:
        print("Error en la petición:", str(e))

if __name__ == "__main__":
    test_compare_pdfs() 