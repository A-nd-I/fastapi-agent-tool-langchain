import requests
import base64

def test_endpoints():
    # URL base
    base_url = "http://localhost:8000"  # Ajusta el puerto según tu configuración

    # 1. Probar el endpoint /frase-legal (GET)
    frase_response = requests.get(f"{base_url}/frase-legal")
    print("\nRespuesta de /frase-legal:")
    print(frase_response.json())

    # 2. Probar el endpoint /compare-pdfs (POST)
    # Primero, necesitamos preparar los archivos PDF
    with open("./2.pdf", "rb") as pdf1_file:
        pdf1_content = pdf1_file.read()
    
    with open("./3.pdf", "rb") as pdf2_file:
        pdf2_content = pdf2_file.read()

    # Crear los archivos para el multipart/form-data
    files = {
        'pdf1': ('documento1.pdf', pdf1_content, 'application/pdf'),
        'pdf2': ('documento2.pdf', pdf2_content, 'application/pdf')
    }

    # Hacer la petición POST
    compare_response = requests.post(
        f"{base_url}/compare-pdfs",
        files=files
    )

    print("\nRespuesta de /compare-pdfs:")
    print(compare_response.json())

if __name__ == "__main__":
    try:
        test_endpoints()
    except Exception as e:
        print(f"Error al probar los endpoints: {str(e)}")