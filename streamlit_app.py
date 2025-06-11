import streamlit as st
import requests
from database import SessionLocal, create_user, get_user_by_username, verify_password, get_user_by_email
from session_manager import SessionManager
import os

# Inicializar el gestor de sesiones
session_manager = SessionManager()

# Inicializar el estado de la sesión si no existe
if 'init' not in st.session_state:
    st.session_state.init = True
    # Intentar cargar la sesión desde el almacenamiento persistente
    try:
        with open('data/current_session.txt', 'r') as f:
            st.session_state.session_token = f.read().strip()
    except:
        st.session_state.session_token = None

# Configurar layout amplio
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

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
        .auth-form {
            max-width: 400px;
            margin: 0 auto;
            padding: 20px;
        }
        .form-divider {
            margin: 20px 0;
            text-align: center;
        }
        .sidebar-title {
            text-align: center;
            font-size: 1.5rem;
            font-weight: bold;
            margin-bottom: 2rem;
            color: #262730;
            cursor: pointer;
            text-decoration: none;
        }
        .sidebar-title:hover {
            color: #0178e4;
        }
    </style>
""", unsafe_allow_html=True)

# URL de tu backend
base_url = "http://localhost:8000"

def show_register_form():
    st.title("📝 Registro de Usuario")
    
    with st.form("register_form"):
        username = st.text_input("Usuario")
        email = st.text_input("Email")
        password = st.text_input("Contraseña", type="password")
        password_confirm = st.text_input("Confirmar Contraseña", type="password")
        
        submit_button = st.form_submit_button("Registrarse")
        
        if submit_button:
            if password != password_confirm:
                st.error("❌ Las contraseñas no coinciden")
                return
            
            if not username or not email or not password:
                st.error("❌ Todos los campos son obligatorios")
                return
            
            try:
                db = SessionLocal()
                # Verificar si el usuario ya existe
                if get_user_by_username(db, username):
                    st.error("❌ El nombre de usuario ya está en uso")
                    return
                
                if get_user_by_email(db, email):
                    st.error("❌ El email ya está registrado")
                    return
                
                # Crear nuevo usuario
                create_user(db, username, email, password)
                st.success("✅ Usuario registrado exitosamente!")
                st.session_state.show_login = True
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error al registrar usuario: {str(e)}")
            finally:
                db.close()

def login_page():
    # Verificar si hay una sesión activa
    if 'session_token' in st.session_state and st.session_state.session_token:
        session = session_manager.validate_session(st.session_state.session_token)
        if session:
            st.session_state.logged_in = True
            st.session_state.username = session['username']
            st.session_state.user_role = session['role']
            return True
        else:
            # Limpiar sesión inválida
            st.session_state.session_token = None
            try:
                os.remove('data/current_session.txt')
            except:
                pass

    st.session_state.setdefault('logged_in', False)
    st.session_state.setdefault('show_login', True)
    
    if not st.session_state.logged_in:
        if st.session_state.show_login:
            st.title("🔐 Login")
            
            username = st.text_input("Usuario")
            password = st.text_input("Contraseña", type="password")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Iniciar Sesión"):
                    if username and password:
                        try:
                            db = SessionLocal()
                            user = get_user_by_username(db, username)
                            
                            if user and verify_password(user.hashed_password, password):
                                # Crear sesión persistente
                                session_token = session_manager.create_session(username, user.role)
                                st.session_state.session_token = session_token
                                # Guardar el token en archivo
                                with open('data/current_session.txt', 'w') as f:
                                    f.write(session_token)
                                st.session_state.logged_in = True
                                st.session_state.username = username
                                st.session_state.user_role = user.role
                                st.rerun()
                            else:
                                st.error("❌ Usuario o contraseña incorrectos")
                        finally:
                            db.close()
                    else:
                        st.error("❌ Por favor ingrese usuario y contraseña")
            
            with col2:
                if st.button("Registrarse"):
                    st.session_state.show_login = False
                    st.rerun()
            
            st.markdown("""
            ---
            **Credenciales de prueba:**
            - Usuario: admin
            - Contraseña: admin
            """)
        else:
            show_register_form()
            if st.button("¿Ya tienes cuenta? Inicia Sesión"):
                st.session_state.show_login = True
                st.rerun()
        
        return False
    
    return True

def main_content():
    st.title("Agente Comparador de Contratos - Lawgent")

    # Agregar título y botón de logout en la barra lateral
    with st.sidebar:
        st.markdown('<div class="sidebar-title">🤖 LawGent</div>', unsafe_allow_html=True)
        if st.sidebar.button("Inicio"):
            st.rerun()
        if st.button("Cerrar Sesión"):
            if 'session_token' in st.session_state:
                session_manager.delete_session(st.session_state.session_token)
                st.session_state.session_token = None
                try:
                    os.remove('data/current_session.txt')
                except:
                    pass
            st.session_state.logged_in = False
            st.rerun()

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

def main():
    if login_page():
        main_content()

if __name__ == "__main__":
    main()
