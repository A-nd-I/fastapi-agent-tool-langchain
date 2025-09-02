import streamlit as st
import requests
from database import SessionLocal, create_user, get_user_by_username, verify_password, get_user_by_email
from session_manager import SessionManager
from stripe_manager import (
    SUBSCRIPTION_PLANS,
    create_checkout_session,
    get_subscription_status,
    update_comparison_count,
    can_make_comparison
)
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
            st.markdown(
                """
                <hr>
                <div style="background-color: #f0f2f6; border-radius: 10px; padding: 18px 24px; margin-top: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.04);">
                    <h4 style="margin-bottom: 8px; color: #2c3e50;">🔖 <span style="color:#1a73e8;">Lawgent</span> <span style="font-size: 0.85em; color: #888;">v0.1.0</span></h4>
                    <p style="margin-bottom: 6px;"><b>Descripción:</b> Agente Comparador de Contratos</p>
                    <p style="margin-bottom: 6px;"><b>Autor:</b> Carlos Andres Mogollón</p>
                    <p style="margin-bottom: 6px;"><b>Email:</b> <a href="mailto:mogollonrojascarlosandres@gmail.com">mogollonrojascarlosandres@gmail.com</a></p>
                    <p style="margin-bottom: 0;"><b>Teléfono:</b> <a href="tel:+573209221591">+57 320 922 15 91</a></p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            show_register_form()
            if st.button("¿Ya tienes cuenta? Inicia Sesión"):
                st.session_state.show_login = True
                st.rerun()
        return False
    return True

def show_profile():
    """Muestra la página de perfil del usuario"""
    st.title("👤 Perfil de Usuario")
    try:
        db = SessionLocal()
        user = get_user_by_username(db, st.session_state.username)
        if user:
            # Crear dos columnas
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Información Personal")
                st.write(f"**Usuario:** {user.username}")
                st.write(f"**Email:** {user.email}")
                st.write(f"**Rol:** {user.role}")
                st.write(f"**Cuenta creada:** {user.created_at.strftime('%d/%m/%Y')}")
                st.write(f"**Estado:** {'Activo' if user.is_active else 'Inactivo'}")
            with col2:
                st.markdown("### Estadísticas")
                st.info("📊 Próximamente se agregarán estadísticas de uso")
    except Exception as e:
        st.error(f"Error al cargar el perfil: {str(e)}")
    finally:
        db.close()

def show_subscription_page():
    """Muestra la página de suscripción"""
    st.title("💳 Planes de Suscripción")
    # Obtener estado actual de suscripción
    try:
        db = SessionLocal()
        user = get_user_by_username(db, st.session_state.username)
        subscription = get_subscription_status(user.id, db) if user else None
        
        if subscription:
            if subscription.get('is_free_plan'):
                st.info(f"🎉 Estás usando el plan gratuito - Te quedan {subscription['remaining_comparisons']} comparaciones")
                if subscription['remaining_comparisons'] <= 0:
                    st.warning("⚠️ Has alcanzado el límite de comparaciones gratuitas")
            else:
                st.success(f"✅ Tienes una suscripción activa al {subscription['plan']}")
                st.write(f"Tu suscripción se renovará el {subscription['current_period_end'].strftime('%d/%m/%Y')}")
                if not subscription.get('is_free_plan') and st.button("❌ Cancelar Suscripción"):
                    st.warning("Función de cancelación en desarrollo")

        # Mostrar planes disponibles
        st.write("### Planes Disponibles")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                f"""
                <div style="padding: 20px; border: 1px solid #ddd; border-radius: 10px; height: 100%;">
                    <h3>{SUBSCRIPTION_PLANS['free']['name']}</h3>
                    <h2>Gratis</h2>
                    <hr>
                    <ul>
                        {''.join([f'<li>{feature}</li>' for feature in SUBSCRIPTION_PLANS['free']['features']])}
                    </ul>
                    </div>
                """,
                unsafe_allow_html=True
            )
            if st.button("🎁 Usar Plan Gratuito"):
                try:
                    user.subscription_plan = "free"
                    db.commit()
                    st.success("✅ Plan gratuito activado!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error al activar plan gratuito: {str(e)}")
        
        with col2:
            st.markdown(
                f"""
                <div style="padding: 20px; border: 1px solid #ddd; border-radius: 10px; height: 100%;">
                    <h3>{SUBSCRIPTION_PLANS['basic']['name']}</h3>
                    <h2>${SUBSCRIPTION_PLANS['basic']['price']}/mes</h2>
                    <hr>
                    <ul>
                        {''.join([f'<li>{feature}</li>' for feature in SUBSCRIPTION_PLANS['basic']['features']])}
                    </ul>
                    </div>
                """,
                unsafe_allow_html=True
            )
            if st.button("🔓 Suscribirse al Plan Básico"):
                checkout_session = create_checkout_session(
                    'basic',
                    user.id,
                    success_url=f"{base_url}/success",
                    cancel_url=f"{base_url}/cancel"
                )
                if checkout_session:
                    st.markdown(f"<meta http-equiv='refresh' content='0;url={checkout_session.url}'>", unsafe_allow_html=True)
        
        with col3:
            st.markdown(
                f"""
                <div style="padding: 20px; border: 1px solid #1a73e8; border-radius: 10px; height: 100%; background-color: #f8f9fa;">
                    <h3>{SUBSCRIPTION_PLANS['pro']['name']}</h3>
                    <h2>${SUBSCRIPTION_PLANS['pro']['price']}/mes</h2>
                    <hr>
                    <ul>
                        {''.join([f'<li>{feature}</li>' for feature in SUBSCRIPTION_PLANS['pro']['features']])}
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )
            if st.button("🚀 Suscribirse al Plan Pro"):
                checkout_session = create_checkout_session(
                    'pro',
                    user.id,
                    success_url=f"{base_url}/success",
                    cancel_url=f"{base_url}/cancel"
                )
                if checkout_session:
                    st.markdown(f"<meta http-equiv='refresh' content='0;url={checkout_session.url}'>", unsafe_allow_html=True)
                    
    except Exception as e:
        st.error(f"Error al cargar los planes de suscripción: {str(e)}")
    finally:
        db.close()

def main_content():
    # Variable para controlar la vista actual
    st.session_state.setdefault('current_view', 'main')

    # Agregar título y botones en la barra lateral
    with st.sidebar:
        st.markdown('<div class="sidebar-title">🤖 LawGent</div>', unsafe_allow_html=True)
        if st.sidebar.button("Inicio"):
            st.session_state.current_view = 'main'
            st.rerun()
        if st.sidebar.button("👤 Mi Perfil"):
            st.session_state.current_view = 'profile'
            st.rerun()
        if st.sidebar.button("💳 Suscripción"):
            st.session_state.current_view = 'subscription'
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

    # Mostrar la vista correspondiente
    if st.session_state.current_view == 'profile':
        show_profile()
    elif st.session_state.current_view == 'subscription':
        show_subscription_page()
    else:
        # Verificar límites de suscripción antes de mostrar la interfaz
        try:
            db = SessionLocal()
            user = get_user_by_username(db, st.session_state.username)
            if not can_make_comparison(user.id, db):
                st.warning("⚠️ Has alcanzado el límite de comparaciones de tu plan")
                if st.button("📝 Ver planes de suscripción"):
                    st.session_state.current_view = 'subscription'
                    st.rerun()
                return

            # Tabs primero, antes del título
            tab1, tab2 = st.tabs(["Comparar PDFs", "Preguntar PDFs"])
            with tab1:
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
                        # Actualizar contador de comparaciones
                        update_comparison_count(user.id, db)
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
                    if explain_with_llm:
                        st.markdown("""
                        ### 👋 Bienvenido al Agente Comparador de Contratos - Lawgent
                        #### Instrucciones:
                        1. Sube dos contratos en formato PDF en la columna izquierda
                        2. Decide si quieres ver las diferencias técnicas o explicaciones legales
                        3. Haz clic en "Comparar PDFs" para ver los resultados
                        ---
                        ⚠️ **AVISO**: Este sistema puede tener una precisión menor al 99%. Verifica siempre los resultados antes de usarlos legalmente.
                        """)

            with tab2:
                st.title("Consultor de Documentos - Lawgent")
                st.header("Preguntar sobre múltiples PDFs")
                uploaded_files = st.file_uploader("Sube tus archivos PDF", accept_multiple_files=True, type=['pdf'])
                question = st.text_input("Introduce tu pregunta")

                if st.button("Enviar pregunta"):
                    if uploaded_files and question:
                        files = [("pdfs", (file.name, file, 'application/pdf')) for file in uploaded_files]
                        data = {"question": question}
                        response = requests.post(f"{base_url}/ask-pdfs", files=files, data=data)
                        if response.status_code == 200:
                            answer = response.json().get("answer", "No se pudo obtener una respuesta.")
                            st.success(f"Respuesta: {answer}")
                        else:
                            st.error("Hubo un error en la solicitud.")
                    else:
                        st.warning("Por favor, sube archivos PDF e introduce una pregunta.")
        except Exception as e:
            st.error(f"Error al verificar suscripción: {str(e)}")
        finally:
            db.close()

def main():
    if login_page():
        main_content()

if __name__ == "__main__":
    main()