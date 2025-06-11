import json
import os
from datetime import datetime, timedelta
import hashlib

class SessionManager:
    def __init__(self):
        self.sessions_file = "data/sessions.json"
        self.session_duration = timedelta(days=7)  # Las sesiones duran 7 días
        self._ensure_sessions_file()

    def _ensure_sessions_file(self):
        """Asegura que el archivo de sesiones existe"""
        os.makedirs('data', exist_ok=True)
        if not os.path.exists(self.sessions_file):
            with open(self.sessions_file, 'w') as f:
                json.dump({}, f)

    def _load_sessions(self):
        """Carga las sesiones desde el archivo"""
        try:
            with open(self.sessions_file, 'r') as f:
                return json.load(f)
        except:
            return {}

    def _save_sessions(self, sessions):
        """Guarda las sesiones en el archivo"""
        with open(self.sessions_file, 'w') as f:
            json.dump(sessions, f)

    def _clean_expired_sessions(self, sessions):
        """Limpia las sesiones expiradas"""
        now = datetime.now()
        return {
            token: data for token, data in sessions.items()
            if datetime.fromisoformat(data['expires']) > now
        }

    def create_session(self, username, user_role):
        """Crea una nueva sesión para un usuario"""
        sessions = self._load_sessions()
        sessions = self._clean_expired_sessions(sessions)

        # Crear token único
        token = hashlib.sha256(f"{username}{datetime.now()}".encode()).hexdigest()
        
        # Guardar sesión
        sessions[token] = {
            'username': username,
            'role': user_role,
            'created': datetime.now().isoformat(),
            'expires': (datetime.now() + self.session_duration).isoformat()
        }
        
        self._save_sessions(sessions)
        return token

    def validate_session(self, token):
        """Valida una sesión existente"""
        if not token:
            return None

        sessions = self._load_sessions()
        sessions = self._clean_expired_sessions(sessions)
        
        session = sessions.get(token)
        if not session:
            return None

        # Actualizar la expiración
        sessions[token]['expires'] = (datetime.now() + self.session_duration).isoformat()
        self._save_sessions(sessions)
        
        return session

    def delete_session(self, token):
        """Elimina una sesión"""
        sessions = self._load_sessions()
        if token in sessions:
            del sessions[token]
            self._save_sessions(sessions) 