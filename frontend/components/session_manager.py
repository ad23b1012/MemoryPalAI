# frontend/components/session_manager.py
import os
import json
import streamlit as st

SESSION_FILE = "frontend/.memorypal_session.json"

class SessionManager:
    """Handles persistent user session data like uploaded files and state."""

    def __init__(self):
        self.session_file = SESSION_FILE
        if "session_data" not in st.session_state:
            st.session_state.session_data = self._load_session()

    def _load_session(self):
        if os.path.exists(self.session_file):
            try:
                with open(self.session_file, "r") as f:
                    return json.load(f)
            except Exception:
                return {"uploaded_files": []}
        else:
            return {"uploaded_files": []}

    def _save_session(self):
        try:
            os.makedirs(os.path.dirname(self.session_file), exist_ok=True)
            with open(self.session_file, "w") as f:
                json.dump(st.session_state.session_data, f, indent=2)
        except Exception as e:
            print(f"❌ Error saving session: {e}")

    def add_file(self, file_path: str):
        """Add a new uploaded file path to persistent session."""
        files = st.session_state.session_data.get("uploaded_files", [])
        if file_path not in files:
            files.append(file_path)
            st.session_state.session_data["uploaded_files"] = files
            self._save_session()

    def list_files(self):
        """Return list of uploaded files from session."""
        return st.session_state.session_data.get("uploaded_files", [])

    def clear(self):
        """Clear session memory."""
        st.session_state.session_data = {"uploaded_files": []}
        self._save_session()
        st.success("🧹 Session memory cleared.")
