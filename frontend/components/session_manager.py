# frontend/components/session_manager.py
import os
import json
import logging
from typing import List, Dict, Any
import streamlit as st

logger = logging.getLogger(__name__)
SESSION_FILE = os.path.join(os.path.dirname(__file__), "..", ".memorypal_session.json")


class SessionManager:
    """Handles persistent user session data, active uploads, and workspace state."""

    def __init__(self):
        self.session_file = os.path.abspath(SESSION_FILE)
        if "session_data" not in st.session_state:
            st.session_state.session_data = self._load_session()

    def _load_session(self) -> Dict[str, Any]:
        if os.path.exists(self.session_file):
            try:
                with open(self.session_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
            except Exception as e:
                logger.warning(f"Failed to read session file: {e}")

        return {"uploaded_files": [], "recent_topics": []}

    def _save_session(self):
        try:
            os.makedirs(os.path.dirname(self.session_file), exist_ok=True)
            with open(self.session_file, "w", encoding="utf-8") as f:
                json.dump(st.session_state.session_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving session: {e}")

    def add_file(self, filename: str):
        """Add an indexed filename to session."""
        files = st.session_state.session_data.setdefault("uploaded_files", [])
        if filename and filename not in files:
            files.append(filename)
            self._save_session()

    def remove_file(self, filename: str):
        """Remove a file from session tracking."""
        files = st.session_state.session_data.get("uploaded_files", [])
        if filename in files:
            files.remove(filename)
            self._save_session()

    def list_files(self) -> List[str]:
        """Return list of uploaded files from session."""
        return st.session_state.session_data.get("uploaded_files", [])

    def clear(self):
        """Clear session memory."""
        st.session_state.session_data = {"uploaded_files": [], "recent_topics": []}
        self._save_session()
