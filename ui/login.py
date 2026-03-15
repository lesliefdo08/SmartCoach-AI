from __future__ import annotations

import streamlit as st

from auth.auth import login_user


def render_login() -> None:
    st.subheader("Login")
    with st.form("login_form", clear_on_submit=False):
        identifier = st.text_input("Username or Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login", use_container_width=True)

    if submitted:
        ok, message, user = login_user(identifier=identifier, password=password)
        if not ok:
            st.error(message)
            return
        st.session_state.authenticated = True
        st.session_state.user = {
            "id": int(user["id"]),
            "username": str(user["username"]),
            "email": str(user["email"]),
        }
        st.success(message)
        st.rerun()
