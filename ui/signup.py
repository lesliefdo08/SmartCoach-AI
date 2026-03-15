from __future__ import annotations

import streamlit as st

from modules.auth import signup_user


def render_signup() -> None:
    st.subheader("Sign Up")
    with st.form("signup_form", clear_on_submit=True):
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submitted = st.form_submit_button("Create Account", use_container_width=True)

    if submitted:
        if password != confirm_password:
            st.error("Passwords do not match.")
            return
        ok, message = signup_user(username=username, email=email, password=password)
        if ok:
            st.success(message + " You can now log in.")
        else:
            st.error(message)
