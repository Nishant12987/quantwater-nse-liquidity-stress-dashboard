# Page structure
import streamlit as st


def section_header(title):
    st.markdown(f"""
        <h2 style='margin-top:20px;'>
            {title}
        </h2>
    """, unsafe_allow_html=True)


def card_container():
    return st.container()
