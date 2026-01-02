import streamlit as st

def pdf_uploader():
    st.title("PDF Uploader")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=True)
    if uploaded_file is not None:
        st.success("File uploaded successfully!")
        # Further processing can be done here
    return uploaded_file