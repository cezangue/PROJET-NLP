import streamlit as st

def main():
    st.title("Analyse de discours")
    url = st.text_input("Entrez l'URL du discours :")
    if url:
        st.write(f"Vous avez entré l'URL : {url}")

if __name__ == "__main__":
    main()
