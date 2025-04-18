import streamlit as st

st.set_page_config(page_title="Test App")

def main():
    st.title("Test Streamlit App")
    st.write("If you can see this, Streamlit is working correctly!")
    
    st.write("Here's a button:")
    if st.button("Click me"):
        st.success("Button clicked!")
    
    st.write("Here's a slider:")
    value = st.slider("Select a value", 0, 100, 50)
    st.write(f"You selected: {value}")

if __name__ == "__main__":
    main() 