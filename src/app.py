import streamlit as st
import PyPDF2
from io import StringIO
import PyPDF2
from RAG_GGUF import RAG_Chain

'''
def chatbot_response(user_input):
    # You can replace this with a more sophisticated chatbot logic
    return f"Bot: You said '{user_input}'"
'''

# Function to send pdf file to RAG pipeline
def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page_num in range(len(pdf_reader.pages) ):
        page = pdf_reader.pages[page_num] 
        text += page.extract_text()
    return text

st.title("Talk with Your PDF")

# PDF Upload
st.header("Upload a PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file is not None:
        # Display the file name
    st.write(f"File uploaded: {uploaded_file.name}")
    # Read and display the content of the uploaded PDF file
    try:
        pdf_content = read_pdf(uploaded_file)
        st.text_area("PDF Content", pdf_content, height=300)
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
# Input field for user messages
user_input = st.text_input("You:", "")

# Initialize a session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Handle user input
if st.button("Send"):
    if user_input:
        # Get the GPT response
        gpt_response = RAG_Chain(uploaded_file,user_input)

        # Store the conversation
        st.session_state.chat_history.append(("User", user_input))
        st.session_state.chat_history.append(("BOT", gpt_response))

        # Clear the input box
        user_input = ""

# Display chat history
for speaker, message in st.session_state.chat_history:
    if speaker == "User":
        st.markdown(f"**{speaker}:** {message}")
    else:
        st.markdown(f"**{speaker}:** {message}")
  

    








