
import asyncio
import sys
import streamlit as st
from pinecone import Pinecone
from streamlit_chat import message
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint


if sys.platform.startswith("win") and sys.version_info >= (3, 8):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


st.set_page_config(page_title="Workcohol AI Buddy", layout="centered")


@st.cache_resource
def load_qa_chain():
    pc = Pinecone(api_key="pcsk_5GaY1y_8sAoHhVACpt45xUSYZFaifSzCeAoMr8HZbuexftJXE6X4bKxP24NLHTW7MAnZTi")
    index_name = "custom-vectordb"

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = LangchainPinecone.from_existing_index(index_name=index_name, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        task="text-generation",
        huggingfacehub_api_token="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",
        temperature=0.5,
        max_new_tokens=512
    )

    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

qa_chain = load_qa_chain()


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "greeted" not in st.session_state:
    st.session_state.greeted = False

if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False

if "injected_input" not in st.session_state:
    st.session_state.injected_input = None


with st.sidebar:
    st.title("🤖 Workcohol AI Buddy")
    st.markdown("Ask me anything about Workcohol's company policies, services, careers, and more.")

    with st.expander("📜 Chat Journal"):
        if st.session_state.chat_history:
            for i, chat in enumerate(st.session_state.chat_history):
                if chat["role"] == "user":
                    if st.button(f"🧑 You: {chat['content'][:40]}", key=f"user-btn-{i}"):
                        st.session_state.injected_input = chat["content"]
                        st.experimental_rerun()
                else:
                    st.markdown(f"🤖 **Bot:** {chat['content'][:60]}")
        else:
            st.write("No chat history yet.")

    st.markdown("---")
    st.markdown("**🔗 Support & Info**")
    st.markdown("- 🌐 [Visit Workcohol Website](https://www.workcohol.com/)")
    st.markdown("- 📧 support@workcohol.com")

    st.markdown("---")
    if st.button("🗑️ Reset Chat"):
        st.session_state.chat_history = []
        st.session_state.feedback_submitted = False
        st.session_state.greeted = False
        st.session_state.injected_input = None
        st.rerun()

    
    st.markdown("---")
    with st.expander("📝 Give Feedback"):
        if not st.session_state.feedback_submitted:
            st.markdown("How was your experience with **Workcohol AI Buddy**?")
            with st.form("sidebar_feedback_form"):
                rating = st.slider("Rate from 1 (poor) to 5 (excellent)", 1, 5, 4)
                comments = st.text_area("Any suggestions or comments?")
                submitted = st.form_submit_button("Submit")
                if submitted:
                    st.success("🎉 Thank you for your feedback!")
                    st.session_state.feedback_submitted = True
        else:
            st.success("✅ Feedback submitted. Thank you!")
            
if not st.session_state.greeted:
    greeting = "👋 Hey! I'm Workcohol AI Buddy. How can I help you today?."
    st.session_state.chat_history.append({"role": "assistant", "content": greeting})
    st.session_state.greeted = True


st.title("💬 Chat with Workcohol AI Buddy")

for i, chat in enumerate(st.session_state.chat_history):
    message(chat["content"], is_user=(chat["role"] == "user"), key=str(i))


user_input = st.chat_input("Ask your question here...")


if st.session_state.injected_input and not user_input:
    user_input = st.session_state.injected_input
    st.session_state.injected_input = None

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    message(user_input, is_user=True)

    with st.spinner("Investigating.."):
        response = qa_chain.run(user_input)

    st.session_state.chat_history.append({"role": "assistant", "content": response})
    message(response, is_user=False)
