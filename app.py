# 1. IMPORT THE LIBRARIES WE NEED

import os  # Lets Python check whether files and folders exist.

import streamlit as st  # Builds the website and its user interface.

# These tools help us create document embeddings and search the policy.
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# This connects our app to the Qwen model running through Ollama.
from langchain_ollama import ChatOllama


# 2. SET THE FILE PATHS AND AI MODEL

# Folder containing the saved FAISS search index.
# The index lets us find relevant sections of the policy.

INDEX_DIR = "./rag_faiss_index"

# Name of the AI model we want Ollama to run.
MODEL_NAME = "qwen2.5:7b"

# Location of the full policy PDF that users can download.
POLICY_PDF = "./docs/Accounts_Payable_Policy.pdf"


# 3. SET UP THE STREAMLIT PAGE

# Sets the browser tab title and the icon for the website.
st.set_page_config(
    page_title="AuditPilot",
    page_icon=":material/shield:"
)

# 4. STYLE THE WEBSITE WITH CSS
# CSS controls how parts of the website look, including button colors, borders, spacing, and hover effects.
st.markdown(
    """
<style>
/* Keep the main content from becoming too wide. */
.block-container {
    max-width: 900px;
    padding-top: 2.5rem;
}

/* Give buttons rounded corners and a border. */
.stButton > button,
[data-testid="stFormSubmitButton"] button,
[data-testid="stDownloadButton"] button {
    border-radius: 8px;
    border: 1px solid #3A4A63;
}

/* Change normal and download buttons when the mouse hovers over them. */
.stButton > button:hover,
[data-testid="stDownloadButton"] button:hover {
    border-color: #4C8BF5;
    color: #4C8BF5;
    background: #162033;
}

/* Give the main Ask button a blue background. */
[data-testid="stFormSubmitButton"] button {
    background: #2F6FDB;
    border-color: #2F6FDB;
    color: #FFFFFF;
    font-weight: 600;
}

/* Make the Ask button a lighter blue when hovered over. */
[data-testid="stFormSubmitButton"] button:hover {
    background: #4C8BF5;
    border-color: #4C8BF5;
    color: #FFFFFF;
}

/* Show an outline when a user focuses on a button using a keyboard. */
.stButton > button:focus-visible,
[data-testid="stFormSubmitButton"] button:focus-visible,
[data-testid="stDownloadButton"] button:focus-visible {
    outline: 2px solid #4C8BF5;
    outline-offset: 2px;
}
</style>
""",
    unsafe_allow_html=True,  # Allows Streamlit to display our custom CSS.
)


# 5. DISPLAY THE PAGE HEADER

# Display the main title in the center of the page.
st.markdown(
    "<h1 style='text-align:center; margin-bottom:0;'>AuditPilot</h1>",
    unsafe_allow_html=True,
)

# Display the subtitle underneath the main title.
st.markdown(
    "<h4 style='text-align:center; color:#b0b0b0; margin-top:0;'>"
    "Internal Audit Policy Navigator</h4>",
    unsafe_allow_html=True,
)

# Draw a horizontal line to separate the header from the content.
st.divider()

# Explain what AuditPilot does.
st.markdown(
    "<p style='text-align:center;'>"
    "A RAG-powered application for retrieving policy-grounded "
    "answers from Accounts Payable documentation."
    "</p>",
    unsafe_allow_html=True,
)


# 6. CREATE HELPER FUNCTIONS

# This function removes the previous answer when the user
# clicks the Clear answer button in the sidebar.
def clear_answer():
    st.session_state.pop("last_answer", None)


# This function checks whether the AI answer suggests
# that it could not find the requested information.
def looks_unanswered(answer):
    # Convert the answer to lowercase so capitalization
    # does not affect the check.
    a = answer.lower()

    # Return True if one of these phrases appears.
    return any(
        phrase in a
        for phrase in ("don't know", "do not know", "don’t know")
    )



# 7. CHECK WHETHER OLLAMA HAS THE MODEL

# Cache the result for 30 seconds so we don't repeatedly
# check Ollama every time Streamlit refreshes the page.
@st.cache_data(ttl=30)
def check_model(name):
    """
    True  = Ollama is reachable and the model is installed.
    False = Ollama is reachable, but the model is missing.
    None  = The check failed, possibly because Ollama is not running.
    """
    try:
        import ollama

        # Different versions of the Ollama library may return
        # model information in slightly different formats.
        def _name(m):
            if isinstance(m, dict):
                return m.get("model") or m.get("name")

            return getattr(m, "model", None) or getattr(m, "name", None)

        # Get the installed models and check for the requested model.
        return name in [
            _name(model)
            for model in ollama.list()["models"]
        ]

    except Exception:
        # If the check fails, return None instead of crashing the app.
        return None



# 8. LOAD THE SEARCH INDEX, EMBEDDINGS, AND AI MODEL

# Cache these resources because loading them can take time.
# Streamlit can reuse them instead of loading everything again.
@st.cache_resource
def load_everything():

    # Convert text into numerical representations called embeddings.
    # This must match the embedding model used when building the index.
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    # Load the saved FAISS index that contains the policy's
    # searchable document chunks.
    db = FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Connect LangChain to the Qwen model running through Ollama.
    # temperature=0 encourages more consistent answers.
    llm = ChatOllama(
        model=MODEL_NAME,
        temperature=0
    )

    # Return both objects so the rest of the app can use them.
    return db, llm



# 9. BUILD THE SIDEBAR

# This function shows the app's status and helpful information
# in the sidebar on the left side of the website.
def render_sidebar(index_ok):

    # Check whether the configured Ollama model is available.
    model_ok = check_model(MODEL_NAME)

    # Put the following content inside the sidebar.
    with st.sidebar:

        st.markdown("**AuditPilot**")
        st.caption("Internal Audit Policy Navigator")

        # Explain which policy the app currently covers.
        st.markdown(
            ":material/description: **Scope**  \n"
            "Accounts Payable Policy"
        )

        # Show the basic process followed by the RAG app.
        st.markdown(
            ":material/format_list_numbered: **How it works**  \n"
            "Question → Policy excerpts → Answer"
        )

        st.markdown(":material/fact_check: **Status**")

        # Show whether the saved policy index exists.
        if index_ok:
            st.markdown(":green[●] Policy documents: Ready")
        else:
            st.markdown(":red[●] Policy documents: Not found")

        # Show whether Ollama and the model are available.
        if model_ok is True:
            st.markdown(":green[●] Answer assistant: Ready")

        elif model_ok is False:
            st.markdown(":red[●] Answer assistant: Model not installed")

        else:
            st.markdown(":red[●] Answer assistant: Not reachable")

        # Only show the Clear answer button when an answer exists.
        if "last_answer" in st.session_state:
            st.button(
                "Clear answer",
                icon=":material/delete:",
                on_click=clear_answer
            )



# 10. CHECK WHETHER THE POLICY INDEX EXISTS
# Check whether the main FAISS index file exists.
index_ok = os.path.exists(
    os.path.join(INDEX_DIR, "index.faiss")
)

# If the index is missing, show an error and stop the app.
# Without the index, AuditPilot cannot search the policy.
if not index_ok:

    st.error(
        "The policy index was not found. "
        "Run rag_index_builder.py first, then reload this page.",
        icon=":material/error:",
    )

    # Still display the sidebar so the user can see the status.
    render_sidebar(index_ok)

    # Stop executing the remaining code.
    st.stop()



# 11. LOAD THE RAG COMPONENTS

# Load the searchable policy database and the AI model.
db, llm = load_everything()


# 12. CREATE THE QUESTION INPUT AND ASK BUTTON
# A Streamlit form groups the question field and button together.


# Pressing Enter in the field can also submit the form.
with st.form("ask_form"):

    # Text box where the user types a question about the policy.
    query = st.text_input("Ask a question about the policy")

    # Button that submits the question.
    submitted = st.form_submit_button(
        "Ask AuditPilot",
        icon=":material/send:"
    )


# 13. PROCESS THE USER'S QUESTION

# Run this section only after the user submits the form.
if submitted:

    # Do not process an empty question.
    if not query.strip():

        st.warning(
            "Please type a question first.",
            icon=":material/edit:"
        )

    else:
        try:
            # Display a loading message while the answer is prepared.
            with st.spinner("Thinking..."):

                # RAG STEP 1:
                # Search the FAISS index for the three most relevant
                # pieces of policy text for this question.
                docs = db.similarity_search(query, k=3)

                # Combine the retrieved text into one context string.
                context = "\n\n".join(
                    document.page_content
                    for document in docs
                )

                # RAG STEP 2:
                # Tell the AI to answer using the retrieved policy text.
                # The rules discourage guessing or inventing policy details.
                prompt = f"""Answer the question using only the policy excerpts below.

Rules:
- Prefer explicit policy names and policy reference numbers when asked which policy governs a requirement.
- Do not confuse a section heading with the name of a governing policy.
- Do not invent or change policy titles or reference numbers.
- If the answer is not explicitly supported by the excerpts, say you don't know.

Excerpts:
{context}

Question: {query}

Answer:"""

                # RAG STEP 3:
                # Send the prompt to Qwen through Ollama.
                # Save the answer in Streamlit's session state so it
                # remains visible when the page refreshes.
                st.session_state.last_answer = llm.invoke(prompt).content

        except Exception as exc:

            # Print the technical error in the terminal for debugging.
            print(f"AuditPilot error: {exc!r}")

            # Show a simpler message to the user.
            st.error(
                "Something went wrong while preparing your answer. "
                "Please check that Ollama is running and try again.",
                icon=":material/error:",
            )

# 14. DISPLAY THE ANSWER

# If an answer has been saved, display it.
if "last_answer" in st.session_state:

    st.subheader("Answer")

    # Show the AI-generated answer on the page.
    st.write(st.session_state.last_answer)

    # If the AI says it doesn't know, explain what the user can do.
    if looks_unanswered(st.session_state.last_answer):

        st.info(
            "This could not be found in the available policy documents. "
            "Try rephrasing the question, or check the full policy.",
            icon=":material/info:",
        )

    # Point users to the verification guidance further down the page.
    st.caption(
        'To check this answer against the policy, '
        'see "How to verify answers" below.'
    )

# If the user hasn't asked anything yet, show a helpful message.
else:

    st.info(
        "Your answer will appear here. Ask about things like approvals, "
        "payment timing, supplier changes, or reconciliation.",
        icon=":material/chat:",
    )
# 15. DISPLAY THE INFORMATION CARDS

st.divider()
st.subheader("About AuditPilot")

# Split the available page width into three columns.
col1, col2, col3 = st.columns(3)


# CARD 1: Explain what AuditPilot does.
with col1:

    # A bordered container groups the card's contents together.
    with st.container(border=True, key="card_what"):

        st.markdown(":material/search: **What AuditPilot does**")

        st.write(
            "Answers questions about your Accounts Payable policy "
            "in plain language, using only the text of the policy."
        )


# CARD 2: Explain how the app works.
with col2:

    with st.container(border=True, key="card_how"):

        st.markdown(
            ":material/format_list_numbered: **How it works**"
        )

        st.write(
            "1. You ask a question.  \n"
            "2. AuditPilot finds the most relevant parts of the policy.  \n"
            "3. It writes an answer based only on those parts."
        )


# CARD 3: Let users download the policy and find support.
with col3:

    with st.container(border=True, key="card_verify"):

        st.markdown(
            ":material/fact_check: **Policy Document & Support**"
        )

        st.write(
            "Download the full Accounts Payable Policy below to review "
            "the relevant sections and verify AuditPilot’s responses. "
            "If AuditPilot cannot provide a clear answer based on the "
            "policy, contact the Finance team for clarification."
        )

        # Only show the download button if the PDF exists.
        if os.path.exists(POLICY_PDF):

            # Open the PDF in binary mode so Streamlit can download it.
            with open(POLICY_PDF, "rb") as f:

                st.download_button(
                    "Download policy (PDF)",
                    data=f.read(),
                    file_name="Accounts_Payable_Policy.pdf",
                    mime="application/pdf",
                    icon=":material/download:",
                )

        else:
            # Tell the user if the PDF is missing.
            st.caption(
                "The policy PDF was not found in the docs folder."
            )


# 16. DISPLAY THE FOOTER
st.divider()

# Display a small reminder at the bottom of the page.
st.markdown(
    "<p style='text-align:center; color:#8a94a6; font-size:0.85rem;'>"
    "AuditPilot answers are generated from the Accounts Payable Policy "
    "(FIN-AP-001). Confirm important decisions against the official policy."
    "</p>",
    unsafe_allow_html=True,
)

# 17. DISPLAY THE SIDEBAR

# Build the sidebar after the main page has been set up.
render_sidebar(index_ok)
