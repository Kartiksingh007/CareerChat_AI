import streamlit as st
from pathlib import Path

from src.resume_parser import extract_resume_text
from src.text_processor import clean_text, chunk_text
from src.embeddings import create_embeddings
from src.vector_store import create_vector_store
from src.retriever import retrieve_relevant_chunks
from src.llm import generate_answer


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CareerChat AI",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "index": None,
    "chunks": [],
    "resume_name": None,
    "processed": False,
    "messages": [],
    "pending_question": "",
    "question_input": "",
}

for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# CUSTOM CSS
# ============================================================

st.html("""
<style>

    /* ========================================================
       GLOBAL
    ======================================================== */

    .stApp {
        background: #f7f8fc;
    }

    [data-testid="stAppViewContainer"] {
        background: #f7f8fc;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }


    /* ========================================================
       SIDEBAR
    ======================================================== */

    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e8eaf0;
    }

    section[data-testid="stSidebar"] > div {
        padding: 1.5rem 1.2rem;
    }

    .brand {
        padding: 10px 5px 25px 5px;
        border-bottom: 1px solid #eeeeee;
        margin-bottom: 25px;
    }

    .brand-row {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .brand-icon {
        width: 44px;
        height: 44px;
        border-radius: 12px;
        background: #111827;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 21px;
    }

    .brand-title {
        font-size: 20px;
        font-weight: 800;
        color: #111827;
        line-height: 1.1;
    }

    .brand-subtitle {
        color: #7b8190;
        font-size: 12px;
        margin-top: 4px;
    }

    .sidebar-heading {
        font-size: 13px;
        font-weight: 700;
        color: #374151;
        margin-bottom: 8px;
    }

    .sidebar-help {
        color: #8a91a0;
        font-size: 12px;
        line-height: 1.6;
        margin-top: 15px;
        margin-bottom: 10px;
    }


    /* ========================================================
       SUGGESTION BUTTONS
       ======================================================== */

    section[data-testid="stSidebar"] .stButton > button {
        width: 100%;
        text-align: left;
        background: #f8f9fc;
        border: 1px solid #e7e9ef;
        border-radius: 10px;
        padding: 10px 12px;
        margin-top: 7px;
        font-size: 12px;
        color: #374151;
        font-weight: 600;
        min-height: 42px;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        border-color: #6366f1;
        color: #6366f1;
        background: #f5f5ff;
    }


    /* ========================================================
       HERO
    ======================================================== */

    .hero {
        background: #ffffff;
        border: 1px solid #e8eaf0;
        border-radius: 22px;
        padding: 48px 45px;
        margin-bottom: 24px;
        box-shadow: 0 8px 30px rgba(15, 23, 42, 0.04);
    }

    .hero-badge {
        display: inline-block;
        background: #f1f5f9;
        color: #475569;
        border: 1px solid #e2e8f0;
        border-radius: 999px;
        padding: 7px 13px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin-bottom: 18px;
    }

    .hero-title {
        font-size: 42px;
        font-weight: 850;
        line-height: 1.15;
        color: #111827;
        letter-spacing: -1.5px;
        margin-bottom: 15px;
    }

    .hero-title span {
        color: #6366f1;
    }

    .hero-description {
        color: #687083;
        font-size: 16px;
        line-height: 1.7;
        max-width: 700px;
    }


    /* ========================================================
       EMPTY STATE
    ======================================================== */

    .empty-card {
        background: #ffffff;
        border: 1px solid #e8eaf0;
        border-radius: 18px;
        padding: 55px 30px;
        text-align: center;
        margin-top: 20px;
    }

    .empty-icon {
        width: 58px;
        height: 58px;
        margin: auto;
        border-radius: 16px;
        background: #f1f5f9;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 27px;
        margin-bottom: 15px;
    }

    .empty-title {
        font-size: 19px;
        font-weight: 750;
        color: #1f2937;
        margin-bottom: 7px;
    }

    .empty-text {
        color: #8a91a0;
        font-size: 13px;
        line-height: 1.6;
    }


    /* ========================================================
       RESUME READY
    ======================================================== */

    .resume-ready {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 13px;
        padding: 13px 16px;
        color: #166534;
        font-size: 13px;
        margin-bottom: 20px;
    }


    /* ========================================================
       CHAT HEADER
    ======================================================== */

    .chat-container {
        background: #ffffff;
        border: 1px solid #e8eaf0;
        border-radius: 18px;
        padding: 25px;
        margin-top: 20px;
        margin-bottom: 20px;
    }

    .chat-heading {
        font-size: 20px;
        font-weight: 800;
        color: #111827;
        margin-bottom: 3px;
    }

    .chat-subheading {
        color: #8a91a0;
        font-size: 12px;
    }


    /* ========================================================
       CHAT MESSAGES
    ======================================================== */

    [data-testid="stChatMessage"] {
        border-radius: 14px;
        margin-bottom: 14px;
    }

    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] ul,
    [data-testid="stChatMessage"] ol,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] div {
        color: #1f2937 !important;
    }

    [data-testid="stChatMessage"] .stMarkdown {
        color: #1f2937 !important;
    }

    [data-testid="stChatMessage"] .stMarkdown p {
        color: #1f2937 !important;
        font-size: 15px !important;
        line-height: 1.7 !important;
    }

    [data-testid="stChatMessage"] .stMarkdown ul,
    [data-testid="stChatMessage"] .stMarkdown ol {
        color: #1f2937 !important;
        padding-left: 25px;
    }

    [data-testid="stChatMessage"] .stMarkdown li {
        color: #1f2937 !important;
        margin-bottom: 6px;
    }

    [data-testid="stChatMessage"] .stMarkdown strong {
        color: #111827 !important;
    }

    [data-testid="stChatMessage"] .stMarkdown h1,
    [data-testid="stChatMessage"] .stMarkdown h2,
    [data-testid="stChatMessage"] .stMarkdown h3,
    [data-testid="stChatMessage"] .stMarkdown h4 {
        color: #111827 !important;
    }


    /* ========================================================
       BUTTONS
    ======================================================== */

    .stButton > button {
        border-radius: 9px;
        border: 1px solid #e1e4ea;
        background: #ffffff;
        color: #374151;
        font-weight: 600;
    }

    .stButton > button:hover {
        border-color: #6366f1;
        color: #6366f1;
    }


    /* ========================================================
       FILE UPLOADER
    ======================================================== */

    [data-testid="stFileUploader"] {
        background: #f8f9fc;
        border-radius: 12px;
    }

    [data-testid="stFileUploaderDropzone"] {
        border: 1px dashed #d8dce5;
        border-radius: 12px;
        background: #fafbfc;
    }


    /* ========================================================
       TEXT INPUT
    ======================================================== */

    div[data-testid="stTextInput"] input {
        border-radius: 14px;
        border: 1px solid #dfe3eb;
        background: #ffffff;
        color: #1f2937;
        padding: 12px 15px;
    }

    div[data-testid="stTextInput"] input:focus {
        border-color: #6366f1;
        box-shadow: 0 0 0 1px #6366f1;
    }


    /* ========================================================
       SEND BUTTON
    ======================================================== */

    .send-button .stButton > button {
        min-height: 42px;
        border-radius: 12px;
        font-size: 18px;
        font-weight: 700;
    }


    /* ========================================================
       FOOTER
    ======================================================== */

    .footer {
        text-align: center;
        color: #9aa1af;
        font-size: 11px;
        padding: 35px 0 10px 0;
    }


    /* ========================================================
       MOBILE
    ======================================================== */

    @media (max-width: 768px) {

        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .hero {
            padding: 30px 22px;
        }

        .hero-title {
            font-size: 30px;
        }

        .hero-description {
            font-size: 14px;
        }
    }

</style>
""")


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    # --------------------------------------------------------
    # BRAND
    # --------------------------------------------------------

    st.html("""
    <div class="brand">

        <div class="brand-row">

            <div class="brand-icon">
                💼
            </div>

            <div>

                <div class="brand-title">
                    CareerChat
                </div>

                <div class="brand-subtitle">
                    AI Resume Assistant
                </div>

            </div>

        </div>

    </div>
    """)


    # --------------------------------------------------------
    # RESUME
    # --------------------------------------------------------

    st.html("""
    <div class="sidebar-heading">
        📄 Resume
    </div>
    """)


    uploaded_file = st.file_uploader(
        "Upload your resume",
        type=["pdf", "docx"],
        help="Upload a PDF or DOCX resume.",
        label_visibility="visible",
    )


    st.caption("Maximum file size: 200MB")
    st.caption("Supported formats: PDF, DOCX")


    # --------------------------------------------------------
    # SUGGESTED QUESTIONS
    # --------------------------------------------------------

    st.html("""
    <div style="height:25px;"></div>

    <div class="sidebar-heading">
        💡 Try asking
    </div>

    <div class="sidebar-help">
        Questions you can ask about your resume:
    </div>
    """)


    suggestion_questions = [

        "What skills does the candidate have?",

        "What is the current job title?",

        "What companies has the candidate worked for?",

        "What machine learning algorithms are mentioned?",

    ]


    for i, suggestion in enumerate(suggestion_questions):

        if st.button(
            suggestion,
            key=f"suggestion_{i}",
            use_container_width=True,
        ):

            # Store clicked question
            st.session_state.pending_question = suggestion

            # Rerun page
            st.rerun()


# ============================================================
# HERO
# ============================================================

st.html("""
<div class="hero">

    <div class="hero-badge">
        ✦ AI-POWERED RESUME ASSISTANT
    </div>

    <div class="hero-title">
        Your Resume, <span>Ready to Chat.</span>
    </div>

    <div class="hero-description">
        Upload your resume and ask questions about your
        skills, experience, projects, education and career
        background.
    </div>

</div>
""")


# ============================================================
# PROCESS RESUME
# ============================================================

if uploaded_file is not None:

    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)

    file_path = upload_dir / uploaded_file.name


    # --------------------------------------------------------
    # SAVE UPLOADED FILE
    # --------------------------------------------------------

    with open(file_path, "wb") as file:

        file.write(
            uploaded_file.getbuffer()
        )


    # ========================================================
    # PROCESS ONLY NEW RESUME
    # ========================================================

    if st.session_state.resume_name != uploaded_file.name:

        st.session_state.processed = False
        st.session_state.index = None
        st.session_state.chunks = []
        st.session_state.messages = []

        try:

            with st.spinner(
                "Analyzing your resume..."
            ):

                # --------------------------------------------
                # 1. EXTRACT RESUME TEXT
                # --------------------------------------------

                resume_text = extract_resume_text(
                    str(file_path)
                )


                if not resume_text or not resume_text.strip():

                    raise ValueError(
                        "Could not extract any text from the resume."
                    )


                # --------------------------------------------
                # 2. CLEAN TEXT
                # --------------------------------------------

                cleaned_text = clean_text(
                    resume_text
                )


                if not cleaned_text.strip():

                    raise ValueError(
                        "Resume text is empty after cleaning."
                    )


                # --------------------------------------------
                # 3. CREATE CHUNKS
                # --------------------------------------------

                chunks = chunk_text(
                    cleaned_text
                )


                if not chunks:

                    raise ValueError(
                        "No text chunks were created."
                    )


                # --------------------------------------------
                # 4. CREATE EMBEDDINGS
                # --------------------------------------------

                embeddings = create_embeddings(
                    chunks
                )


                if embeddings is None:

                    raise ValueError(
                        "Failed to create embeddings."
                    )


                # --------------------------------------------
                # 5. CREATE FAISS VECTOR STORE
                # --------------------------------------------

                index = create_vector_store(
                    embeddings
                )


                if index is None:

                    raise ValueError(
                        "Failed to create vector store."
                    )


                # --------------------------------------------
                # 6. SAVE IN SESSION STATE
                # --------------------------------------------

                st.session_state.index = index

                st.session_state.chunks = chunks

                st.session_state.resume_name = (
                    uploaded_file.name
                )

                st.session_state.processed = True


            # =================================================
            # SUCCESS MESSAGE
            # =================================================

            st.html(f"""
            <div class="resume-ready">

                ✓ <b>{uploaded_file.name}</b> is ready.

                You can now ask questions about the resume.

            </div>
            """)


        except Exception as e:

            st.error(
                f"Unable to process resume: {str(e)}"
            )


# ============================================================
# CHAT SECTION
# ============================================================

if st.session_state.processed:

    # --------------------------------------------------------
    # CHAT HEADER
    # --------------------------------------------------------

    st.html("""
    <div class="chat-container">

        <div class="chat-heading">
            💬 Ask CareerChat
        </div>

        <div class="chat-subheading">
            Ask anything about the uploaded resume.
        </div>

    </div>
    """)


    # ========================================================
    # CHAT HISTORY
    # ========================================================

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )


    # ========================================================
    # PUT SIDEBAR QUESTION INTO INPUT
    # ========================================================

    if st.session_state.pending_question:

        st.session_state.question_input = (
            st.session_state.pending_question
        )

        st.session_state.pending_question = ""


    # ========================================================
    # CHAT INPUT
    # ========================================================

    with st.form(
        "chat_form",
        clear_on_submit=True,
    ):

        col1, col2 = st.columns(
            [8, 1]
        )


        # ----------------------------------------------------
        # TEXT INPUT
        # ----------------------------------------------------

        with col1:

            question = st.text_input(
                "Ask something about your resume...",
                key="question_input",
                label_visibility="collapsed",
                placeholder="Ask something about your resume...",
            )


        # ----------------------------------------------------
        # SEND BUTTON
        # ----------------------------------------------------

        with col2:

            st.markdown(
                '<div class="send-button">',
                unsafe_allow_html=True,
            )

            send_clicked = st.form_submit_button(
                "↑",
                use_container_width=True,
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )


    # ========================================================
    # PROCESS QUESTION
    # ========================================================

    if send_clicked and question:

        question = question.strip()


        if question:

            # ------------------------------------------------
            # SAVE USER QUESTION
            # ------------------------------------------------

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": question,
                }
            )


            # ------------------------------------------------
            # DISPLAY USER QUESTION
            # ------------------------------------------------

            with st.chat_message("user"):

                st.markdown(
                    question
                )


            # ------------------------------------------------
            # GENERATE AI ANSWER
            # ------------------------------------------------

            with st.chat_message("assistant"):

                with st.spinner(
                    "Searching your resume..."
                ):

                    try:

                        # ====================================
                        # 1. RETRIEVE RELEVANT CHUNKS
                        # ====================================

                        results = retrieve_relevant_chunks(

                            index=st.session_state.index,

                            chunks=st.session_state.chunks,

                            query=question,

                            top_k=3,
                        )


                        # ====================================
                        # 2. NO RESULTS
                        # ====================================

                        if not results:

                            answer = (
                                "I could not find that information "
                                "in the resume."
                            )


                        # ====================================
                        # 3. RESULTS FOUND
                        # ====================================

                        else:

                            context_parts = []


                            for result in results:

                                chunk = result.get(
                                    "chunk",
                                    "",
                                )


                                if chunk:

                                    context_parts.append(
                                        chunk
                                    )


                            context = "\n\n".join(
                                context_parts
                            )


                            # =================================
                            # 4. GENERATE LLM RESPONSE
                            # =================================

                            answer = generate_answer(
                                context,
                                question,
                            )


                            # =================================
                            # 5. SAFETY CHECK
                            # =================================

                            if answer is None:

                                answer = (
                                    "I could not generate an answer."
                                )


                            answer = str(
                                answer
                            ).strip()


                    except Exception as e:

                        answer = (
                            f"Error generating answer: {str(e)}"
                        )


                # ------------------------------------------------
                # DISPLAY ANSWER
                # ------------------------------------------------

                st.markdown(
                    answer
                )


            # ------------------------------------------------
            # SAVE ASSISTANT RESPONSE
            # ------------------------------------------------

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )


# ============================================================
# EMPTY STATE
# ============================================================

else:

    st.html("""
    <div class="empty-card">

        <div class="empty-icon">
            📄
        </div>

        <div class="empty-title">
            Upload your resume to get started
        </div>

        <div class="empty-text">

            Upload a PDF or DOCX resume using the sidebar.

            <br>

            CareerChat will analyze it and make it searchable.

        </div>

    </div>
    """)


# ============================================================
# FOOTER
# ============================================================

st.html("""
<div class="footer">
    CareerChat AI &nbsp;•&nbsp; Resume Intelligence Assistant
</div>
""")