import streamlit as st
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="IPC Legal Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROFESSIONAL MILD THEME
# ============================================================

st.markdown(
    """
    <style>

    /* --------------------------------------------------------
       MAIN APPLICATION BACKGROUND
       -------------------------------------------------------- */

    .stApp {
        background:
            linear-gradient(
                135deg,
                #e8eef3 0%,
                #dfe8ee 50%,
                #e9eef2 100%
            );
    }


    /* --------------------------------------------------------
       MAIN CONTENT WIDTH
       -------------------------------------------------------- */

    .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* --------------------------------------------------------
       TITLE
       -------------------------------------------------------- */

    h1 {
        color: #1d3449 !important;
        font-size: 42px !important;
        font-weight: 800 !important;
        letter-spacing: -1px;
    }


    h2, h3 {
        color: #29465e !important;
    }


    /* --------------------------------------------------------
       SUBTITLE
       -------------------------------------------------------- */

    .subtitle {
        color: #607386;
        font-size: 16px;
        margin-top: -12px;
        margin-bottom: 18px;
    }


    /* --------------------------------------------------------
       TECHNOLOGY BADGE
       -------------------------------------------------------- */

    .tech-badge {
        background: #e6ddbc;
        color: #665523;
        border: 1px solid #cbbd8c;
        border-radius: 20px;
        padding: 7px 14px;
        font-size: 12px;
        font-weight: 700;
        display: inline-block;
        margin-bottom: 18px;
    }


    /* --------------------------------------------------------
       DISCLAIMER
       -------------------------------------------------------- */

    .notice-box {
        background: #f1ead1;
        border-left: 5px solid #c5a84d;
        border-radius: 12px;
        padding: 15px 18px;
        margin-bottom: 22px;
        color: #5b5033;
        font-size: 13px;
        line-height: 1.6;
    }


    /* --------------------------------------------------------
       FEATURE CARDS
       -------------------------------------------------------- */

    .feature-card {
        background: #e7edf2;
        border: 1px solid #cbd6df;
        border-radius: 15px;
        padding: 18px;
        min-height: 125px;
        box-shadow: 0 5px 15px rgba(40, 60, 80, 0.06);
    }


    .feature-icon {
        font-size: 26px;
        margin-bottom: 7px;
    }


    .feature-title {
        color: #29465e;
        font-size: 15px;
        font-weight: 800;
        margin-bottom: 5px;
    }


    .feature-text {
        color: #657687;
        font-size: 12px;
        line-height: 1.5;
    }


    /* --------------------------------------------------------
       CHAT AREA
       -------------------------------------------------------- */

    [data-testid="stChatMessage"] {
        border-radius: 14px;
        margin-bottom: 8px;
    }


    /* --------------------------------------------------------
       CHAT INPUT
       -------------------------------------------------------- */

    [data-testid="stChatInput"] textarea {
        background: #edf2f5 !important;
        border: 1px solid #b9c7d2 !important;
        border-radius: 15px !important;
        color: #253746 !important;
    }


    [data-testid="stChatInput"] textarea:focus {
        border: 1px solid #71889d !important;
        box-shadow: 0 0 0 2px rgba(113, 136, 157, 0.12) !important;
    }


    /* --------------------------------------------------------
       SIDEBAR
       -------------------------------------------------------- */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #1d354c,
                #284962,
                #1d354c
            );
    }


    section[data-testid="stSidebar"] * {
        color: #f4f7fa;
    }


    /* --------------------------------------------------------
       SIDEBAR HEADINGS
       -------------------------------------------------------- */

    .sidebar-heading {
        color: #e0c86f;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 1px;
        margin-top: 12px;
        margin-bottom: 7px;
    }


    .sidebar-description {
        color: rgba(255,255,255,0.78);
        font-size: 12px;
        line-height: 1.6;
    }


    /* --------------------------------------------------------
       SIDEBAR STAT BOX
       -------------------------------------------------------- */

    .stat-box {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 8px;
    }


    .stat-number {
        color: #e0c86f;
        font-size: 21px;
        font-weight: 800;
    }


    .stat-label {
        color: rgba(255,255,255,0.70);
        font-size: 11px;
    }


   /* ============================================================
   ATTRACTIVE CHAT SEARCH BAR
   ============================================================ */

[data-testid="stChatInput"] {
    background: transparent !important;
    padding-top: 10px !important;
    padding-bottom: 18px !important;
}

[data-testid="stChatInput"] > div {
    background: #f4f7f9 !important;
    border: 2px solid #b8c7d3 !important;
    border-radius: 22px !important;
    padding: 4px 8px !important;
    box-shadow: 0 8px 22px rgba(42, 62, 80, 0.12) !important;
    transition: all 0.25s ease !important;
}

[data-testid="stChatInput"] > div:focus-within {
    border: 2px solid #8da2b3 !important;
    box-shadow: 0 8px 26px rgba(42, 62, 80, 0.18) !important;
}

[data-testid="stChatInput"] textarea {
    background: transparent !important;
    border: none !important;
    color: #263746 !important;
    font-size: 15px !important;
    padding: 12px 8px !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #7b8b98 !important;
    font-size: 14px !important;
}

[data-testid="stChatInput"] textarea:focus {
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
}

    /* --------------------------------------------------------
       NORMAL BUTTONS
       -------------------------------------------------------- */

    .stButton > button {
        border-radius: 10px !important;
        border: 1px solid #bca557 !important;
        background: #eee5c6 !important;
        color: #34495b !important;
        font-weight: 700 !important;
    }


    .stButton > button:hover {
        background: #d7c477 !important;
        color: #20374c !important;
    }


    /* --------------------------------------------------------
       FOOTER
       -------------------------------------------------------- */

    .footer-text {
        text-align: center;
        color: #657687;
        font-size: 12px;
        padding-top: 15px;
        padding-bottom: 20px;
    }


    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.title("⚖️ IPC Legal Assistant")

st.markdown(
    """
    <div class="subtitle">
        An NLP-powered educational chatbot for exploring
        the Indian Penal Code
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="tech-badge">
        ✦ AI &nbsp;•&nbsp; NLP &nbsp;•&nbsp; TF-IDF &nbsp;•&nbsp;
        COSINE SIMILARITY
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# EDUCATIONAL NOTICE
# ============================================================

st.warning(
    "Educational Notice: This chatbot is developed for "
    "academic and educational purposes. Information is "
    "retrieved from the project's knowledge base and should "
    "not be considered professional legal advice."
)


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    try:

        data = pd.read_csv(
            "data/ipc_dataset.csv"
        )

        data = data.fillna("")

        return data

    except FileNotFoundError:

        st.error(
            "❌ Could not find data/ipc_dataset.csv. "
            "Please make sure the data folder exists."
        )

        st.stop()

    except Exception as error:

        st.error(
            f"❌ Error loading dataset: {error}"
        )

        st.stop()


df = load_dataset()


# ============================================================
# REQUIRED DATASET COLUMNS
# ============================================================

required_columns = [
    "section",
    "title",
    "offence",
    "definition",
    "description",
    "punishment",
    "keywords",
    "related_sections"
]


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    st.error(
        "❌ Missing columns in ipc_dataset.csv: "
        + ", ".join(missing_columns)
    )

    st.stop()


# ============================================================
# CREATE SEARCH TEXT
# ============================================================

df["search_text"] = (
    df["section"].astype(str) + " " +
    df["title"].astype(str) + " " +
    df["offence"].astype(str) + " " +
    df["definition"].astype(str) + " " +
    df["description"].astype(str) + " " +
    df["punishment"].astype(str) + " " +
    df["keywords"].astype(str) + " " +
    df["related_sections"].astype(str)
)


# ============================================================
# TF-IDF MODEL
# ============================================================

@st.cache_resource
def create_tfidf_model(text_data):

    vectorizer = TfidfVectorizer(
        stop_words="english",
        lowercase=True
    )

    matrix = vectorizer.fit_transform(
        text_data
    )

    return vectorizer, matrix


vectorizer, tfidf_matrix = create_tfidf_model(
    tuple(df["search_text"].tolist())
)


# ============================================================
# FIND SECTION NUMBER
# ============================================================

def find_section(question):

    match = re.search(
        r"(?:section|sec\.?)\s*(\d+[A-Za-z]*)",
        question,
        re.IGNORECASE
    )

    if match:

        return match.group(1)

    return None


# ============================================================
# GET SECTION INFORMATION
# ============================================================

def get_section(section_number):

    result = df[
        df["section"].astype(str).str.lower()
        == str(section_number).lower()
    ]

    if result.empty:

        return None

    return result.iloc[0]


# ============================================================
# GENERATE CHATBOT ANSWER
# ============================================================

def get_answer(question):

    question = question.strip()

    if not question:

        return "Please enter a question."


    lower_question = question.lower()


    # ========================================================
    # GREETINGS
    # ========================================================

    greetings = [
        "hi",
        "hello",
        "hey",
        "hii",
        "hiii",
        "good morning",
        "good afternoon",
        "good evening"
    ]


    if lower_question in greetings:

        return """
### 👋 Hello!

Welcome to the **IPC Legal Assistant**.

I can help you explore information about:

📌 IPC Sections  
⚖️ Offences  
📖 Definitions  
🔒 Punishments  
🔗 Related Sections  

Try asking:

**What is Section 302?**
"""


    # ========================================================
    # THANK YOU
    # ========================================================

    if any(
        phrase in lower_question
        for phrase in [
            "thank you",
            "thanks",
            "thank"
        ]
    ):

        return """
### 😊 You're welcome!

Feel free to ask another question
about the IPC knowledge base.
"""


    # ========================================================
    # FIND SECTION FROM QUESTION
    # ========================================================

    section_number = find_section(
        question
    )


    # ========================================================
    # USE PREVIOUS SECTION FOR FOLLOW-UP
    # ========================================================

    if section_number is None:

        previous_section = st.session_state.get(
            "current_section",
            None
        )


        follow_up_words = [
            "punishment",
            "penalty",
            "sentence",
            "punished",
            "tell me more",
            "more information",
            "explain more",
            "more details",
            "definition",
            "define",
            "meaning",
            "offence",
            "offense",
            "crime",
            "related section",
            "related sections",
            "what is this",
            "explain this"
        ]


        if (
            previous_section
            and any(
                word in lower_question
                for word in follow_up_words
            )
        ):

            section_number = previous_section


    # ========================================================
    # DIRECT SECTION SEARCH
    # ========================================================

    if section_number:

        row = get_section(
            section_number
        )


        if row is not None:

            st.session_state.current_section = str(
                row["section"]
            )


            # ------------------------------------------------
            # PUNISHMENT
            # ------------------------------------------------

            if any(
                word in lower_question
                for word in [
                    "punishment",
                    "penalty",
                    "sentence",
                    "punished"
                ]
            ):

                return f"""
### ⚖️ Punishment — IPC Section {row["section"]}

**{row["punishment"]}**
"""


            # ------------------------------------------------
            # DEFINITION
            # ------------------------------------------------

            if any(
                word in lower_question
                for word in [
                    "definition",
                    "define",
                    "meaning"
                ]
            ):

                return f"""
### 📖 Definition — IPC Section {row["section"]}

{row["definition"]}
"""


            # ------------------------------------------------
            # OFFENCE
            # ------------------------------------------------

            if any(
                word in lower_question
                for word in [
                    "offence",
                    "offense",
                    "crime"
                ]
            ):

                return f"""
### ⚖️ Offence — IPC Section {row["section"]}

**{row["offence"]}**
"""


            # ------------------------------------------------
            # RELATED SECTIONS
            # ------------------------------------------------

            if (
                "related" in lower_question
                or "relation" in lower_question
            ):

                return f"""
### 🔗 Related Sections — IPC Section {row["section"]}

**{row["related_sections"]}**
"""


            # ------------------------------------------------
            # MORE INFORMATION
            # ------------------------------------------------

            if any(
                phrase in lower_question
                for phrase in [
                    "tell me more",
                    "more information",
                    "explain more",
                    "more details",
                    "explain this",
                    "what is this"
                ]
            ):

                return f"""
### 📚 More Information — IPC Section {row["section"]}

**Title**  
{row["title"]}

**Offence**  
{row["offence"]}

**Explanation**  
{row["description"]}

**Related Sections**  
{row["related_sections"]}

---

💡 You can continue asking:

• What is the punishment?  
• What is the definition?  
• What is the offence?  
• What are the related sections?
"""


            # ------------------------------------------------
            # NORMAL SECTION RESPONSE
            # ------------------------------------------------

            return f"""
### ⚖️ IPC Section {row["section"]}

**Title**  
{row["title"]}

**Offence**  
{row["offence"]}

**Definition**  
{row["definition"]}

**Description**  
{row["description"]}

**Punishment**  
{row["punishment"]}

**Related Sections**  
{row["related_sections"]}
"""


    # ========================================================
    # NLP SEARCH USING TF-IDF
    # ========================================================

    question_vector = vectorizer.transform(
        [question]
    )


    similarity_scores = cosine_similarity(
        question_vector,
        tfidf_matrix
    ).flatten()


    best_index = similarity_scores.argmax()

    best_score = similarity_scores[best_index]


    # ========================================================
    # NO MATCH
    # ========================================================

    if best_score < 0.15:

        return """
### ❓ No Matching Information Found

I couldn't find a sufficiently relevant answer
in the current IPC knowledge base.

Try asking:

• What is Section 302?
• What is theft?
• What is cheating?
• What is criminal intimidation?
• What is the punishment for theft?
"""


    # ========================================================
    # BEST NLP MATCH
    # ========================================================

    row = df.iloc[
        best_index
    ]


    st.session_state.current_section = str(
        row["section"]
    )


    return f"""
### ⚖️ IPC Section {row["section"]}

**Title**  
{row["title"]}

**Offence**  
{row["offence"]}

**Definition**  
{row["definition"]}

**Description**  
{row["description"]}

**Punishment**  
{row["punishment"]}

**Related Sections**  
{row["related_sections"]}

---

🔎 **NLP Match Score:** `{best_score:.2f}`
"""


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


if "current_section" not in st.session_state:

    st.session_state.current_section = None


# ============================================================
# PROJECT FEATURES
# ============================================================

if not st.session_state.messages:

    st.markdown("## ✨ Welcome to IPC Legal Assistant")

    st.write(
        "Ask questions about the Indian Penal Code using "
        "natural language. The chatbot searches its "
        "knowledge base using NLP techniques."
    )

    st.markdown("### 🚀 Project Features")

    feature1, feature2, feature3 = st.columns(3)

    with feature1:
        st.info(
            """
            ### 🔎 Smart Search

            Uses **TF-IDF** and **Cosine Similarity**
            to find the most relevant IPC information
            from the knowledge base.
            """
        )

    with feature2:
        st.info(
            """
            ### 🧠 NLP Based

            Understands natural-language questions
            about IPC sections, offences, definitions
            and punishments.
            """
        )

    with feature3:
        st.info(
            """
            ### 💬 Conversation Memory

            Remembers the previously discussed IPC
            section and supports follow-up questions.
            """
        )

    st.markdown("### 💡 Example Questions")

    example1, example2 = st.columns(2)

    with example1:

        st.markdown(
            """
            **⚖️ Section Information**

            • What is Section 302?

            • What is Section 379?

            • What is Section 420?
            """
        )

    with example2:

        st.markdown(
            """
            **🔎 Natural Language Questions**

            • What is theft?

            • What is cheating?

            • What is the punishment for theft?
            """
        )

    st.divider()
# ============================================================
# CONVERSATION HEADER
# ============================================================

st.markdown("### 💬 Conversation")


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# CHAT INPUT
# ============================================================

user_question = st.chat_input(
    "Ask me about an IPC section, offence or punishment..."
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if user_question:

    # --------------------------------------------------------
    # USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )


    # --------------------------------------------------------
    # BOT ANSWER
    # --------------------------------------------------------

    answer = get_answer(
        user_question
    )


    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )


    st.rerun()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    # --------------------------------------------------------
    # SIDEBAR HEADER
    # --------------------------------------------------------

    st.markdown(
        "# ⚖️ IPC Assistant"
    )

    st.caption(
        "Educational NLP Chatbot"
    )

    st.divider()


    # --------------------------------------------------------
    # PROJECT
    # --------------------------------------------------------

    st.markdown(
        '<div class="sidebar-heading">PROJECT</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-description">

        <b>Indian Penal Code Educational Chatbot</b>

        <br><br>

        A natural language chatbot that retrieves
        IPC information from a structured knowledge base.

        </div>
        """,
        unsafe_allow_html=True
    )


    st.divider()


    # --------------------------------------------------------
    # TECHNOLOGY
    # --------------------------------------------------------

    st.markdown(
        '<div class="sidebar-heading">TECHNOLOGY</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-description">

        🐍 Python<br>
        🧠 Natural Language Processing<br>
        🔎 TF-IDF<br>
        📐 Cosine Similarity<br>
        🌐 Streamlit<br>
        🗃️ CSV Knowledge Base

        </div>
        """,
        unsafe_allow_html=True
    )


    st.divider()


    # --------------------------------------------------------
    # KNOWLEDGE BASE
    # --------------------------------------------------------

    st.markdown(
        '<div class="sidebar-heading">KNOWLEDGE BASE</div>',
        unsafe_allow_html=True
    )


    stat_col1, stat_col2 = st.columns(2)


    with stat_col1:

        st.markdown(
            f"""
            <div class="stat-box">

                <div class="stat-number">
                    {len(df)}
                </div>

                <div class="stat-label">
                    IPC Records
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with stat_col2:

        st.markdown(
            """
            <div class="stat-box">

                <div class="stat-number">
                    NLP
                </div>

                <div class="stat-label">
                    Search
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    st.divider()


    # --------------------------------------------------------
    # EXAMPLE QUESTIONS
    # --------------------------------------------------------

    st.markdown(
        '<div class="sidebar-heading">💡 TRY ASKING</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-description">

        • What is Section 302?<br>
        • What is theft?<br>
        • What is cheating?<br>
        • What is criminal intimidation?<br>
        • What is the punishment?<br>
        • Tell me more

        </div>
        """,
        unsafe_allow_html=True
    )


    st.divider()


    # --------------------------------------------------------
    # CURRENT SECTION
    # --------------------------------------------------------

    current_section = st.session_state.get(
        "current_section"
    )


    if current_section:

        st.markdown(
            '<div class="sidebar-heading">CURRENT TOPIC</div>',
            unsafe_allow_html=True
        )

        st.write(
            f"⚖️ IPC Section **{current_section}**"
        )


        st.divider()


    # --------------------------------------------------------
    # CLEAR CHAT
    # --------------------------------------------------------

    if st.button(
        "🧹 Clear Conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.session_state.current_section = None

        st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.divider()

footer_col1, footer_col2, footer_col3 = st.columns(
    [1, 2, 1]
)

with footer_col2:

    st.markdown(
        """
        ### ⚖️ IPC Legal Assistant
        """,
        unsafe_allow_html=True
    )

    st.caption(
        "NLP-Based Educational Chatbot"
    )

    st.caption(
        "Developed for academic and educational purposes"
    )