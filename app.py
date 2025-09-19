"""Web interface using Streamlit."""

import tempfile
from pathlib import Path

import streamlit as st

from app import ConversationManager, RAGPipeline
from app.config import config

CONFIDENCE_HIGH = 0.6
CONFIDENCE_MEDIUM = 0.2

MAX_CONTEXT_PREVIEW_LENGTH = 200

config.setup_logging()
logger = config.get_logger(__name__)


class SessionState:
    """Centralized session state management."""

    @staticmethod
    def initialize() -> None:
        """Initialize all session state variables."""
        defaults = {
            "rag_pipeline": None,
            "conversation_manager": None,
            "document_processed": False,
            "system_ready": False,
            "current_result": None,
        }

        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

    @staticmethod
    def reset_system() -> None:
        """Reset system components when configuration changes."""
        st.session_state.rag_pipeline = None
        st.session_state.conversation_manager = None
        st.session_state.document_processed = False
        st.session_state.system_ready = False
        st.session_state.current_result = None

    @staticmethod
    def is_system_ready() -> bool:
        """Check if the system is properly initialized.

        Returns:
            bool: True if both rag_pipeline and conversation_manager are initialized,
            False otherwise.
        """
        return (
            st.session_state.get("rag_pipeline") is not None
            and st.session_state.get("conversation_manager") is not None
        )


def validate_configuration() -> bool:
    """Validate application configuration and show user feedback.

    Returns:
        bool: True if configuration is valid, False otherwise.
    """
    try:
        config.validate()
    except ValueError as e:
        st.error(f"Configuration Error: {e}")
        return False
    else:
        return True


def initialize_system() -> bool:
    """Initialize the RAG pipeline and conversation manager.

    Returns:
        bool: True if initialization succeeds, False otherwise.
    """
    try:
        with st.spinner("Initializing system..."):
            st.session_state.rag_pipeline = RAGPipeline()
            st.session_state.conversation_manager = ConversationManager(
                st.session_state.rag_pipeline
            )
            st.session_state.system_ready = True

        logger.info("RAG system initialized successfully")
        st.success("System initialized successfully!")

    except (ValueError, RuntimeError) as e:
        logger.exception("Failed to initialize system")
        st.error(f"Failed to initialize system: {e}")
        return False
    else:
        return True


def process_document(uploaded_file) -> bool:  # noqa: ANN001
    """Process uploaded file through RAG pipeline.

    Returns:
        bool: True if document processing succeeds, False otherwise.
    """
    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=f".{uploaded_file.name.split('.')[-1]}",
        ) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_file_path = Path(tmp_file.name)

        with st.spinner(
            f"Processing '{uploaded_file.name}'... This may take a few moments."
        ):
            st.session_state.rag_pipeline.process_document(tmp_file_path)

        tmp_file_path.unlink()

        st.session_state.document_processed = True
        st.success(f"Document '{uploaded_file.name}' processed successfully!")
        st.info("You can now ask questions about your document below.")

    except (OSError, ValueError, RuntimeError) as e:
        logger.exception("Document processing failed")
        st.error(f"Failed to process document: {e}")
        return False
    else:
        return True


def render_sidebar() -> None:
    """Render the sidebar with configuration and system status."""
    with st.sidebar:
        st.header("System Configuration")

        if (
            st.button("Initialize System", use_container_width=True)
            and validate_configuration()
            and initialize_system()
        ):
            st.rerun()

        st.divider()
        st.subheader("System Status")
        config_status = "Valid" if validate_configuration() else "Invalid"
        st.write(f"**Configuration:** {config_status}")
        if SessionState.is_system_ready():
            st.write("**System:** Ready")
        else:
            st.write("**System:** Not Initialized")
        if st.session_state.document_processed:
            st.write("**Document:** Loaded")
        else:
            st.write("**Document:** None")

        st.divider()
        if SessionState.is_system_ready():
            st.subheader("Conversation")
            if (
                st.button("Clear History", use_container_width=True)
                and st.session_state.conversation_manager
            ):
                st.session_state.conversation_manager.clear_history()
                st.session_state.current_result = None
                st.success("Conversation cleared!")
                st.rerun()


def render_document_upload() -> None:
    """Render document upload section."""
    st.header("Document Upload")
    uploaded_file = st.file_uploader(
        "Upload a PDF or TXT document",
        type=["pdf", "txt"],
        help="Upload a document to start asking questions about it",
    )
    if (
        uploaded_file
        and not st.session_state.document_processed
        and st.button("Process Document", use_container_width=True)
        and process_document(uploaded_file)
    ):
        st.rerun()


def render_chat_interface() -> None:
    """Render the main chat interface."""
    if not st.session_state.document_processed:
        return

    st.header("Ask Questions About Your Document")
    question = st.text_input(
        "Your Question:",
        placeholder="Ask anything about your uploaded document...",
        help="Ask questions about the content of your uploaded document",
    )

    button_clicked = st.button("Ask Question", use_container_width=True)

    # Process new question if button was clicked
    if button_clicked and question.strip():
        if not st.session_state.conversation_manager:
            st.error("System not properly initialized. Please reinitialize.")
            return

        with st.spinner("Processing..."):
            try:
                st.session_state.current_result = (
                    st.session_state.conversation_manager.answer_question(question)
                )
            except (ValueError, RuntimeError, OSError) as e:
                logger.exception("Question processing failed")
                st.error(f"Failed to process question: {e}")
                return

    # Display result if available
    if st.session_state.current_result:
        result = st.session_state.current_result

        st.subheader("Answer:")
        st.write(result["answer"])

        col1, col2 = st.columns(2)
        with col1:
            confidence = result.get("confidence", 0.5)
            confidence_color = (
                "green"
                if confidence > CONFIDENCE_HIGH
                else "orange"
                if confidence > CONFIDENCE_MEDIUM
                else "red"
            )
            st.markdown(f"**Confidence:** :{confidence_color}[{confidence:.2f}]")

        with col2:
            if "standalone_query" in result:
                st.markdown(f"**Search Query:** {result['standalone_query']}")

        if st.checkbox("Show Retrieved Contexts (Debug)"):
            st.subheader("Retrieved Document Sections:")
            for i, (chunk, score) in enumerate(result["retrieved_contexts"]):
                with st.expander(
                    (
                        f"Context {i + 1} - Similarity: {score:.4f} - "
                        f"Source: {chunk.metadata['source']}"
                    ),
                    expanded=False,
                ):
                    st.code(chunk.content)


def render_conversation_history() -> None:
    """Render conversation history in a clean format."""
    if not (
        st.session_state.conversation_manager
        and st.session_state.conversation_manager.conversation_history
    ):
        return

    st.markdown("---")
    st.subheader("Conversation History")

    for i, turn in enumerate(
        st.session_state.conversation_manager.conversation_history
    ):
        with st.expander(
            f"Turn {i + 1}: {turn.user_question[:50]}...",
            expanded=False,
        ):
            st.write("**Your Question:**")
            st.write(turn.user_question)

            st.write("**Assistant's Response:**")
            st.write(turn.bot_response)

            st.write("**Retrieved Context (Debug Info):**")
            for j, (chunk, score) in enumerate(turn.retrieved_contexts):
                st.write(
                    f"Context {j + 1} (Similarity: {score:.4f}) from "
                    f"{chunk.metadata['source']}:",
                )
                st.code(
                    chunk.content[:MAX_CONTEXT_PREVIEW_LENGTH] + "..."
                    if len(chunk.content) > MAX_CONTEXT_PREVIEW_LENGTH
                    else chunk.content,
                )


def render_system_info() -> None:
    """Render system information footer."""
    st.markdown("---")
    st.subheader("System Overview")

    if SessionState.is_system_ready():
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Embedding Model**")
            st.markdown(f"**{config.EMBEDDING_MODEL}**")

        with col2:
            st.markdown("**Chunk Size**")
            st.markdown(f"**{config.CHUNK_SIZE}**")

        with col3:
            conversation_count = 0
            if (
                st.session_state.conversation_manager
                and st.session_state.conversation_manager.conversation_history
            ):
                conversation_count = len(
                    st.session_state.conversation_manager.conversation_history
                )

            st.markdown("**Conversations**")
            st.markdown(f"**{conversation_count}**")

    with st.expander("How This RAG System Works", expanded=False):
        st.markdown("""
        **Retrieval-Augmented Generation (RAG)** combines information retrieval
        with language generation:

        1. **Document Processing**: Text is split into chunks and converted to
           vector embeddings
        2. **Query Processing**: Your questions are converted to vectors for
           similarity search
        3. **Retrieval**: Most relevant document sections are found using
           cosine similarity
        4. **Generation**: OpenAI's GPT model generates answers using
           retrieved context
        5. **Context Awareness**: Conversation history helps maintain context
           across questions

        **Native Implementation**: This system uses NumPy for vector operations
        and SQLite for storage, demonstrating RAG principles without heavy
        dependencies like ChromaDB or Pinecone.
        """)

    st.markdown(
        (
            "<div style='text-align: center; color: gray; font-size: small; "
            "margin-top: 20px;'>\n"
            "LCEngine v0.1 - Native RAG System MVP<br>\n"
            "</div>\n"
        ),
        unsafe_allow_html=True,
    )


def main() -> None:
    """Main entry point for the Streamlit web application.

    Sets up the page configuration, initializes session state, renders the
    sidebar, and orchestrates the document upload, chat interface, conversation
    history, and system info.
    """
    st.set_page_config(
        page_title="LCEngine v0.1 - Native RAG System",
        layout="wide",
    )

    SessionState.initialize()

    st.title("LCEngine v0.1 - Native RAG System MVP")
    st.markdown("---")

    render_sidebar()

    if not SessionState.is_system_ready():
        st.info("Please initialize the system using the sidebar to get started.")
        return

    render_document_upload()
    render_chat_interface()
    render_conversation_history()
    render_system_info()


if __name__ == "__main__":
    main()
