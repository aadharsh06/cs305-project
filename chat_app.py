
# Main front end - Streamlit UI

# Imported libraries

from __future__ import annotations

import streamlit as st

import importlib.util
import html as html_lib
import json
import re
import sys
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from uuid import uuid4

import markdown
from fpdf import FPDF

# Resolve paths

ROOT = Path(__file__).resolve().parent
CHAT_DIR = ROOT / "chats"
AGENT_PATH = ROOT / "cs305-llm-rag.py"

# Use global address

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load agents

def load_agent_module():
    module_name = "requirements_rag_agent"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, AGENT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

@st.cache_resource
def get_agent_resources():
    agent = load_agent_module()
    client = agent.create_client()
    return agent, client

def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()

def chat_path(chat_id: str) -> Path:
    return CHAT_DIR / f"{chat_id}.json"


def create_chat() -> dict:
    chat = {
        "id": uuid4().hex,
        "title": "New chat",
        "created_at": timestamp(),
        "updated_at": timestamp(),
        "messages": [],
    }
    save_chat(chat)
    return chat


def save_chat(chat: dict) -> None:
    CHAT_DIR.mkdir(parents=True, exist_ok=True)
    path = chat_path(chat["id"])
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(chat, ensure_ascii=False, indent=2) + "\n")
    temporary_path.replace(path)


def load_chat(chat_id: str) -> dict:
    with chat_path(chat_id).open(encoding="utf-8") as chat_file:
        return json.load(chat_file)


def list_chats() -> list[dict]:
    CHAT_DIR.mkdir(parents=True, exist_ok=True)
    chats = []
    for path in CHAT_DIR.glob("*.json"):
        try:
            chats.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(chats, key=lambda chat: chat.get("updated_at", ""), reverse=True)


def message_agent(message: dict) -> str:
    return message.get("agent", "requirements")


def latest_requirements_srs(chat: dict) -> str | None:
    for message in reversed(chat.get("messages", [])):
        if message.get("role") == "assistant" and message_agent(message) == "requirements":
            return message.get("content")
    return None


def latest_sdlc_recommendation(chat: dict) -> str | None:
    for message in reversed(chat.get("messages", [])):
        if message.get("role") == "assistant" and message_agent(message) == "sdlc":
            return message.get("content")
    return None


class MarkdownTableParser(HTMLParser):
    """Read table text so it can be rendered without nested HTML table cells."""

    def __init__(self):
        super().__init__()
        self.rows = []
        self.row = None
        self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        elif tag in {"td", "th"}:
            self.cell = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in {"td", "th"} and self.cell is not None:
            self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None


def flatten_markdown_tables(html: str) -> str:
    def replace_table(match):
        parser = MarkdownTableParser()
        parser.feed(match.group(0))
        if not parser.rows:
            return ""

        headings, *rows = parser.rows
        if not rows:
            rows = [headings]
            headings = [f"Column {index}" for index in range(1, len(headings) + 1)]

        paragraphs = []
        for row in rows:
            fields = [
                f"<b>{html_lib.escape(heading)}:</b> {html_lib.escape(value)}"
                for heading, value in zip(headings, row)
                if value
            ]
            if fields:
                paragraphs.append(f"<p>{'; '.join(fields)}</p>")
        return "".join(paragraphs)

    return re.sub(r"<table\b[^>]*>.*?</table>", replace_table, html, flags=re.DOTALL)


def markdown_for_pdf(srs_markdown: str) -> str:
    html = markdown.markdown(
        srs_markdown,
        extensions=["tables", "fenced_code", "sane_lists"],
    )
    html = (
        html.replace("<strong>", "<b>")
        .replace("</strong>", "</b>")
        .replace("<em>", "<i>")
        .replace("</em>", "</i>")
        .replace("<del>", "<s>")
        .replace("</del>", "</s>")
    )
    return flatten_markdown_tables(html)


@st.cache_data
def render_srs_pdf(srs_markdown: str) -> bytes:
    html = markdown_for_pdf(srs_markdown)
    html = flatten_markdown_tables(html)
    pdf = FPDF()
    pdf.set_margins(18, 16, 18)
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_page()
    pdf.set_title("Software Requirements Specification")
    font_dir = Path("/usr/share/fonts/truetype/dejavu")
    font_files = {
        "": font_dir / "DejaVuSans.ttf",
        "B": font_dir / "DejaVuSans-Bold.ttf",
        "I": font_dir / "DejaVuSans-Oblique.ttf",
        "BI": font_dir / "DejaVuSans-BoldOblique.ttf",
    }
    if all(path.is_file() for path in font_files.values()):
        for style, path in font_files.items():
            pdf.add_font("DejaVu", style=style, fname=str(path))
        font_family = "DejaVu"
    else:
        font_family = "Helvetica"
        srs_markdown = srs_markdown.encode("cp1252", errors="replace").decode("cp1252")
        html = markdown_for_pdf(srs_markdown)
    pdf.set_font(font_family, size=10)
    pdf.write_html(html, font_family=font_family)
    return bytes(pdf.output())


def history_for_agent(chat: dict, agent_name: str) -> list[dict[str, str]]:
    return [
        message
        for message in chat.get("messages", [])
        if message_agent(message) == agent_name
        and message.get("role") in {"user", "assistant"}
    ]


def select_agent(agent_mode_key: str, agent_name: str) -> None:
    st.session_state[agent_mode_key] = agent_name


st.set_page_config(page_title="Software Engineering Agents", layout="wide")
st.title("Software Engineering Agents")
st.caption(
    "Generate and verify requirements and SDLC recommendations. Replies use documents "
    "indexed in Kaggle and the recent conversation for the selected agent."
)

available_chats = list_chats()
if not available_chats:
    available_chats = [create_chat()]
if st.session_state.get("active_chat_id") not in {
    chat["id"] for chat in available_chats
}:
    st.session_state.active_chat_id = available_chats[0]["id"]

with st.sidebar:
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] {
            gap: 0 !important;
        }
        section[data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"] {
            margin: 0 !important;
            padding-block: 0 !important;
            min-height: 1.6rem;
        }
        section[data-testid="stSidebar"] [data-testid="stRadio"] {
            margin-block: -0.55rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.header("Chats")
    if st.button("＋ New chat", use_container_width=True):
        st.session_state.active_chat_id = create_chat()["id"]
        st.rerun()

    chat_ids = [chat["id"] for chat in available_chats]
    titles = {chat["id"]: chat.get("title", "New chat") for chat in available_chats}
    selected_chat_id = st.selectbox(
        "Conversation",
        options=chat_ids,
        index=chat_ids.index(st.session_state.active_chat_id),
        format_func=lambda chat_id: titles.get(chat_id, "New chat"),
        label_visibility="collapsed",
    )
    st.session_state.active_chat_id = selected_chat_id
    st.caption("Each conversation is saved separately in `chats/`.")

    confirm_delete_key = f"confirm_delete_{selected_chat_id}"
    selected_chat = next(
        (chat for chat in available_chats if chat["id"] == selected_chat_id),
        None,
    )
    can_delete_chat = bool(selected_chat and selected_chat.get("messages"))
    if not can_delete_chat:
        st.session_state[confirm_delete_key] = False
        st.button(
            "Delete selected chat",
            key=f"delete_{selected_chat_id}",
            disabled=True,
            help="Add at least one message before deleting this chat.",
        )
    elif not st.session_state.get(confirm_delete_key, False):
        if st.button("Delete selected chat", key=f"delete_{selected_chat_id}"):
            st.session_state[confirm_delete_key] = True
            st.rerun()
    else:
        st.warning("Delete this conversation and its saved messages?")
        confirm_col, cancel_col = st.columns(2)
        if confirm_col.button("Delete", key=f"confirm_{selected_chat_id}"):
            selected_path = chat_path(selected_chat_id)
            if selected_path.is_file():
                selected_path.unlink()
            st.session_state.pop(confirm_delete_key, None)
            remaining_chats = [
                chat for chat in available_chats if chat["id"] != selected_chat_id
            ]
            st.session_state.active_chat_id = (
                remaining_chats[0]["id"] if remaining_chats else create_chat()["id"]
            )
            st.rerun()
        if cancel_col.button("Cancel", key=f"cancel_{selected_chat_id}"):
            st.session_state[confirm_delete_key] = False
            st.rerun()

    chat = load_chat(selected_chat_id)
    requirements_srs = latest_requirements_srs(chat)
    sdlc_recommendation = latest_sdlc_recommendation(chat)
    has_generated_requirements = bool(requirements_srs)
    available_agents = ["Requirements Agent"]
    if has_generated_requirements:
        available_agents.extend(["Requirements Verifier", "SDLC Agent"])
    if sdlc_recommendation:
        available_agents.append("SDLC Verifier")
    agent_mode_key = f"agent_mode_{chat['id']}"
    if st.session_state.get(agent_mode_key) not in available_agents:
        st.session_state[agent_mode_key] = "Requirements Agent"

    st.markdown("**Main agents**")
    generation_agents = {
        "Requirements Agent": True,
        "SDLC Agent": has_generated_requirements,
    }
    with st.container(key="generation-agent-radios", gap=0):
        for agent_name, enabled in generation_agents.items():
            radio_key = f"agent_radio_{chat['id']}_{agent_name}"
            st.session_state[radio_key] = (
                agent_name if st.session_state[agent_mode_key] == agent_name else None
            )
            st.radio(
                agent_name,
                options=[agent_name],
                key=radio_key,
                index=None,
                label_visibility="collapsed",
                disabled=not enabled,
                on_change=select_agent,
                args=(agent_mode_key, agent_name),
            )

    st.markdown("**Verifier agents**")
    verifier_agents = {
        "Requirements Verifier": has_generated_requirements,
        "SDLC Verifier": bool(sdlc_recommendation),
    }
    with st.container(key="verifier-agent-radios", gap=0):
        for agent_name, enabled in verifier_agents.items():
            radio_key = f"agent_radio_{chat['id']}_{agent_name}"
            st.session_state[radio_key] = (
                agent_name if st.session_state[agent_mode_key] == agent_name else None
            )
            st.radio(
                agent_name,
                options=[agent_name],
                key=radio_key,
                index=None,
                label_visibility="collapsed",
                disabled=not enabled,
                on_change=select_agent,
                args=(agent_mode_key, agent_name),
            )
    st.caption(f"Selected for the next prompt: **{st.session_state[agent_mode_key]}**")

    st.markdown("---")
    st.subheader("SRS PDF")
    pdf_choice = st.radio(
        "Export the latest Requirements Agent response",
        options=["Do not generate", "Generate PDF"],
        key=f"srs_pdf_{chat['id']}",
        disabled=not has_generated_requirements,
        help=(
            "Generate a requirements response before exporting its SRS."
            if not has_generated_requirements
            else "The PDF always uses the newest Requirements Agent response in this chat."
        ),
    )
    if pdf_choice == "Generate PDF" and requirements_srs:
        pdf_name = re.sub(r"[^A-Za-z0-9._-]+", "_", chat.get("title", "SRS")).strip("._")
        try:
            pdf_data = render_srs_pdf(requirements_srs)
            st.download_button(
                "Download latest SRS PDF",
                data=pdf_data,
                file_name=f"{pdf_name or 'SRS'}-requirements.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as error:
            st.error(f"Could not create the SRS PDF: {error}")
    elif not has_generated_requirements:
        st.caption("Available after the Requirements Agent finishes a response.")

for message in chat.get("messages", []):
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            label = {
                "requirements": "Requirements Agent",
                "requirements_verifier": "Requirements Verifier",
                "sdlc": "SDLC Agent",
                "sdlc_verifier": "SDLC Verifier",
            }.get(message_agent(message), "Requirements Agent")
            st.caption(label)
        st.markdown(message["content"])

try:
    with st.spinner("Connecting to the Kaggle model..."):
        agent, client = get_agent_resources()
except Exception as error:
    st.error(f"Could not load the RAG agent: {error}")
    st.stop()

selected_agent = st.session_state[agent_mode_key]
agent_names = {
    "Requirements Agent": "requirements",
    "Requirements Verifier": "requirements_verifier",
    "SDLC Agent": "sdlc",
    "SDLC Verifier": "sdlc_verifier",
}
agent_name = agent_names[selected_agent]
input_prompts = {
    "Requirements Agent": "Describe the scenario or ask a follow-up",
    "Requirements Verifier": "Ask the verifier to review the latest SRS",
    "SDLC Agent": "Ask for an SDLC recommendation or a follow-up",
    "SDLC Verifier": "Ask the verifier to review the latest SDLC recommendation",
}
user_text = st.chat_input(input_prompts[selected_agent])
if user_text:
    prior_messages = history_for_agent(chat, agent_name)
    chat["messages"].append(
        {
            "role": "user",
            "content": user_text,
            "agent": agent_name,
            "created_at": timestamp(),
        }
    )
    if chat.get("title") == "New chat":
        chat["title"] = " ".join(user_text.split())[:60] or "New chat"
    chat["updated_at"] = timestamp()
    save_chat(chat)

    with st.chat_message("user"):
        st.markdown(user_text)

    with st.chat_message("assistant"):
        st.caption(selected_agent)
        response_slot = st.empty()
        response_generated = False
        with st.spinner("Retrieving sources and generating a response..."):
            try:
                if agent_name == "requirements":
                    response = agent.generate_xai_requirements(
                        user_text,
                        client,
                        history=prior_messages,
                        on_response=response_slot.markdown,
                    )
                elif agent_name == "sdlc":
                    if not requirements_srs:
                        raise RuntimeError(
                            "Generate a requirements SRS before requesting an SDLC recommendation."
                        )
                    response = agent.generate_sdlc_recommendation(
                        requirements_srs,
                        user_text,
                        client,
                        history=prior_messages,
                        on_response=response_slot.markdown,
                    )
                elif agent_name == "requirements_verifier":
                    if not requirements_srs:
                        raise RuntimeError("Generate a requirements SRS before verifying it.")
                    response = agent.verify_requirements(
                        requirements_srs,
                        user_text,
                        client,
                        history=prior_messages,
                        on_response=response_slot.markdown,
                    )
                else:
                    if not requirements_srs or not sdlc_recommendation:
                        raise RuntimeError(
                            "Generate requirements and an SDLC recommendation before verifying the SDLC."
                        )
                    response = agent.verify_sdlc_recommendation(
                        sdlc_recommendation,
                        user_text,
                        client,
                        history=prior_messages,
                        on_response=response_slot.markdown,
                    )
                chat["messages"].append(
                    {
                        "role": "assistant",
                        "content": response,
                        "agent": agent_name,
                        "created_at": timestamp(),
                    }
                )
                chat["updated_at"] = timestamp()
                save_chat(chat)
                response_generated = True
            except Exception as error:
                st.error(f"Generation failed: {error}")
        if response_generated:
            st.rerun()
