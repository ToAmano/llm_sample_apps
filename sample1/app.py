import streamlit as st
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from typing import List, Dict, Optional, Literal
import uuid
import datetime
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# LLMモデルの種類を定義
ModelType = Literal["gpt-4o", "gemini-2.0-pro", "claude-3-7-sonnet"]


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]
    current_model: ModelType
    system_message: Optional[str]


# LLMモデルを初期化する関数
def get_llm(model_type: ModelType):
    if model_type == "gpt-4o":
        return ChatOpenAI(model="gpt-4o", temperature=0.7)
    elif model_type == "gemini-2.0-pro":
        return ChatGoogleGenerativeAI(model="gemini-2.0-pro", temperature=0.7)
    elif model_type == "claude-3-7-sonnet":
        return ChatAnthropic(model="claude-3-7-sonnet-20250211", temperature=0.7)
    else:
        raise ValueError(f"不明なモデルタイプ: {model_type}")


from typing import List, Dict, Any, TypedDict, Literal, Optional
import os
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

# 必要な環境変数を設定（実際の利用時は.envファイルなどで管理）
# os.environ["OPENAI_API_KEY"] = "your-openai-key"
# os.environ["GOOGLE_API_KEY"] = "your-google-key"
# os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-key"

# LLMモデルの種類を定義
ModelType = Literal["gpt-4o", "gemini-2.0-pro", "claude-3-7-sonnet"]

# チャット状態を表すクラス
class ChatState(TypedDict):
    messages: List[Dict[str, Any]]
    current_model: ModelType
    system_message: Optional[str]

# LLMモデルを初期化する関数
def get_llm(model_type: ModelType):
    if model_type == "gpt-4o":
        return ChatOpenAI(model="gpt-4o", temperature=0.7)
    elif model_type == "gemini-2.0-pro":
        return ChatGoogleGenerativeAI(model="gemini-2.0-pro", temperature=0.7)
    elif model_type == "claude-3-7-sonnet":
        return ChatAnthropic(model="claude-3-7-sonnet-20250211", temperature=0.7)
    else:
        raise ValueError(f"不明なモデルタイプ: {model_type}")

# LLMにメッセージを送信して応答を取得する関数
def query_llm(state: ChatState):
    """現在のモデルを使用してLLMに問い合わせを行う"""
    current_model = state["current_model"]
    llm = get_llm(current_model)
    
    # メッセージ履歴を準備
    messages = []
    
    # システムメッセージがあれば追加
    if state["system_message"]:
        messages.append(SystemMessage(content=state["system_message"]))
    
    # 会話履歴を追加
    for msg in state["messages"]:
        if msg["role"] == "human":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "ai":
            messages.append(AIMessage(content=msg["content"]))
    
    # LLMに問い合わせ
    response = llm.invoke(messages)
    
    # 応答をステートに追加
    new_messages = state["messages"].copy()
    new_messages.append({
        "role": "ai",
        "content": response.content,
        "model": current_model
    })
    
    return {"messages": new_messages, "current_model": current_model, "system_message": state["system_message"]}



# --- Session State ---
if "sessions" not in st.session_state:
    st.session_state["sessions"] = {}

if "current_session" not in st.session_state:
    session_id = str(uuid.uuid4())
    st.session_state["current_session"] = session_id
    st.session_state["sessions"][session_id] = {
        "model": "gpt-3.5-turbo",
        "messages": [],
        "created_at": datetime.datetime.now()
    }

# --- UI Components ---
st.set_page_config(page_title="LangGraph Chat App", layout="wide")

# Sidebar for session and model selection
with st.sidebar:
    st.markdown("### セッション一覧")
    for sid, sess in st.session_state["sessions"].items():
        label = sess["created_at"].strftime("%Y-%m-%d %H:%M:%S")
        if st.button(label, key=sid):
            st.session_state["current_session"] = sid

    if st.button("➕ 新しいセッションを開始"):
        new_id = str(uuid.uuid4())
        st.session_state["current_session"] = new_id
        st.session_state["sessions"][new_id] = {
            "model": "gpt-3.5-turbo",
            "messages": [],
            "created_at": datetime.datetime.now()
        }

    st.markdown("---")
    st.markdown("### モデル選択")
    model = st.selectbox("使用するLLMを選択", [
        "gpt-3.5-turbo", 
        "gpt-4o", 
        "claude-3.5", 
        "claude-3-opus-20240229", 
        "gemini-2.0pro"
    ])
    current_session = st.session_state["current_session"]
    st.session_state["sessions"][current_session]["model"] = model

    st.markdown("### 会話履歴")
    for msg in st.session_state["sessions"][current_session]["messages"]:
        st.markdown(f"**{msg['role'].capitalize()}**: {msg['content']}")

# --- LangGraph Setup ---
def llm_node(state: Dict) -> Dict:
    model_name = state["model"]
    input_text = state["messages"]

    if model_name.startswith("gpt"):
        llm = ChatOpenAI(model=model_name, temperature=0)
        messages = [HumanMessage(content=input_text)]
        response = llm(messages)
        return {"response": response.content}

    elif model_name.startswith("claude"):
        return {"response": f"(Claudeによる応答のモック): {input_text}"}

    elif model_name.startswith("gemini"):
        return {"response": f"(Geminiによる応答のモック): {input_text}"}

    return {"response": "不明なモデルが選択されました。"}

builder = StateGraph(State)
builder.add_node("llm", llm_node)
builder.set_entry_point("llm")
builder.set_finish_point("llm")
chat_graph = builder.compile()

# --- Chat UI ---
st.title("🧠 LangGraph チャットアプリ")

user_input = st.chat_input("メッセージを入力...")
if user_input:
    current_session = st.session_state["current_session"]
    model = st.session_state["sessions"][current_session]["model"]

    st.session_state["sessions"][current_session]["messages"].append({"role": "user", "content": user_input})

    with st.spinner("考え中..."):
        state = {"input": user_input, "model": model}
        result = chat_graph.invoke(state)
        response = result["response"]

    st.session_state["sessions"][current_session]["messages"].append({"role": "assistant", "content": response})

# Display chat messages
for msg in st.session_state["sessions"][st.session_state["current_session"]]["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
