from typing import List, Dict, Any, TypedDict, Literal, Optional
import os
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END, START

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

# ユーザー入力を処理する関数
def process_user_input(state: ChatState, user_input: str):
    """ユーザー入力をメッセージリストに追加"""
    new_messages = state["messages"].copy()
    new_messages.append({"role": "human", "content": user_input})
    
    return {"messages": new_messages, "current_model": state["current_model"], "system_message": state["system_message"]}

# モデルを切り替える関数
def switch_model(state: ChatState, model: ModelType):
    """使用するLLMモデルを切り替える"""
    return {"messages": state["messages"], "current_model": model, "system_message": state["system_message"]}

# システムメッセージを設定する関数
def set_system_message(state: ChatState, system_message: str):
    """システムメッセージを設定"""
    return {"messages": state["messages"], "current_model": state["current_model"], "system_message": system_message}

# langgraphのワークフローを定義
def build_chat_graph():
    # 状態グラフの作成
    builder = StateGraph(ChatState)
    
    # ノードの定義
    builder.add_node("process_input", process_user_input)
    builder.add_node("query_llm", query_llm)
    builder.add_node("switch_model", switch_model)
    builder.add_node("set_system", set_system_message)
    
    # エッジの定義
    builder.add_edge(START, "process_input")
    builder.add_edge("process_input", "query_llm")
    builder.add_edge("query_llm", END)
    builder.add_edge("switch_model", END)
    builder.add_edge("set_system", END)
    
    # グラフの構築
    return builder.compile()

# チャットアプリケーションクラス
class MultiModelChatApp:
    def __init__(self):
        self.graph = build_chat_graph()
        self.state = {
            "messages": [],
            "current_model": "gpt-4o",  # デフォルトモデル
            "system_message": None
        }
    
    def chat(self, user_input: str):
        """ユーザー入力に対する応答を生成"""
        self.state = self.graph.invoke("process_input", {"user_input": user_input}, self.state)
        return self.state["messages"][-1]["content"]
    
    def change_model(self, model: ModelType):
        """使用するモデルを変更"""
        self.state = self.graph.invoke("switch_model", {"model": model}, self.state)
        return f"モデルを {model} に切り替えました"
    
    def set_system_prompt(self, system_prompt: str):
        """システムプロンプトを設定"""
        self.state = self.graph.invoke("set_system", {"system_message": system_prompt}, self.state)
        return "システムプロンプトを設定しました"
    
    def get_current_model(self):
        """現在のモデルを取得"""
        return self.state["current_model"]
    
    def get_conversation_history(self):
        """会話履歴を取得"""
        return self.state["messages"]

# 使用例
if __name__ == "__main__":
    chat_app = MultiModelChatApp()
    
    # モデルを切り替える
    print(chat_app.change_model("claude-3-7-sonnet"))
    
    # システムプロンプトを設定
    print(chat_app.set_system_prompt("あなたは親切なアシスタントです。"))
    
    # チャット
    response = chat_app.chat("こんにちは、自己紹介してください")
    print(f"AI: {response}")
    
    # モデルを切り替える
    print(chat_app.change_model("gemini-2.0-pro"))
    
    # 再度チャット
    response = chat_app.chat("前と違うモデルに切り替えました。あなたは何ができますか？")
    print(f"AI: {response}")