import streamlit as st
from typing import List, Dict, Any, Literal
import os
from dotenv import load_dotenv

# バックエンドのコードをインポート
from backend import MultiModelChatApp, ModelType

# 環境変数の読み込み
load_dotenv()

# ページ設定
st.set_page_config(
    page_title="マルチLLMチャットボット",
    page_icon="🤖",
    layout="wide",
)

# セッション状態の初期化
if "chat_app" not in st.session_state:
    st.session_state.chat_app = MultiModelChatApp()

if "messages" not in st.session_state:
    st.session_state.messages = []

# タイトル
st.title("マルチLLMチャットボット")

# サイドバー設定
with st.sidebar:
    st.header("設定")
    
    # モデル選択
    model_options = {
        "OpenAI GPT-4o": "gpt-4o",
        "Google Gemini 2.0 Pro": "gemini-2.0-pro",
        "Anthropic Claude 3.7 Sonnet": "claude-3-7-sonnet"
    }
    
    selected_model = st.selectbox(
        "使用するLLMモデルを選択",
        options=list(model_options.keys()),
        index=list(model_options.values()).index(st.session_state.chat_app.get_current_model())
    )
    
    # モデル切り替えボタン
    if st.button("モデルを切り替え"):
        model_key = model_options[selected_model]
        response = st.session_state.chat_app.change_model(model_key)
        st.success(response)
    
    # システムプロンプト設定
    st.subheader("システムプロンプト")
    system_prompt = st.text_area(
        "AIへの指示を入力",
        placeholder="例: あなたは親切なアシスタントです。必ず日本語で回答してください。",
        height=150
    )
    
    if st.button("システムプロンプトを設定"):
        if system_prompt:
            response = st.session_state.chat_app.set_system_prompt(system_prompt)
            st.success(response)
        else:
            st.warning("プロンプトを入力してください")
    
    # API状態の表示
    st.subheader("API接続状態")
    
    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    if openai_key:
        st.success("OpenAI API: 接続可能")
    else:
        st.error("OpenAI API: APIキーが設定されていません")
    
    if google_key:
        st.success("Google API: 接続可能")
    else:
        st.error("Google API: APIキーが設定されていません")
    
    if anthropic_key:
        st.success("Anthropic API: 接続可能")
    else:
        st.error("Anthropic API: APIキーが設定されていません")
    
    # 使用方法説明
    st.markdown("---")
    st.subheader("使用方法")
    st.markdown("""
    1. サイドバーからLLMモデルを選択
    2. 必要に応じてシステムプロンプトを設定
    3. メッセージ入力欄にテキストを入力してチャット開始
    4. モデルを切り替えて同じ会話を続けることができます
    """)

# チャット履歴の表示
chat_container = st.container()
with chat_container:
    # チャット履歴を表示
    for message in st.session_state.chat_app.get_conversation_history():
        if message["role"] == "human":
            st.chat_message("user").write(message["content"])
        elif message["role"] == "ai":
            model_name = message.get("model", "unknown")
            # モデル名を人間が読みやすい形式に変換
            if model_name == "gpt-4o":
                display_model = "OpenAI GPT-4o"
            elif model_name == "gemini-2.0-pro":
                display_model = "Google Gemini 2.0 Pro"
            elif model_name == "claude-3-7-sonnet":
                display_model = "Anthropic Claude 3.7 Sonnet"
            else:
                display_model = model_name
                
            with st.chat_message("assistant"):
                st.write(message["content"])
                st.caption(f"回答モデル: {display_model}")

# メッセージ入力
with st.container():
    user_input = st.chat_input("メッセージを入力してください...")
    
    if user_input:
        # ユーザーメッセージの表示
        st.chat_message("user").write(user_input)
        
        # 処理中の表示
        with st.spinner("AIが考え中..."):
            # LLMからの回答を取得
            response = st.session_state.chat_app.chat(user_input)
        
        # 回答の表示とセッション状態の更新
        current_model = st.session_state.chat_app.get_current_model()
        model_display_name = {
            "gpt-4o": "OpenAI GPT-4o",
            "gemini-2.0-pro": "Google Gemini 2.0 Pro",
            "claude-3-7-sonnet": "Anthropic Claude 3.7 Sonnet"
        }.get(current_model, current_model)
        
        with st.chat_message("assistant"):
            st.write(response)
            st.caption(f"回答モデル: {model_display_name}")
        
        # 画面のリロード
        st.rerun()