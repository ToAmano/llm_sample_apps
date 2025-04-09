"""
もっともシンプルなGemini 2.0 Pro チャットアプリ
"""

import streamlit as st
import os
from dotenv import load_dotenv
from datetime import datetime

# 環境変数の読み込み
load_dotenv()

# バックエンドのインポート
from backend import GeminiChatApp

# ページ設定
st.set_page_config(
    page_title="Gemini 2.0 チャット",
    page_icon="🤖",
    layout="wide"
)

# セッション状態の初期化
if "chat_app" not in st.session_state:
    st.session_state.chat_app = GeminiChatApp()

# タイトル
st.title("Gemini 2.0 Pro チャットボット")

# サイドバー設定
with st.sidebar:
    st.header("チャットセッション")
    
    # セッション一覧
    all_sessions = st.session_state.chat_app.get_all_sessions()
    current_session = st.session_state.chat_app.get_current_session()
    current_session_id = current_session.session_id if current_session else None
    
    # セッション選択
    session_names = {session_id: session["name"] for session_id, session in all_sessions.items()}
    
    # 現在のセッションのインデックスを特定
    session_ids = list(session_names.keys())
    current_index = session_ids.index(current_session_id) if current_session_id in session_ids else 0
    
    selected_session_name = st.selectbox(
        "セッションを選択",
        options=list(session_names.values()),
        index=current_index
    )
    
    # 選択されたセッション名からIDを取得
    selected_session_id = [sid for sid, name in session_names.items() if name == selected_session_name][0]
    
    # セッションが変更された場合に切り替え
    if selected_session_id != current_session_id:
        st.session_state.chat_app.switch_session(selected_session_id)
        st.rerun()
    
    # セッション管理ボタン
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("新規セッション", key="new_session"):
            st.session_state.chat_app.create_session()
            st.rerun()
    
    with col2:
        if st.button("セッション削除", key="delete_session"):
            if len(all_sessions) > 1:  # 少なくとも1つのセッションは残す
                if st.session_state.chat_app.delete_session(selected_session_id):
                    st.success("セッションを削除しました")
                    st.rerun()
            else:
                st.error("削除できません。少なくとも1つのセッションが必要です。")
    
    # セッション名変更
    new_session_name = st.text_input("セッション名", value=selected_session_name)
    if new_session_name != selected_session_name:
        if st.session_state.chat_app.rename_session(selected_session_id, new_session_name):
            st.success("セッション名を変更しました")
            st.rerun()
    
    st.divider()
    
    # システムプロンプト設定
    st.header("設定")
    st.subheader("システムプロンプト")
    
    # 現在のシステムプロンプトを取得
    current_system_prompt = ""
    if current_session and current_session.system_message:
        current_system_prompt = current_session.system_message
    
    system_prompt = st.text_area(
        "AIへの指示を入力",
        value=current_system_prompt,
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
    
    google_key = os.getenv("GOOGLE_API_KEY")
    
    if google_key:
        st.success("Google API: 接続可能")
    else:
        st.error("Google API: APIキーが設定されていません")
    
    # 使用方法説明
    st.markdown("---")
    st.subheader("使用方法")
    st.markdown("""
    1. サイドバーからセッションを選択または新規作成
    2. 必要に応じてシステムプロンプトを設定
    3. メッセージ入力欄にテキストを入力してチャット開始
    """)

# チャット履歴の表示
chat_container = st.container()
with chat_container:
    # 現在のセッション情報を表示
    if current_session:
        st.caption(f"現在のセッション: {current_session.name}")
        
        # チャット履歴を表示
        for message in st.session_state.chat_app.get_conversation_history():
            if message["role"] == "human":
                st.chat_message("user").write(message["content"])
            elif message["role"] == "ai":
                st.chat_message("assistant").write(message["content"])

# メッセージ入力
with st.container():
    user_input = st.chat_input("メッセージを入力してください...")
    
    if user_input:
        # ユーザーメッセージの表示
        st.chat_message("user").write(user_input)
        
        # 処理中の表示
        with st.spinner("Gemini AIが考え中..."):
            # LLMからの回答を取得
            try:
                response = st.session_state.chat_app.chat(user_input)
                # 回答の表示
                st.chat_message("assistant").write(response)
            except Exception as e:
                st.error(f"エラーが発生しました: {str(e)}")
        
        # 画面のリロード
        st.rerun()