from typing import List, Dict, Any, Optional
import os
import uuid
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from prompt import SYSTEM_PROMPT_CLAUDE

class ChatSession:
    """チャットセッションを管理するクラス"""
    
    def __init__(self, session_id=None, name=None):
        """セッションの初期化"""
        self.session_id = session_id or str(uuid.uuid4())
        self.name = name or f"セッション {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        self.messages = []
        self.system_message = None
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    def add_message(self, role: str, content: str):
        """メッセージを追加する"""
        self.messages.append({"role": role, "content": content})
        self.updated_at = datetime.now()
    
    def get_messages(self):
        """すべてのメッセージを取得する"""
        return self.messages
    
    def set_system_message(self, content: str):
        """システムメッセージを設定する"""
        self.system_message = content
        self.updated_at = datetime.now()
    
    def to_dict(self):
        """セッション情報を辞書形式で返す"""
        return {
            "session_id": self.session_id,
            "name": self.name,
            "messages": self.messages,
            "system_message": self.system_message,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


class GeminiChatApp:
    """Gemini 2.0 Proを使用したチャットアプリケーション（セッション対応）"""
    
    def __init__(self):
        """チャットアプリケーションの初期化"""
        self.sessions = {}
        self.current_session_id = None
        self.model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
        
        # デフォルトセッションを作成
        self.create_session()
    
    def create_session(self, name=None):
        """新しいセッションを作成し、そのセッションに切り替える"""
        session = ChatSession(name=name)
        self.sessions[session.session_id] = session
        self.current_session_id = session.session_id
        return session.session_id
    
    def switch_session(self, session_id):
        """指定されたIDのセッションに切り替える"""
        if session_id in self.sessions:
            self.current_session_id = session_id
            return True
        return False
    
    def delete_session(self, session_id):
        """指定されたIDのセッションを削除する"""
        if session_id in self.sessions:
            # 現在のセッションを削除する場合は別のセッションに切り替える
            if session_id == self.current_session_id:
                # 他のセッションがあれば最初のセッションに切り替え
                other_sessions = [s for s in self.sessions if s != session_id]
                if other_sessions:
                    self.current_session_id = other_sessions[0]
                else:
                    # 他のセッションがなければ新しいセッションを作成
                    self.create_session()
            
            # セッションを削除
            del self.sessions[session_id]
            return True
        return False
    
    def rename_session(self, session_id, new_name):
        """セッション名を変更する"""
        if session_id in self.sessions:
            self.sessions[session_id].name = new_name
            return True
        return False
    
    def get_llm(model_type: str):
        if model_type == "gpt-4o":
            return ChatOpenAI(model="gpt-4o", temperature=0.7)
        elif model_type == "gemini-2.0-pro":
            return ChatGoogleGenerativeAI(model="gemini-2.0-pro", temperature=0.7)
        elif model_type == "claude-3-7-sonnet":
            return ChatAnthropic(model="claude-3-7-sonnet-20250211", temperature=0.7)
        elif model_type == "gemini-2.0-flash":
            return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
        else:
            raise ValueError(f"不明なモデルタイプ: {model_type}")
    
    def get_current_session(self):
        """現在のセッションを取得する"""
        return self.sessions.get(self.current_session_id)
    
    def get_all_sessions(self):
        """すべてのセッション情報を取得する"""
        return {session_id: session.to_dict() for session_id, session in self.sessions.items()}
    
    def chat(self, user_input: str) -> str:
        """ユーザー入力に対する応答を生成する"""
        current_session = self.get_current_session()
        if not current_session:
            return "エラー: アクティブなセッションがありません。"
        
        # ユーザーメッセージを追加
        current_session.add_message("human", user_input)
        
        # LLMに送信するメッセージを作成
        llm_messages = []
        
        # システムメッセージがあれば追加
        if current_session.system_message:
            llm_messages.append(SystemMessage(content=current_session.system_message))
        else:
            llm_messages.append(SystemMessage(content=SYSTEM_PROMPT_CLAUDE))
        
        # 会話履歴を追加
        for msg in current_session.get_messages():
            if msg["role"] == "human":
                llm_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "ai":
                llm_messages.append(AIMessage(content=msg["content"]))
        # LLMに問い合わせ
        try:
            response = self.model.invoke(llm_messages)
            ai_response = response.content
            
            # AIの応答を履歴に追加
            current_session.add_message("ai", ai_response)
            
            return ai_response
        except Exception as e:
            error_message = f"エラーが発生しました: {str(e)}"
            current_session.add_message("ai", error_message)
            return error_message
    
    def set_system_prompt(self, system_prompt: str) -> str:
        """現在のセッションのシステムプロンプトを設定"""
        current_session = self.get_current_session()
        if not current_session:
            return "エラー: アクティブなセッションがありません。"
        
        current_session.set_system_message(system_prompt)
        return "システムプロンプトを設定しました"
    
    def get_conversation_history(self):
        """現在のセッションの会話履歴を取得"""
        current_session = self.get_current_session()
        if not current_session:
            return []
        
        return current_session.get_messages()
    
    
    def _update_session_summary(self, session):
        """会話内容に基づいてセッション名を要約・更新する"""
        # メッセージが少なすぎる場合は要約しない
        if len(session.get_messages()) < 2:
            return
        
        try:
            # 最初の数メッセージを取得して要約用に使用
            initial_messages = session.get_messages()[:6]  # 最初の6メッセージまでを使用
            
            # 要約用のプロンプト
            summary_prompt = f"""
            以下はチャット会話の抜粋です。この会話を最も適切に表現する短いタイトル（20文字以内）を作成してください。
            タイトルは日本語で、会話の主なトピックや目的を表現するものにしてください。

            会話:
            """
            
            # 会話内容を追加
            for msg in initial_messages:
                role = "ユーザー" if msg["role"] == "human" else "AI"
                summary_prompt += f"\n{role}: {msg['content'][:100]}..." if len(msg["content"]) > 100 else f"\n{role}: {msg['content']}"
            
            summary_prompt += "\n\nタイトル: "
            
            # 要約モデルに問い合わせ
            summary_response = self.summary_model.invoke([HumanMessage(content=summary_prompt)])
            summary = summary_response.content.strip()
            
            # 長すぎる場合はカット
            if len(summary) > 30:
                summary = summary[:27] + "..."
            
            # タイトルを更新
            session.name = summary
            session.summary_updated = True
            
        except Exception as e:
            print(f"セッション名の要約中にエラーが発生しました: {str(e)}")
            # エラーが発生しても処理は続行