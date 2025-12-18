"""
LINE通知サービス
================
売買シグナルをLINE Notifyで通知する

セットアップ:
1. LINE Notify (https://notify-bot.line.me/) にアクセス
2. 「トークンを発行する」でアクセストークンを取得
3. .env ファイルに LINE_NOTIFY_TOKEN=your_token を設定
"""

import requests
import os
from datetime import datetime
from typing import Optional, List, Dict
from pathlib import Path

# .envファイルから環境変数を読み込む
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class LineNotifyService:
    """LINE Notify API を使った通知サービス"""
    
    API_URL = "https://notify-api.line.me/api/notify"
    
    def __init__(self, token: Optional[str] = None):
        """
        Args:
            token: LINE Notify アクセストークン。
                   None の場合は環境変数 LINE_NOTIFY_TOKEN から取得
        """
        self.token = token or os.getenv("LINE_NOTIFY_TOKEN")
        if not self.token:
            print("⚠️ LINE_NOTIFY_TOKEN が設定されていません")
            print("   .env ファイルに LINE_NOTIFY_TOKEN=your_token を追加してください")
    
    def send(self, message: str, disable_notification: bool = False) -> bool:
        """
        LINE通知を送信
        
        Args:
            message: 送信するメッセージ (最大1000文字)
            disable_notification: True の場合、通知音なしで送信
            
        Returns:
            成功した場合 True
        """
        if not self.token:
            print(f"[LINE] トークン未設定: {message[:50]}...")
            return False
        
        headers = {
            "Authorization": f"Bearer {self.token}"
        }
        
        data = {
            "message": message[:1000],  # 最大1000文字
            "notificationDisabled": disable_notification
        }
        
        try:
            response = requests.post(self.API_URL, headers=headers, data=data, timeout=10)
            if response.status_code == 200:
                return True
            else:
                print(f"[LINE] エラー: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"[LINE] 送信エラー: {e}")
            return False
    
    def send_buy_signal(self, ticker: str, price: float, ai_score: float, 
                        reason: str = "", additional_info: Dict = None) -> bool:
        """
        買いシグナル通知
        """
        emoji = "🟢"
        now = datetime.now().strftime("%m/%d %H:%M")
        
        message = f"""
{emoji} 買いシグナル {emoji}
━━━━━━━━━━━━━━
📈 銘柄: {ticker}
💰 現在値: ¥{price:,.0f}
🤖 AIスコア: {ai_score:.1f}
📝 理由: {reason or 'シグナル条件達成'}
⏰ {now}
"""
        if additional_info:
            if 'rsi' in additional_info:
                message += f"📊 RSI: {additional_info['rsi']:.1f}\n"
            if 'macd' in additional_info:
                message += f"📉 MACD: {additional_info['macd']:.2f}\n"
        
        return self.send(message)
    
    def send_sell_signal(self, ticker: str, price: float, profit_rate: float,
                         reason: str = "", hold_days: int = 0) -> bool:
        """
        売りシグナル通知
        """
        emoji = "🔴" if profit_rate < 0 else "🟡"
        profit_emoji = "📈" if profit_rate > 0 else "📉"
        now = datetime.now().strftime("%m/%d %H:%M")
        
        message = f"""
{emoji} 売りシグナル {emoji}
━━━━━━━━━━━━━━
📊 銘柄: {ticker}
💰 現在値: ¥{price:,.0f}
{profit_emoji} 損益: {profit_rate:+.2f}%
📝 理由: {reason or '売りシグナル'}
📅 保有日数: {hold_days}日
⏰ {now}
"""
        return self.send(message)
    
    def send_daily_summary(self, summary: Dict) -> bool:
        """
        日次サマリー通知
        """
        now = datetime.now().strftime("%Y/%m/%d %H:%M")
        
        message = f"""
📊 日次レポート 📊
━━━━━━━━━━━━━━
💼 保有銘柄数: {summary.get('holdings', 0)}
💰 評価額: ¥{summary.get('total_value', 0):,.0f}
📈 本日損益: {summary.get('daily_return', 0):+.2f}%
📊 累計損益: {summary.get('total_return', 0):+.2f}%

🔔 本日のシグナル:
  買い: {summary.get('buy_signals', 0)}件
  売り: {summary.get('sell_signals', 0)}件
  
⏰ {now}
"""
        return self.send(message)
    
    def send_alert(self, title: str, message: str, level: str = "info") -> bool:
        """
        アラート通知
        
        Args:
            level: "info", "warning", "error"
        """
        emoji_map = {
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "🚨"
        }
        emoji = emoji_map.get(level, "ℹ️")
        now = datetime.now().strftime("%m/%d %H:%M")
        
        full_message = f"""
{emoji} {title}
━━━━━━━━━━━━━━
{message}
⏰ {now}
"""
        return self.send(full_message)


def test_notification():
    """通知テスト"""
    service = LineNotifyService()
    
    print("LINE通知テストを送信中...")
    
    # テスト通知
    success = service.send_alert(
        "テスト通知",
        "stock-predictor からの通知テストです。\nこのメッセージが届けば設定成功です！",
        "info"
    )
    
    if success:
        print("✅ 通知送信成功！")
    else:
        print("❌ 通知送信失敗。トークンを確認してください。")
    
    return success


if __name__ == "__main__":
    test_notification()
