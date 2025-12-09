"""
バックグラウンド更新サービス
===========================
ウォッチリストの株価データを自動更新
"""

import threading
import time
from datetime import datetime, timedelta
from typing import Callable, Optional, List, Dict
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BackgroundUpdater:
    """バックグラウンドで株価データを更新するサービス"""
    
    def __init__(self, db_manager, fetch_func: Callable, interval_minutes: int = 60):
        """
        Args:
            db_manager: DatabaseManagerインスタンス
            fetch_func: 株価データを取得する関数 (ticker, period) -> DataFrame
            interval_minutes: 更新間隔（分）
        """
        self.db = db_manager
        self.fetch_func = fetch_func
        self.interval = interval_minutes * 60  # 秒に変換
        
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._is_running = False
        self._last_update: Optional[datetime] = None
        self._update_results: List[Dict] = []
    
    def start(self):
        """バックグラウンド更新を開始"""
        if self._is_running:
            logger.warning("Updater is already running")
            return
        
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()
        self._is_running = True
        logger.info(f"Background updater started (interval: {self.interval // 60} minutes)")
    
    def stop(self):
        """バックグラウンド更新を停止"""
        if not self._is_running:
            return
        
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._is_running = False
        logger.info("Background updater stopped")
    
    def _update_loop(self):
        """更新ループ（バックグラウンドスレッド）"""
        while not self._stop_event.is_set():
            try:
                self._update_all_tickers()
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
            
            # 次の更新まで待機（stop_eventで中断可能）
            self._stop_event.wait(timeout=self.interval)
    
    def _update_all_tickers(self):
        """全ウォッチリスト銘柄を更新"""
        tickers = self.db.get_all_watchlist_tickers()
        
        if not tickers:
            logger.info("No tickers in watchlist")
            return
        
        logger.info(f"Updating {len(tickers)} tickers...")
        self._update_results = []
        
        for ticker in tickers:
            result = self._update_single_ticker(ticker)
            self._update_results.append(result)
            
            # API制限を避けるため少し待機
            time.sleep(0.5)
        
        self._last_update = datetime.now()
        
        success_count = sum(1 for r in self._update_results if r['success'])
        logger.info(f"Update completed: {success_count}/{len(tickers)} successful")
    
    def _update_single_ticker(self, ticker: str) -> Dict:
        """単一銘柄を更新"""
        result = {
            'ticker': ticker,
            'success': False,
            'records': 0,
            'error': None,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # 株価データを取得
            df = self.fetch_func(ticker, "2y")
            
            if df is not None and len(df) > 0:
                # DBにキャッシュ
                count = self.db.cache_prices(ticker, df)
                result['success'] = True
                result['records'] = count
                logger.debug(f"Updated {ticker}: {count} records")
            else:
                result['error'] = "No data returned"
                logger.warning(f"No data for {ticker}")
                
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Error updating {ticker}: {e}")
        
        return result
    
    def update_single(self, ticker: str) -> Dict:
        """単一銘柄を即座に更新（同期）"""
        return self._update_single_ticker(ticker)
    
    def update_stale(self, max_age_hours: int = 24) -> List[Dict]:
        """古いキャッシュの銘柄のみ更新"""
        stale_tickers = self.db.get_stale_tickers(max_age_hours)
        
        if not stale_tickers:
            logger.info("All caches are fresh")
            return []
        
        logger.info(f"Updating {len(stale_tickers)} stale tickers...")
        results = []
        
        for ticker in stale_tickers:
            result = self._update_single_ticker(ticker)
            results.append(result)
            time.sleep(0.5)
        
        return results
    
    @property
    def is_running(self) -> bool:
        """更新サービスが実行中かどうか"""
        return self._is_running
    
    @property
    def last_update(self) -> Optional[datetime]:
        """最後の更新日時"""
        return self._last_update
    
    @property
    def update_results(self) -> List[Dict]:
        """最後の更新結果"""
        return self._update_results
    
    def get_status(self) -> Dict:
        """サービスの状態を取得"""
        return {
            'is_running': self._is_running,
            'interval_minutes': self.interval // 60,
            'last_update': self._last_update.isoformat() if self._last_update else None,
            'total_tickers': len(self.db.get_all_watchlist_tickers()),
            'stale_tickers': len(self.db.get_stale_tickers()),
            'last_results': self._update_results[-10:] if self._update_results else []
        }


# グローバルインスタンス（Streamlitのセッション間で共有）
_updater_instance: Optional[BackgroundUpdater] = None


def get_updater(db_manager=None, fetch_func=None) -> Optional[BackgroundUpdater]:
    """バックグラウンドアップデーターのシングルトンインスタンスを取得"""
    global _updater_instance
    
    if _updater_instance is None and db_manager is not None and fetch_func is not None:
        _updater_instance = BackgroundUpdater(db_manager, fetch_func)
    
    return _updater_instance


def smart_fetch_stock_data(ticker: str, period: str, db_manager, 
                           api_fetch_func: Callable, 
                           cache_max_age_hours: int = 6):
    """
    スマートキャッシュを使った株価データ取得
    
    1. キャッシュが新鮮（< max_age_hours）→ DBから取得
    2. キャッシュが古い or 無い → APIから取得してDBに保存
    
    Args:
        ticker: 銘柄コード
        period: データ期間 ("1y", "2y", "5y")
        db_manager: DatabaseManagerインスタンス
        api_fetch_func: API取得関数 (ticker, period) -> DataFrame
        cache_max_age_hours: キャッシュ有効時間（時間）
    
    Returns:
        tuple: (DataFrame, source: "cache" | "api")
    """
    import pandas as pd
    
    # 期間を日数に変換
    period_days = {"1y": 365, "2y": 730, "5y": 1825}.get(period, 365)
    
    # キャッシュ状態をチェック
    cache_status = db_manager.get_cache_status(ticker)
    
    use_cache = False
    if cache_status['has_cache']:
        # キャッシュが新鮮で、十分なデータがある場合
        if cache_status['is_fresh'] or (cache_status['days_since_update'] and cache_status['days_since_update'] < cache_max_age_hours / 24):
            if cache_status['record_count'] >= period_days * 0.7:  # 70%以上のデータがあれば使用
                use_cache = True
    
    if use_cache:
        # キャッシュから取得
        df = db_manager.get_cached_prices(ticker, days=period_days)
        if df is not None and len(df) > 0:
            return df, "cache"
    
    # APIから取得
    try:
        df = api_fetch_func(ticker, period)
        if df is not None and len(df) > 0:
            # キャッシュに保存
            db_manager.cache_prices(ticker, df)
            return df, "api"
    except Exception as e:
        logger.error(f"API fetch failed for {ticker}: {e}")
    
    # APIが失敗した場合、古いキャッシュでも返す
    if cache_status['has_cache']:
        df = db_manager.get_cached_prices(ticker, days=period_days)
        if df is not None and len(df) > 0:
            return df, "stale_cache"
    
    return None, "none"
