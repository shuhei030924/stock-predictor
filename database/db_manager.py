"""
データベース管理モジュール
========================
SQLiteを使用した銘柄・予測データの永続化
"""

import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import json
import pandas as pd


class DatabaseManager:
    """SQLiteデータベース管理クラス"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            # デフォルトはプロジェクトルートのdata/stock_predictor.db
            db_dir = Path(__file__).parent.parent / "data"
            db_dir.mkdir(exist_ok=True)
            db_path = db_dir / "stock_predictor.db"
        
        self.db_path = str(db_path)
        self._init_database()
    
    def _get_connection(self) -> sqlite3.Connection:
        """データベース接続を取得"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_database(self):
        """データベースとテーブルを初期化"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # ウォッチリストテーブル
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS watchlist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT UNIQUE NOT NULL,
                name TEXT,
                sector TEXT,
                market TEXT,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                notes TEXT,
                is_favorite INTEGER DEFAULT 0
            )
        """)
        
        # 株価キャッシュテーブル
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS price_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                date DATE NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, date)
            )
        """)
        
        # 予測履歴テーブル
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                prediction_date DATE NOT NULL,
                target_date DATE NOT NULL,
                model_type TEXT NOT NULL,
                predicted_price REAL,
                actual_price REAL,
                prediction_error REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # アラート設定テーブル
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                condition TEXT NOT NULL,
                target_price REAL NOT NULL,
                note TEXT,
                is_active INTEGER DEFAULT 1,
                triggered INTEGER DEFAULT 0,
                triggered_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # ユーザー設定テーブル
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_settings (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # ポートフォリオテーブル
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS portfolio (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                shares REAL NOT NULL,
                avg_cost REAL NOT NULL,
                purchase_date DATE,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # シグナルキャッシュテーブル
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signal_cache (
                ticker TEXT PRIMARY KEY,
                price REAL,
                change REAL,
                rsi REAL,
                rsi_signal REAL,
                ma_signal REAL,
                macd_signal REAL,
                bb_signal REAL,
                vol_signal REAL,
                total_score REAL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # バックテスト用テーブル
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS backtest_portfolio (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                shares REAL NOT NULL,
                avg_cost REAL NOT NULL,
                current_price REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # バックテスト取引履歴
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS backtest_transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                action TEXT NOT NULL,
                shares REAL NOT NULL,
                price REAL NOT NULL,
                amount REAL NOT NULL,
                signal_score REAL,
                reason TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # バックテスト資産推移
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS backtest_balance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cash REAL NOT NULL,
                stock_value REAL NOT NULL,
                total_value REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # インデックス作成
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_price_cache_ticker ON price_cache(ticker)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_price_cache_date ON price_cache(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_prediction_ticker ON prediction_history(ticker)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_portfolio_ticker ON portfolio(ticker)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_ticker ON alerts(ticker)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_signal_cache_score ON signal_cache(total_score)")
        
        conn.commit()
        conn.close()
        
        # マイグレーション実行
        self._run_migrations()
    
    def _run_migrations(self):
        """既存テーブルのマイグレーション"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # alertsテーブルのマイグレーション（古いスキーマから新しいスキーマへ）
        try:
            cursor.execute("SELECT condition FROM alerts LIMIT 1")
        except sqlite3.OperationalError:
            # 古いスキーマの場合、テーブルを再作成
            try:
                cursor.execute("DROP TABLE IF EXISTS alerts")
                cursor.execute("""
                    CREATE TABLE alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ticker TEXT NOT NULL,
                        condition TEXT NOT NULL,
                        target_price REAL NOT NULL,
                        note TEXT,
                        is_active INTEGER DEFAULT 1,
                        triggered INTEGER DEFAULT 0,
                        triggered_at TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
            except Exception as e:
                print(f"Migration error: {e}")
        
        conn.close()
    
    # ==================== ウォッチリスト操作 ====================
    
    def add_to_watchlist(self, ticker: str, name: str = None, 
                         sector: str = None, market: str = None,
                         notes: str = None) -> bool:
        """ウォッチリストに銘柄を追加"""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO watchlist (ticker, name, sector, market, notes, added_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (ticker.upper(), name, sector, market, notes))
            conn.commit()
            return True
        except Exception as e:
            print(f"Error adding to watchlist: {e}")
            return False
        finally:
            conn.close()
    
    def remove_from_watchlist(self, ticker: str) -> bool:
        """ウォッチリストから銘柄を削除"""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM watchlist WHERE ticker = ?", (ticker.upper(),))
            conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            print(f"Error removing from watchlist: {e}")
            return False
        finally:
            conn.close()
    
    def get_watchlist(self, favorites_only: bool = False) -> List[Dict]:
        """ウォッチリストを取得"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if favorites_only:
            cursor.execute("SELECT * FROM watchlist WHERE is_favorite = 1 ORDER BY added_at DESC")
        else:
            cursor.execute("SELECT * FROM watchlist ORDER BY is_favorite DESC, added_at DESC")
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def toggle_favorite(self, ticker: str) -> bool:
        """お気に入りを切り替え"""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                UPDATE watchlist 
                SET is_favorite = CASE WHEN is_favorite = 1 THEN 0 ELSE 1 END
                WHERE ticker = ?
            """, (ticker.upper(),))
            conn.commit()
            return True
        except Exception as e:
            print(f"Error toggling favorite: {e}")
            return False
        finally:
            conn.close()
    
    def update_notes(self, ticker: str, notes: str) -> bool:
        """メモを更新"""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                UPDATE watchlist SET notes = ? WHERE ticker = ?
            """, (notes, ticker.upper()))
            conn.commit()
            return True
        except Exception as e:
            print(f"Error updating notes: {e}")
            return False
        finally:
            conn.close()
    
    # ==================== 株価キャッシュ操作 ====================
    
    def cache_prices(self, ticker: str, df: pd.DataFrame) -> int:
        """株価データをキャッシュ"""
        conn = self._get_connection()
        cursor = conn.cursor()
        count = 0
        
        try:
            for date, row in df.iterrows():
                cursor.execute("""
                    INSERT OR REPLACE INTO price_cache 
                    (ticker, date, open, high, low, close, volume, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    ticker.upper(),
                    date.strftime('%Y-%m-%d'),
                    float(row['Open']),
                    float(row['High']),
                    float(row['Low']),
                    float(row['Close']),
                    int(row['Volume'])
                ))
                count += 1
            conn.commit()
        except Exception as e:
            print(f"Error caching prices: {e}")
        finally:
            conn.close()
        
        return count
    
    def get_cached_prices(self, ticker: str, days: int = 365) -> Optional[pd.DataFrame]:
        """キャッシュから株価データを取得"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        since_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        cursor.execute("""
            SELECT date, open, high, low, close, volume
            FROM price_cache
            WHERE ticker = ? AND date >= ?
            ORDER BY date
        """, (ticker.upper(), since_date))
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return None
        
        df = pd.DataFrame([dict(row) for row in rows])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        return df
    
    def is_cache_fresh(self, ticker: str, max_age_hours: int = 24) -> bool:
        """キャッシュが新鮮かどうかチェック"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT MAX(updated_at) as last_update
            FROM price_cache
            WHERE ticker = ?
        """, (ticker.upper(),))
        
        row = cursor.fetchone()
        conn.close()
        
        if row is None or row['last_update'] is None:
            return False
        
        last_update = datetime.fromisoformat(row['last_update'])
        return (datetime.now() - last_update).total_seconds() < max_age_hours * 3600
    
    # ==================== 予測履歴操作 ====================
    
    def save_prediction(self, ticker: str, prediction_date: str, 
                       target_date: str, model_type: str, 
                       predicted_price: float) -> bool:
        """予測結果を保存"""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO prediction_history 
                (ticker, prediction_date, target_date, model_type, predicted_price)
                VALUES (?, ?, ?, ?, ?)
            """, (ticker.upper(), prediction_date, target_date, model_type, predicted_price))
            conn.commit()
            return True
        except Exception as e:
            print(f"Error saving prediction: {e}")
            return False
        finally:
            conn.close()
    
    def update_actual_price(self, ticker: str, target_date: str, actual_price: float) -> int:
        """実際の価格で予測履歴を更新"""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                UPDATE prediction_history 
                SET actual_price = ?,
                    prediction_error = ABS(predicted_price - ?) / ? * 100
                WHERE ticker = ? AND target_date = ? AND actual_price IS NULL
            """, (actual_price, actual_price, actual_price, ticker.upper(), target_date))
            conn.commit()
            return cursor.rowcount
        except Exception as e:
            print(f"Error updating actual price: {e}")
            return 0
        finally:
            conn.close()
    
    def get_prediction_accuracy(self, ticker: str = None, model_type: str = None) -> Dict:
        """予測精度を取得"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        query = """
            SELECT 
                model_type,
                COUNT(*) as total_predictions,
                AVG(prediction_error) as avg_error,
                MIN(prediction_error) as min_error,
                MAX(prediction_error) as max_error
            FROM prediction_history
            WHERE actual_price IS NOT NULL
        """
        params = []
        
        if ticker:
            query += " AND ticker = ?"
            params.append(ticker.upper())
        
        if model_type:
            query += " AND model_type = ?"
            params.append(model_type)
        
        query += " GROUP BY model_type"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        return {row['model_type']: dict(row) for row in rows}
    
    # ==================== アラート操作 ====================
    
    def add_alert(self, ticker: str, target_price: float, condition: str = 'above',
                  note: str = None) -> bool:
        """アラートを追加"""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO alerts (ticker, target_price, condition, note)
                VALUES (?, ?, ?, ?)
            """, (ticker.upper(), target_price, condition, note))
            conn.commit()
            return True
        except Exception as e:
            print(f"Error adding alert: {e}")
            return False
        finally:
            conn.close()
    
    def get_alerts(self, ticker: str = None, active_only: bool = True) -> List[Dict]:
        """アラートを取得"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        query = "SELECT * FROM alerts WHERE 1=1"
        params = []
        
        if ticker:
            query += " AND ticker = ?"
            params.append(ticker.upper())
        
        if active_only:
            query += " AND is_active = 1"
        
        query += " ORDER BY created_at DESC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def delete_alert(self, alert_id: int) -> bool:
        """アラートを削除"""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM alerts WHERE id = ?", (alert_id,))
            conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            print(f"Error deleting alert: {e}")
            return False
        finally:
            conn.close()
    
    def mark_alert_triggered(self, alert_id: int) -> bool:
        """アラートを発動済みにする"""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                UPDATE alerts 
                SET triggered = 1, triggered_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (alert_id,))
            conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            print(f"Error marking alert triggered: {e}")
            return False
        finally:
            conn.close()
    
    # ==================== ユーザー設定操作 ====================
    
    def set_setting(self, key: str, value: Any) -> bool:
        """設定を保存"""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO user_settings (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            """, (key, json.dumps(value)))
            conn.commit()
            return True
        except Exception as e:
            print(f"Error setting value: {e}")
            return False
        finally:
            conn.close()
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """設定を取得"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT value FROM user_settings WHERE key = ?", (key,))
        row = cursor.fetchone()
        conn.close()
        
        if row is None:
            return default
        
        try:
            return json.loads(row['value'])
        except:
            return row['value']
    
    # ==================== 統計情報 ====================
    
    def get_stats(self) -> Dict:
        """データベース統計情報を取得"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        stats = {}
        
        cursor.execute("SELECT COUNT(*) as count FROM watchlist")
        stats['watchlist_count'] = cursor.fetchone()['count']
        
        cursor.execute("SELECT COUNT(DISTINCT ticker) as count FROM price_cache")
        stats['cached_tickers'] = cursor.fetchone()['count']
        
        cursor.execute("SELECT COUNT(*) as count FROM price_cache")
        stats['cached_prices'] = cursor.fetchone()['count']
        
        cursor.execute("SELECT COUNT(*) as count FROM prediction_history")
        stats['total_predictions'] = cursor.fetchone()['count']
        
        cursor.execute("SELECT COUNT(*) as count FROM alerts WHERE is_active = 1")
        stats['active_alerts'] = cursor.fetchone()['count']
        
        conn.close()
        return stats
    
    # ==================== スマートキャッシュ ====================
    
    def get_cache_status(self, ticker: str) -> Dict:
        """キャッシュの状態を取得"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                COUNT(*) as record_count,
                MIN(date) as oldest_date,
                MAX(date) as latest_date,
                MAX(updated_at) as last_update
            FROM price_cache
            WHERE ticker = ?
        """, (ticker.upper(),))
        
        row = cursor.fetchone()
        conn.close()
        
        if row['record_count'] == 0:
            return {
                'has_cache': False,
                'record_count': 0,
                'oldest_date': None,
                'latest_date': None,
                'last_update': None,
                'is_fresh': False,
                'days_since_update': None
            }
        
        last_update = datetime.fromisoformat(row['last_update']) if row['last_update'] else None
        days_since_update = (datetime.now() - last_update).total_seconds() / 86400 if last_update else None
        
        return {
            'has_cache': True,
            'record_count': row['record_count'],
            'oldest_date': row['oldest_date'],
            'latest_date': row['latest_date'],
            'last_update': row['last_update'],
            'is_fresh': days_since_update < 1 if days_since_update else False,
            'days_since_update': days_since_update
        }
    
    def get_stale_tickers(self, max_age_hours: int = 24) -> List[str]:
        """古いキャッシュの銘柄リストを取得"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # ウォッチリストにあってキャッシュが古い or 無い銘柄
        cursor.execute("""
            SELECT w.ticker
            FROM watchlist w
            LEFT JOIN (
                SELECT ticker, MAX(updated_at) as last_update
                FROM price_cache
                GROUP BY ticker
            ) p ON w.ticker = p.ticker
            WHERE p.last_update IS NULL 
               OR datetime(p.last_update) < datetime('now', ? || ' hours')
        """, (f'-{max_age_hours}',))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [row['ticker'] for row in rows]
    
    def get_all_watchlist_tickers(self) -> List[str]:
        """ウォッチリストの全銘柄コードを取得"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT ticker FROM watchlist")
        rows = cursor.fetchall()
        conn.close()
        return [row['ticker'] for row in rows]
    
    def clear_old_cache(self, days: int = 365) -> int:
        """古いキャッシュデータを削除"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        cursor.execute("DELETE FROM price_cache WHERE date < ?", (cutoff_date,))
        deleted = cursor.rowcount
        
        conn.commit()
        conn.close()
        return deleted
    
    # ==================== ポートフォリオ操作 ====================
    
    def add_portfolio_item(self, ticker: str, shares: float, avg_cost: float,
                          purchase_date: str = None, notes: str = None) -> bool:
        """ポートフォリオに銘柄を追加"""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO portfolio (ticker, shares, avg_cost, purchase_date, notes)
                VALUES (?, ?, ?, ?, ?)
            """, (ticker.upper(), shares, avg_cost, purchase_date, notes))
            conn.commit()
            return True
        except Exception as e:
            print(f"Error adding portfolio item: {e}")
            return False
        finally:
            conn.close()
    
    def update_portfolio_item(self, item_id: int, shares: float = None, 
                             avg_cost: float = None, notes: str = None) -> bool:
        """ポートフォリオアイテムを更新"""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            updates = []
            params = []
            if shares is not None:
                updates.append("shares = ?")
                params.append(shares)
            if avg_cost is not None:
                updates.append("avg_cost = ?")
                params.append(avg_cost)
            if notes is not None:
                updates.append("notes = ?")
                params.append(notes)
            
            if updates:
                updates.append("updated_at = CURRENT_TIMESTAMP")
                params.append(item_id)
                cursor.execute(f"""
                    UPDATE portfolio SET {', '.join(updates)} WHERE id = ?
                """, params)
                conn.commit()
            return True
        except Exception as e:
            print(f"Error updating portfolio item: {e}")
            return False
        finally:
            conn.close()
    
    def delete_portfolio_item(self, item_id: int) -> bool:
        """ポートフォリオからアイテムを削除"""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM portfolio WHERE id = ?", (item_id,))
            conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            print(f"Error deleting portfolio item: {e}")
            return False
        finally:
            conn.close()
    
    def get_portfolio(self) -> List[Dict]:
        """ポートフォリオを取得"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT p.*, w.name as ticker_name
            FROM portfolio p
            LEFT JOIN watchlist w ON p.ticker = w.ticker
            ORDER BY p.created_at DESC
        """)
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    def get_portfolio_summary(self) -> Dict:
        """ポートフォリオのサマリーを取得"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                ticker,
                SUM(shares) as total_shares,
                SUM(shares * avg_cost) / SUM(shares) as weighted_avg_cost,
                SUM(shares * avg_cost) as total_cost
            FROM portfolio
            GROUP BY ticker
        """)
        rows = cursor.fetchall()
        conn.close()
        return {row['ticker']: dict(row) for row in rows}
    
    # ==================== シグナルキャッシュ操作 ====================
    
    def save_signal_cache(self, ticker: str, signal_data: Dict) -> bool:
        """シグナルデータをキャッシュに保存"""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO signal_cache 
                (ticker, price, change, rsi, rsi_signal, ma_signal, macd_signal, 
                 bb_signal, vol_signal, total_score, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                ticker.upper(),
                signal_data.get('price'),
                signal_data.get('change'),
                signal_data.get('rsi'),
                signal_data.get('rsi_signal'),
                signal_data.get('ma_signal'),
                signal_data.get('macd_signal'),
                signal_data.get('bb_signal'),
                signal_data.get('vol_signal'),
                signal_data.get('total_score')
            ))
            conn.commit()
            return True
        except Exception as e:
            print(f"Error saving signal cache: {e}")
            return False
        finally:
            conn.close()
    
    def save_signals_batch(self, signals: Dict[str, Dict]) -> int:
        """複数のシグナルデータを一括保存"""
        conn = self._get_connection()
        cursor = conn.cursor()
        saved = 0
        try:
            for ticker, signal_data in signals.items():
                cursor.execute("""
                    INSERT OR REPLACE INTO signal_cache 
                    (ticker, price, change, rsi, rsi_signal, ma_signal, macd_signal, 
                     bb_signal, vol_signal, total_score, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    ticker.upper(),
                    signal_data.get('price'),
                    signal_data.get('change'),
                    signal_data.get('rsi'),
                    signal_data.get('rsi_signal'),
                    signal_data.get('ma_signal'),
                    signal_data.get('macd_signal'),
                    signal_data.get('bb_signal'),
                    signal_data.get('vol_signal'),
                    signal_data.get('total_score')
                ))
                saved += 1
            conn.commit()
            return saved
        except Exception as e:
            print(f"Error saving signals batch: {e}")
            return saved
        finally:
            conn.close()
    
    def get_signal_cache(self, ticker: str = None, max_age_minutes: int = 30) -> List[Dict]:
        """シグナルキャッシュを取得"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if ticker:
            cursor.execute("""
                SELECT * FROM signal_cache 
                WHERE ticker = ? 
                AND datetime(updated_at) > datetime('now', ? || ' minutes')
            """, (ticker.upper(), f'-{max_age_minutes}'))
        else:
            cursor.execute("""
                SELECT s.*, w.name as ticker_name, w.market
                FROM signal_cache s
                LEFT JOIN watchlist w ON s.ticker = w.ticker
                WHERE datetime(s.updated_at) > datetime('now', ? || ' minutes')
                ORDER BY s.total_score DESC
            """, (f'-{max_age_minutes}',))
        
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    def get_signal_cache_all(self) -> List[Dict]:
        """全てのシグナルキャッシュを取得（有効期限なし）"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT s.*, w.name as ticker_name, w.market
            FROM signal_cache s
            LEFT JOIN watchlist w ON s.ticker = w.ticker
            ORDER BY s.total_score DESC
        """)
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    def clear_signal_cache(self) -> int:
        """シグナルキャッシュをクリア"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM signal_cache")
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        return deleted

    # ==================== バックテスト操作 ====================
    
    def backtest_get_balance(self) -> Dict:
        """バックテストの現在残高を取得"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # 最新の残高を取得
        cursor.execute("""
            SELECT * FROM backtest_balance ORDER BY id DESC LIMIT 1
        """)
        row = cursor.fetchone()
        
        if row:
            result = dict(row)
        else:
            # 初期値: 100万円
            result = {'cash': 1000000, 'stock_value': 0, 'total_value': 1000000}
        
        conn.close()
        return result
    
    def backtest_get_portfolio(self) -> List[Dict]:
        """バックテストのポートフォリオを取得"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM backtest_portfolio WHERE shares > 0
        """)
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    def backtest_get_transactions(self, limit: int = 50) -> List[Dict]:
        """バックテストの取引履歴を取得"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM backtest_transactions 
            ORDER BY created_at DESC LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    def backtest_get_balance_history(self) -> List[Dict]:
        """バックテストの資産推移を取得"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM backtest_balance ORDER BY created_at
        """)
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    def backtest_buy(self, ticker: str, amount: float, price: float, signal_score: float, reason: str) -> bool:
        """バックテスト: 買い注文"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            # 現在の残高取得
            balance = self.backtest_get_balance()
            cash = balance['cash']
            
            # 資金チェック
            if cash < amount:
                amount = cash  # 残高分だけ買う
            
            if amount <= 0:
                return False
            
            shares = amount / price
            
            # 既存ポジションをチェック
            cursor.execute("SELECT * FROM backtest_portfolio WHERE ticker = ?", (ticker,))
            existing = cursor.fetchone()
            
            if existing:
                # 平均取得単価を更新
                old_shares = existing['shares']
                old_cost = existing['avg_cost']
                new_shares = old_shares + shares
                new_avg_cost = ((old_shares * old_cost) + (shares * price)) / new_shares
                
                cursor.execute("""
                    UPDATE backtest_portfolio 
                    SET shares = ?, avg_cost = ?, current_price = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE ticker = ?
                """, (new_shares, new_avg_cost, price, ticker))
            else:
                cursor.execute("""
                    INSERT INTO backtest_portfolio (ticker, shares, avg_cost, current_price)
                    VALUES (?, ?, ?, ?)
                """, (ticker, shares, price, price))
            
            # 取引履歴に追加
            cursor.execute("""
                INSERT INTO backtest_transactions (ticker, action, shares, price, amount, signal_score, reason)
                VALUES (?, 'BUY', ?, ?, ?, ?, ?)
            """, (ticker, shares, price, amount, signal_score, reason))
            
            # 残高更新
            new_cash = cash - amount
            stock_value = self._calc_stock_value(cursor, price_map={ticker: price})
            total_value = new_cash + stock_value
            
            cursor.execute("""
                INSERT INTO backtest_balance (cash, stock_value, total_value)
                VALUES (?, ?, ?)
            """, (new_cash, stock_value, total_value))
            
            conn.commit()
            return True
        except Exception as e:
            print(f"Backtest buy error: {e}")
            return False
        finally:
            conn.close()
    
    def backtest_sell(self, ticker: str, sell_ratio: float, price: float, signal_score: float, reason: str) -> bool:
        """バックテスト: 売り注文 (sell_ratio: 0.5=半分, 1.0=全部)"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            # 既存ポジションをチェック
            cursor.execute("SELECT * FROM backtest_portfolio WHERE ticker = ?", (ticker,))
            existing = cursor.fetchone()
            
            if not existing or existing['shares'] <= 0:
                return False
            
            shares_to_sell = existing['shares'] * sell_ratio
            amount = shares_to_sell * price
            remaining_shares = existing['shares'] - shares_to_sell
            
            if remaining_shares < 0.0001:
                # 全売却
                cursor.execute("DELETE FROM backtest_portfolio WHERE ticker = ?", (ticker,))
            else:
                cursor.execute("""
                    UPDATE backtest_portfolio 
                    SET shares = ?, current_price = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE ticker = ?
                """, (remaining_shares, price, ticker))
            
            # 取引履歴に追加
            cursor.execute("""
                INSERT INTO backtest_transactions (ticker, action, shares, price, amount, signal_score, reason)
                VALUES (?, 'SELL', ?, ?, ?, ?, ?)
            """, (ticker, shares_to_sell, price, amount, signal_score, reason))
            
            # 残高更新
            balance = self.backtest_get_balance()
            new_cash = balance['cash'] + amount
            stock_value = self._calc_stock_value(cursor, price_map={ticker: price})
            total_value = new_cash + stock_value
            
            cursor.execute("""
                INSERT INTO backtest_balance (cash, stock_value, total_value)
                VALUES (?, ?, ?)
            """, (new_cash, stock_value, total_value))
            
            conn.commit()
            return True
        except Exception as e:
            print(f"Backtest sell error: {e}")
            return False
        finally:
            conn.close()
    
    def _calc_stock_value(self, cursor, price_map: Dict[str, float] = None) -> float:
        """保有株の時価総額を計算"""
        cursor.execute("SELECT ticker, shares, current_price FROM backtest_portfolio WHERE shares > 0")
        rows = cursor.fetchall()
        
        total = 0
        for row in rows:
            price = price_map.get(row['ticker'], row['current_price']) if price_map else row['current_price']
            total += row['shares'] * price
        
        return total
    
    def backtest_update_prices(self, price_map: Dict[str, float]) -> None:
        """バックテスト: ポートフォリオの現在価格を更新"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        for ticker, price in price_map.items():
            cursor.execute("""
                UPDATE backtest_portfolio SET current_price = ?, updated_at = CURRENT_TIMESTAMP
                WHERE ticker = ?
            """, (price, ticker))
        
        conn.commit()
        conn.close()
    
    def backtest_reset(self, initial_cash: float = 1000000) -> bool:
        """バックテストをリセット"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("DELETE FROM backtest_portfolio")
            cursor.execute("DELETE FROM backtest_transactions")
            cursor.execute("DELETE FROM backtest_balance")
            
            # 初期残高を設定
            cursor.execute("""
                INSERT INTO backtest_balance (cash, stock_value, total_value)
                VALUES (?, 0, ?)
            """, (initial_cash, initial_cash))
            
            conn.commit()
            return True
        except Exception as e:
            print(f"Backtest reset error: {e}")
            return False
        finally:
            conn.close()


# シングルトンインスタンス
_db_instance: Optional[DatabaseManager] = None

def get_db() -> DatabaseManager:
    """データベースマネージャーのシングルトンインスタンスを取得"""
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseManager()
    return _db_instance
