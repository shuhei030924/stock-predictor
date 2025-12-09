@echo off
chcp 65001 > nul
echo ========================================
echo    株価分析ツール 起動中...
echo ========================================
echo.

cd /d "%~dp0"

:: Pythonが利用可能か確認
python --version > nul 2>&1
if errorlevel 1 (
    echo [エラー] Pythonが見つかりません
    echo Pythonをインストールしてください
    pause
    exit /b 1
)

:: Streamlitが利用可能か確認
python -m streamlit --version > nul 2>&1
if errorlevel 1 (
    echo [情報] Streamlitをインストール中...
    pip install streamlit
)

echo [起動] http://localhost:8501 でアプリを開きます
echo [終了] Ctrl+C で終了できます
echo.

:: ブラウザを自動で開く
start http://localhost:8501

:: Streamlitアプリを起動
python -m streamlit run app.py --server.headless true

pause
