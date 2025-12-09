@echo off
chcp 65001 > nul
echo ========================================
echo    株価分析ツール - 初期セットアップ
echo ========================================
echo.

cd /d "%~dp0"

:: Pythonが利用可能か確認
python --version > nul 2>&1
if errorlevel 1 (
    echo [エラー] Pythonが見つかりません
    echo https://www.python.org/ からPythonをインストールしてください
    pause
    exit /b 1
)

echo [1/4] 必要なパッケージをインストール中...
pip install -r requirements.txt

echo.
echo [2/4] LightGBMをインストール中...
pip install lightgbm

echo.
echo [3/4] archをインストール中...
pip install arch

echo.
echo [4/4] セットアップ完了!
echo.
echo ========================================
echo   run.bat をダブルクリックで起動できます
echo ========================================
pause
