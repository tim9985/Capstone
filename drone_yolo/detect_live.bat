@echo off
rem 항공 영상 실시간 탐지 테스트기 - 영상 파일을 이 파일 위로 끌어다 놓거나
rem   detect_live.bat "D:\경로\영상.mp4"
rem 가중치를 지정하려면 두 번째 인자부터 .pt 경로를 적는다 (비우면 weights 폴더에서 자동으로 찾는다)
setlocal
set PY=C:\Users\timjj\miniconda3\envs\drone\python.exe
cd /d "%~dp0"
if "%~1"=="" (
  set /p SRC=영상 경로를 입력하세요:
) else (
  set SRC=%~1
)
if "%~2"=="" (
  "%PY%" detect_live.py --source "%SRC%"
) else (
  "%PY%" detect_live.py --source "%SRC%" --weights %2 %3 %4 %5
)
endlocal
pause
