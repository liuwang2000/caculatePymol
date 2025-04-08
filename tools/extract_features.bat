@echo off
echo 开始提取特征数据...

REM 设置默认参数
set DATA_FILE=./trainData/analyze_pdb_merged_20250403_164250.csv
set RESULTS_FILE=./models/rf_feature_models/feature_evaluation_results_20250403_204300.csv
set OUTPUT_DIR=./output
set RANK=1
set TOP_N=0
set INCLUDE_SEQUENCE=
set FORMAT=combined
set INCLUDE_METADATA=

REM 解析命令行参数
:parse
if "%~1"=="" goto execute
if /i "%~1"=="--data" set DATA_FILE=%~2& shift & shift & goto parse
if /i "%~1"=="--results" set RESULTS_FILE=%~2& shift & shift & goto parse
if /i "%~1"=="--output" set OUTPUT_DIR=%~2& shift & shift & goto parse
if /i "%~1"=="--rank" set RANK=%~2& shift & shift & goto parse
if /i "%~1"=="--top_n" set TOP_N=%~2& shift & shift & goto parse
if /i "%~1"=="--include_sequence" set INCLUDE_SEQUENCE=--include_sequence& shift & goto parse
if /i "%~1"=="--format" set FORMAT=%~2& shift & shift & goto parse
if /i "%~1"=="--include_metadata" set INCLUDE_METADATA=--include_metadata& shift & goto parse
echo 未知参数: %~1
shift
goto parse

:execute
echo 使用参数:
echo 数据文件: %DATA_FILE%
echo 结果文件: %RESULTS_FILE%
echo 输出目录: %OUTPUT_DIR%
echo 特征排名: %RANK%
echo 提取前N个: %TOP_N%
echo 包含序列: %INCLUDE_SEQUENCE%
echo 输出格式: %FORMAT%
echo 包含元数据: %INCLUDE_METADATA%

REM 执行Python脚本
python extract_features_by_rank.py --data "%DATA_FILE%" --results "%RESULTS_FILE%" --output "%OUTPUT_DIR%" --rank %RANK% --top_n %TOP_N% --format %FORMAT% %INCLUDE_SEQUENCE% %INCLUDE_METADATA%

echo 处理完成！ 