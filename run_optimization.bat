@echo off
echo 特征选择优化工具启动器
echo ====================
echo.

:: 检查Python是否安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未检测到Python安装。请安装Python 3.6+后再运行此脚本。
    goto :end
)

:: 检查必要的库是否已安装
echo 检查依赖项...
python -c "import pandas, numpy, sklearn, matplotlib, seaborn, joblib, tqdm" >nul 2>&1
if %errorlevel% neq 0 (
    echo [警告] 缺少部分依赖库，正在安装...
    pip install scikit-learn pandas numpy matplotlib seaborn joblib tqdm
    if %errorlevel% neq 0 (
        echo [错误] 安装依赖项失败，请手动安装。
        goto :end
    )
    echo 依赖项安装完成！
) else (
    echo 所有依赖项已安装。
)

:: 检测CPU核心数并显示
for /f "tokens=2 delims==" %%a in ('wmic cpu get NumberOfCores /value') do (
    set cores=%%a
)
set /a used_cores=%cores%-1
echo 检测到 %cores% 个CPU核心，将使用 %used_cores% 个核心进行并行处理

echo.
echo 请选择要运行的优化方式:
echo 1. 基本特征选择优化
echo 2. 高级特征选择优化
echo 3. 同时运行两种优化方法并对比
echo.

set /p choice=请输入选项编号 (1-3): 

echo.
set /p data_file=请输入训练数据CSV文件路径 (默认: trainData/merged_data_20250401_213334.csv): 
if "%data_file%"=="" set data_file=trainData/merged_data_20250401_213334.csv

echo.
set /p exclude_aa=是否排除氨基酸比例特征? (y/n, 默认: y): 
if /i "%exclude_aa%"=="" set exclude_aa=y

echo.
set /p min_features=最小特征数量 (默认: 5): 
if "%min_features%"=="" set min_features=5

echo.
set /p max_features=最大特征数量 (默认: 15): 
if "%max_features%"=="" set max_features=15

set exclude_aa_flag=
if /i "%exclude_aa%"=="n" set exclude_aa_flag=--include_aa

if "%choice%"=="1" (
    echo.
    echo 正在运行基本特征选择优化...
    python feature_selection_optimizer.py --data "%data_file%" --output ./models --min_features %min_features% --max_features %max_features% %exclude_aa_flag%
) else if "%choice%"=="2" (
    echo.
    set /p interactions=是否创建特征交互项? (y/n, 默认: n): 
    set /p polynomials=是否创建多项式特征? (y/n, 默认: n): 
    
    set cmd=python advanced_feature_optimizer.py --data "%data_file%" --output ./models_advanced --min_features %min_features% --max_features %max_features% %exclude_aa_flag%
    if /i "%interactions%"=="y" set cmd=%cmd% --interactions
    if /i "%polynomials%"=="y" set cmd=%cmd% --polynomials
    
    echo 正在运行高级特征选择优化...
    %cmd%
) else if "%choice%"=="3" (
    echo.
    set /p interactions=是否创建特征交互项? (y/n, 默认: n): 
    set /p polynomials=是否创建多项式特征? (y/n, 默认: n): 
    
    set cmd=python run_feature_optimization.py --data "%data_file%" --mode both --min_features %min_features% --max_features %max_features% %exclude_aa_flag%
    if /i "%interactions%"=="y" set cmd=%cmd% --interactions
    if /i "%polynomials%"=="y" set cmd=%cmd% --polynomials
    
    echo 正在运行两种优化方法...
    %cmd%
) else (
    echo [错误] 无效的选项: %choice%
    goto :end
)

echo.
echo 优化完成！结果已保存到相应目录。
echo 请查看生成的结果文件和可视化图表。

:end
echo.
pause 