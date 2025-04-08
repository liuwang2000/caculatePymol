#!/bin/bash

echo "正在检查中文字体安装情况..."

# 检查系统类型
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$NAME
elif [ -f /etc/redhat-release ]; then
    OS="Rocky Linux/RHEL/CentOS"
else
    OS="Unknown"
fi

echo "检测到系统: $OS"

# Rocky Linux/RHEL/CentOS系列的中文字体包
FONT_PACKAGES=("wqy-microhei-fonts" "wqy-zenhei-fonts" "google-noto-sans-cjk-sc-fonts" "google-noto-sans-cjk-tc-fonts")
MISSING_PACKAGES=()

for pkg in "${FONT_PACKAGES[@]}"; do
    if ! rpm -q $pkg &> /dev/null; then
        MISSING_PACKAGES+=($pkg)
    fi
done

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo "未找到以下中文字体包: ${MISSING_PACKAGES[*]}"
    echo "这可能导致图表中的中文显示不正确"
    
    read -p "是否要安装这些字体包? (y/n): " INSTALL
    if [[ "$INSTALL" == "y" || "$INSTALL" == "Y" ]]; then
        echo "正在安装中文字体包..."
        # 检查使用dnf还是yum
        if command -v dnf &> /dev/null; then
            sudo dnf install -y ${MISSING_PACKAGES[*]}
        else
            sudo yum install -y ${MISSING_PACKAGES[*]}
        fi
        
        # 更新字体缓存
        fc-cache -fv
        
        echo "字体安装完成!"
        echo "请重新运行脚本以使用新安装的字体"
    else
        echo "未安装字体，图表中文可能无法正确显示"
    fi
else
    echo "已找到所需的中文字体包，无需额外安装"
fi

# 提供替代解决方案
echo ""
echo "如果仍然遇到中文显示问题，您也可以尝试："
echo "1. 安装更多中文字体："
echo "   sudo dnf install wqy-microhei-fonts wqy-zenhei-fonts google-noto-sans-cjk-sc-fonts"
echo "2. 在Python代码中使用matplotlib.rcParams[\"font.sans-serif\"] = [\"WenQuanYi Micro Hei\"]"
echo "3. 使用matplotlib.use(\"Agg\")后端"
echo ""
echo "您也可以尝试安装EPEL仓库以获取更多字体："
echo "sudo dnf install epel-release"
echo "sudo dnf install cjkuni-uming-fonts"
echo "" 