#!/bin/bash
# self-driving-sim launchd 安装脚本
# 用法：bash launchd/install.sh
#
# 干 3 件事：
#   1) plutil -lint 验证 plist 合法
#   2) 拷到 ~/Library/LaunchAgents/
#   3) launchctl load -w 加载并开机自启
#
# 卸载：bash launchd/install.sh uninstall

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PLIST_SRC="$SCRIPT_DIR/com.mac.selfdrivingsim.webui.plist"
LABEL="com.mac.selfdrivingsim.webui"
PLIST_DEST="$HOME/Library/LaunchAgents/$LABEL.plist"

case "${1:-install}" in
  install)
    if [ ! -f "$PLIST_SRC" ]; then
      echo "❌ 找不到 $PLIST_SRC"; exit 1
    fi

    # 1) 验证
    plutil -lint "$PLIST_SRC" || { echo "❌ plist 语法错"; exit 1; }

    # 2) 拷到 LaunchAgents
    mkdir -p "$HOME/Library/LaunchAgents"
    cp "$PLIST_SRC" "$PLIST_DEST"

    # 3) 加载
    launchctl load -w "$PLIST_DEST"

    # 4) 等 4 秒，streamlit 启动稍慢
    sleep 4
    if curl -sf -o /dev/null http://localhost:8501/; then
      echo "✅ self-driving-sim WebUI 已在 8501 跑起来（launchd）"
    else
      echo "⚠️  plist 已加载但 8501 没响应，看日志："
      echo "   tail -F /tmp/selfdrivingsim-webui.{out,err}.log"
    fi
    echo ""
    echo "管理命令："
    echo "  状态:   launchctl list | grep $LABEL"
    echo "  停:     launchctl unload $PLIST_DEST"
    echo "  重启:   launchctl kickstart -k gui/\$(id -u)/$LABEL"
    echo "  日志:   tail -F /tmp/selfdrivingsim-webui.{out,err}.log"
    echo "  访问:   http://localhost:8501/"
    ;;

  uninstall)
    launchctl unload "$PLIST_DEST" 2>/dev/null || true
    rm -f "$PLIST_DEST"
    echo "✅ $LABEL 已卸载"
    ;;

  *)
    echo "用法: $0 {install|uninstall}"; exit 1
    ;;
esac
