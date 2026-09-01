#!/usr/bin/env bash

set -u

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if ! cd -- "$script_dir"; then
  echo "无法进入项目目录：$script_dir"
  read -r -p "按回车键关闭窗口..." _
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "未找到 uv，请先安装 uv 并将它加入 PATH。"
  read -r -p "按回车键关闭窗口..." _
  exit 1
fi

echo "正在启动本地语料管理服务..."
echo "管理页将自动在浏览器中打开；按 Ctrl+C 可停止服务。"
echo

uv run xifeng corpus manage
status=$?

if [[ $status -eq 0 || $status -eq 130 ]]; then
  exit 0
fi

echo
echo "服务启动失败（退出码：$status）。"
read -r -p "按回车键关闭窗口..." _
exit "$status"
