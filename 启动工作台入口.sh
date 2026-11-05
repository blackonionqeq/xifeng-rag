#!/usr/bin/env bash

set -u

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if ! cd -- "$script_dir"; then
  echo "无法进入项目目录：$script_dir"
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "未找到 uv，请先安装 uv 并将它加入 PATH。"
  exit 1
fi

echo "正在启动西风 RAG 项目控制台……"
echo "按 Ctrl+C 可停止服务。"
echo

uv run xifeng workbench
