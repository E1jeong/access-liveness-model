#!/bin/bash
# 다른 장비(Mac, WSL)와 서브노트북 간의 git 충돌 방지 및 pull 자동화 스크립트.
# 작업 중이던 임시 변경 사항이 있으면 stash한 뒤 pull(rebase)하고 다시 pop하여 충돌 없이 동기화합니다.
# 만약 강제로 원격 origin/master 상태로 일치시키고 싶다면 --force 옵션을 사용합니다.

set -e

FORCE=false
if [ "$1" == "--force" ] || [ "$1" == "-f" ]; then
    FORCE=true
fi

echo "=== Git Pull & Clean Sync ==="

if [ "$FORCE" = true ]; then
    echo "[!] Force reset option enabled."
    echo "[*] Fetching latest from origin..."
    git fetch origin
    echo "[*] Hard resetting local branch to origin/master..."
    git reset --hard origin/master
    echo "[*] Cleaning untracked files (excluding dataset/, model/, and venvs)..."
    git clean -fd -e dataset/ -e model/ -e .venv/ -e .venv-tf/
    echo "[+] Done. Working directory is completely synced with remote master."
else
    echo "[*] Checking status..."
    STATUS=$(git status --porcelain)
    if [ -n "$STATUS" ]; then
        echo "[*] Local modifications detected. Autostashing..."
        git stash push -m "Auto-stash before clean pull" -u
    fi

    echo "[*] Pulling latest changes (with rebase)..."
    git pull origin master --rebase

    if [ -n "$STATUS" ]; then
        echo "[*] Applying auto-stashed changes..."
        if git stash pop; then
            echo "[+] Stash popped successfully."
        else
            echo "[!] Conflict detected during stash pop. Please resolve conflicts manually."
            exit 1
        fi
    fi
    echo "[+] Done. Pulled successfully."
fi
