#!/bin/sh
set -eu

REPO_NAME="${1:-documind-recruiter-demo}"

echo "This script publishes ONLY this standalone recruiter-demo folder."
echo "The GitHub repository will be PUBLIC: ${REPO_NAME}"
echo "No parent project files, .env files, databases, or model files are included."
printf "Continue? Type YES to publish: "
read answer
[ "$answer" = "YES" ] || { echo "Cancelled."; exit 0; }

command -v git >/dev/null 2>&1 || { echo "Git is required."; exit 1; }
command -v gh >/dev/null 2>&1 || {
  echo "GitHub CLI (gh) is required. Install: brew install gh"
  exit 1
}
gh auth status >/dev/null 2>&1 || {
  echo "Sign in first using: gh auth login"
  exit 1
}

git init
git branch -M main
git add app.py requirements.txt README.md .gitignore run_local.sh publish_to_github.sh data/
git diff --cached --check -- . ':!*.pdf'
git commit -m "Build DocuMind AI recruiter demo"
gh repo create "$REPO_NAME" --public --source=. --remote=origin --push

echo
echo "Published successfully. Open the repository URL printed above."
echo "Next: https://share.streamlit.io/ → deploy app.py from this repository."
