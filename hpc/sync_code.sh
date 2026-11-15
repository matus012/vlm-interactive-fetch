#!/bin/bash
# Ship the committed HEAD of this checkout (main repo or a worktree) to PERUN as a clean clone at that commit.
#   hpc/sync_code.sh [DEST]     DEST default $PROJ/code; a track's dev clone: $PROJ/dev/<track>/code
# Refuses a dirty tree. Sends a git bundle of HEAD (works from worktrees, where .git is a file), then on <login-node>
# clones DEST from it once / fetches it afterwards, checks out detached at HEAD's commit, and requires
# `git status --porcelain` to be empty there. GPU nodes have no git: sync on <login-node>, jobs only read DEST.
# Never sync into a clone a running job uses.
set -euo pipefail
LOGIN=${PERUN_LOGIN:?set PERUN_LOGIN to your cluster login node}
PROJ=${PROJECT_DIR:?set PROJECT_DIR}
SSH=(ssh -o BatchMode=yes -o LogLevel=ERROR "$LOGIN")
TOP=$(git rev-parse --show-toplevel)
DEST=${1:-$PROJ/code}
[[ $DEST == "$PROJ"/* ]] || { echo "DEST must be under $PROJ"; exit 2; }
[ -z "$(git -C "$TOP" status --porcelain)" ] || { echo "refusing: $TOP has uncommitted changes"; git -C "$TOP" status --short; exit 1; }
SHA=$(git -C "$TOP" rev-parse HEAD)
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
git -C "$TOP" branch -f _sync "$SHA" 2>/dev/null || true
git -C "$TOP" bundle create -q "$TMP/code.bundle" _sync
"${SSH[@]}" "mkdir -p $(dirname "$DEST")"
scp -q -o BatchMode=yes -o LogLevel=ERROR "$TMP/code.bundle" "$LOGIN:$DEST.bundle"
"${SSH[@]}" bash -s -- "$DEST" "$SHA" <<'REMOTE'
set -euo pipefail
DEST=$1 SHA=$2
if [ -d "$DEST/.git" ]; then
    [ -z "$(git -C "$DEST" status --porcelain)" ] || { echo "refusing: $DEST was edited on PERUN"; git -C "$DEST" status --short; exit 1; }
else
    git init -q "$DEST"
fi
cd "$DEST"
git fetch -q --force "$DEST.bundle" '+refs/heads/_sync:refs/remotes/bundle/_sync'
git -c advice.detachedHead=false checkout -q --detach "$SHA"
[ "$(git rev-parse HEAD)" = "$SHA" ] && [ -z "$(git status --porcelain)" ] || { echo "checkout of $SHA not clean"; exit 1; }
echo "PERUN $DEST at $(git log -1 --format='%H %cI %s')"
REMOTE
echo "synced commit $SHA -> $DEST"
