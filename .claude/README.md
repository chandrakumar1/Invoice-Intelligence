<!--
What:     Documents the Claude Code project settings in this directory (JSON can't hold comments).
In:       settings.json, read by Claude Code when it runs in this repo.
Out:      Commits and pull requests made with Claude Code carry no attribution trailers.
Fails if: A personal setting in .claude/settings.local.json or ~/.claude/settings.json overrides it.
-->

# Claude Code settings

| File | Purpose |
|---|---|
| `settings.json` | `attribution`: empty `commit` and `pr` text and `sessionUrl: false`, so no `Co-Authored-By`, "Generated with Claude Code" or session-link lines are added to commits or PRs. Uses the object form because older Claude Code versions reject `"attribution": false` and then ignore the whole file. |
