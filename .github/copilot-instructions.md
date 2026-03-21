# Alway on, global instructions

- Never run a terminal process in the background. If you need to run a process for a long time, use a tool like `tmux` or `screen` to manage it in the foreground.
- If a development phase is complete, suggest pushing to github to create a checkpoint. For example, after working in one directory, starting work on a different aspect of the code may be a good time to push to github.
- Never create a fallback. Code should fail verbosely if the implementation does not work.

