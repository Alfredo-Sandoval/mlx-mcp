# Security

## Report a vulnerability

Email [Alfredo.sandoval@protonmail.com](mailto:Alfredo.sandoval@protonmail.com).
Don't post it in a public issue or pull request. Include your mlx-mcp version
(`mlx-mcp --version`), your macOS version, the agent you use, and the steps
that cause the problem. Leave out tokens and personal data.

mlx-mcp is in alpha, so fixes go into the latest release only.

## What mlx-mcp can access

- The tools only read the docs stored on your Mac. They can't run commands,
  open websites or read your files.
- It talks only to the agent that started it, over stdin and stdout. It opens
  no network port.
- It downloads docs only from the official MLX sources, and models only from
  fixed versions on Hugging Face.
- Docs pages are untrusted text. Your agent's own permission settings are what
  stop it from acting on instructions hidden in them.
- It keeps no record of what your agent asks or gets back.
