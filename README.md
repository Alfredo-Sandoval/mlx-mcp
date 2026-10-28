<div align="center">

# mlx-mcp

[![PyPI](https://img.shields.io/pypi/v/mlx-mcp?style=flat-square&labelColor=24292f&logo=pypi&logoColor=white&color=3775A9&label=PyPI)](https://pypi.org/project/mlx-mcp/)
[![Python 3.13 | 3.14](https://img.shields.io/badge/Python-3.13%20%7C%203.14-3776AB?style=flat-square&labelColor=24292f&logo=python&logoColor=white)](https://pypi.org/project/mlx-mcp/)
[![macOS 14+ on Apple silicon](https://img.shields.io/badge/macOS%2014%2B-Apple%20silicon-6e7781?style=flat-square&labelColor=24292f&logo=apple&logoColor=white)](#install)
[![MCP stdio server](https://img.shields.io/badge/MCP-stdio%20server-8250df?style=flat-square&labelColor=24292f&logo=modelcontextprotocol&logoColor=white)](https://modelcontextprotocol.io)
[![MIT license](https://img.shields.io/badge/License-MIT-2da44e?style=flat-square&labelColor=24292f)](https://github.com/Alfredo-Sandoval/mlx-mcp/blob/main/LICENSE)

</div>

<!-- mcp-name: io.github.Alfredo-Sandoval/mlx-mcp -->

mlx-mcp gives Claude Code, Codex and other coding agents the
[MLX](https://ml-explore.github.io/mlx/build/html/),
[MLX Data](https://ml-explore.github.io/mlx-data/build/html/) and
[MLX LM](https://github.com/ml-explore/mlx-lm) docs, so your agent can check
the docs before it writes MLX code instead of working from memory.

It runs on your Mac, needs no account or API key, and works offline after the
first search.

An independent project, not affiliated with Apple or the MLX maintainers.

Maintained by [Alfredo Sandoval](https://github.com/Alfredo-Sandoval) and
[Joseph Sandoval](https://github.com/sandovaljoseph).

## Install

You need an Apple silicon Mac with macOS 14 or later, and
[uv](https://docs.astral.sh/uv/getting-started/installation/).

Add mlx-mcp to your agent below. Every client runs the same command,
`uvx --python 3.13 mlx-mcp serve`, so there's nothing else to install. uv
downloads Python 3.13 if you don't have it.

### Claude Code

```bash
claude mcp add --scope user mlx-docs -- uvx --python 3.13 mlx-mcp serve
```

### Codex

```bash
codex mcp add mlx-docs -- uvx --python 3.13 mlx-mcp serve
```

### Cursor

[<img src="https://cursor.com/deeplink/mcp-install-dark.svg" alt="Install in Cursor">](https://cursor.com/en/install-mcp?name=mlx-docs&config=eyJjb21tYW5kIjoidXZ4IiwiYXJncyI6WyItLXB5dGhvbiIsIjMuMTMiLCJtbHgtbWNwIiwic2VydmUiXX0%3D)

Or add it by hand in `~/.cursor/mcp.json`:

```json
{
	"mcpServers": {
		"mlx-docs": {
			"command": "uvx",
			"args": ["--python", "3.13", "mlx-mcp", "serve"]
		}
	}
}
```

### VS Code

[<img src="https://img.shields.io/badge/VS_Code-VS_Code?style=flat-square&label=Install%20Server&color=0098FF" alt="Install in VS Code">](https://insiders.vscode.dev/redirect?url=vscode%3Amcp%2Finstall%3F%257B%2522name%2522%253A%2522mlx-docs%2522%252C%2522command%2522%253A%2522uvx%2522%252C%2522args%2522%253A%255B%2522--python%2522%252C%25223.13%2522%252C%2522mlx-mcp%2522%252C%2522serve%2522%255D%257D)

Or add it by hand in `.vscode/mcp.json`:

```json
{
	"servers": {
		"mlx-docs": {
			"type": "stdio",
			"command": "uvx",
			"args": ["--python", "3.13", "mlx-mcp", "serve"]
		}
	}
}
```

### Other MCP clients

Add a stdio server named `mlx-docs` that runs this command:

```json
{
	"command": "uvx",
	"args": ["--python", "3.13", "mlx-mcp", "serve"]
}
```

## Use it

Ask your agent, for example: "Use mlx-mcp to port this code to MLX."

Your agent can also check which docs versions mlx-mcp has, and every page it
reads says which version it's from.

Your agent starts mlx-mcp when it needs it. The first start downloads about
65 MB, mostly MLX, and the first search downloads the 90 MB search model.
After that it works offline.

## Update

```bash
uvx --python 3.13 mlx-mcp@latest --version
```

Then start a new agent session.

## Uninstall

Remove `mlx-docs` from each agent you connected:

- Claude Code: `claude mcp remove --scope user mlx-docs`
- Codex: `codex mcp remove mlx-docs`
- Cursor, VS Code and other clients: delete the `mlx-docs` entry.

Then delete `~/.cache/mlx-mcp` to free the space.

## If something doesn't work

Run `uvx --python 3.13 mlx-mcp prepare` and start a new session. It downloads
anything missing and checks that search and fetch work. If that doesn't fix
it, [open an issue](https://github.com/Alfredo-Sandoval/mlx-mcp/issues) with
what the command printed.

If Cursor, VS Code or another app can't find `uvx`, run `command -v uvx` in a
terminal and use the path it prints as the command.

## Good to know

- The bundled docs may not match the MLX version you have installed.
  `uvx --python 3.13 mlx-mcp info` shows the versions mlx-mcp has.
  `uvx --python 3.13 mlx-mcp prepare --refresh` downloads the latest MLX and
  MLX Data docs. MLX LM stays at its bundled version. Start a new session
  after refreshing.
- mlx-mcp has the published docs, not the MLX source code. The docs don't
  always say which kernel or device runs an operation. For that, read the
  MLX source or measure it.
- `uvx --python 3.13 mlx-mcp --help` lists the commands.

## License

MIT. Includes text from the MLX, MLX Data and MLX LM projects, © Apple Inc.,
under the MIT License. See the
[third-party notices](https://github.com/Alfredo-Sandoval/mlx-mcp/blob/main/THIRD_PARTY_NOTICES.md).
To report a security problem, see
[Security](https://github.com/Alfredo-Sandoval/mlx-mcp/blob/main/SECURITY.md).
