# Third-party notices

## MLX, MLX Data, and MLX LM

Release packages of mlx-mcp include a search index built from the published
[MLX documentation](https://ml-explore.github.io/mlx/build/html/) and
[MLX Data documentation](https://ml-explore.github.io/mlx-data/build/html/).
The index also contains [MLX LM](https://github.com/ml-explore/mlx-lm) guide text,
extracted API signatures/docstrings, and captured CLI output. All three projects
use the MIT License below. MLX Data's copyright notice is
“Copyright © 2023 Apple Inc. All rights reserved.”

The copied MLX LM signatures and docstrings also carry these file-level notices:

```text
Copyright © 2023-2024 Apple Inc.
Copyright © 2024 Apple Inc.
```

```text
MIT License

Copyright © 2023 Apple Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Search model and runtime

The MiniLM reranker is
[cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2),
which is released under the Apache License 2.0; its weights are downloaded
separately and not included. It runs on [MLX](https://github.com/ml-explore/mlx),
under the MIT License.
