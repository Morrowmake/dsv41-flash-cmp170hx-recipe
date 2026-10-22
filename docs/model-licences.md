# Model licences

Model weights are separate from the MIT recipe and Apache-2.0 engine. Neither
software licence overrides a checkpoint's terms. Downloads come directly from
Hugging Face at the revisions this release pins; the recipe does not distribute
weights in its container image. The hybrid checkpoint the server reads is built
on your machine as links into the two downloads, with a config, index and
quantization config derived from them. Retain the downloaded cards and record
the checkpoint revisions if you redistribute a permitted copy.

## DeepSeek-V4.1-Flash-1017

[deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
at revision `1bae0e1088993b9b82de08f59de5be5a34e42170` supplies every weight
but the routed experts, and the Engram tables. It is MIT.

## EXL3 2.9 bpw export

[LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw)
at revision `c178a01284fc5c0e859d8668cffaac81ca8f4042` supplies the routed
experts, the tokenizer and the model config. Its card declares `license: mit`
with `base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017`, and its LICENSE file
is the base model's notice, preserved here:

```text
MIT License

Copyright (c) 2023 DeepSeek

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

Include this notice and the quantiser's supplied notices with any
redistributed copy of either checkpoint or of the hybrid built from them.
