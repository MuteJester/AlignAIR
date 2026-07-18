---
name: Bug report
about: Report incorrect behaviour, a crash, or a wrong result
title: ''
labels: bug
assignees: ''

---

**What happened**
A clear description of the problem, and what you expected instead.

**Command**
The exact command you ran:

```bash
alignair predict --input ... --out ... --model ...
```

**Version**
Output of `alignair --version` (e.g. `alignair 3.0.0`).

**Model**
The model id you used, pinned (e.g. `alignair-igh-human@1.0.0`), or the path to your `.alignair` file.

**Environment**
Paste the full output of `alignair doctor` (Python, PyTorch + CUDA/MPS, GenAIRR, parasail, cache dir):

```
<alignair doctor output>
```

**Minimal reproducer**
The smallest input that triggers it (a few reads is ideal). Attach the file, or paste a synthetic
sequence that reproduces the same error. If the input is sensitive, a synthetic read that triggers it
is perfect.

**Additional context**
Anything else that helps - a stack trace, a screenshot, or the relevant rows of the output TSV.
