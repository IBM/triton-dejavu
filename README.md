triton_dejavu
=================
Framework to try to reduce overhead to (close to) 0 for well known deployments.

Install:
```
pip install -e triton-dejavu/
```

```
docker build -f tests/Dockerfile . -t test-triton-dejavu
```

This small framework contributes two features:
1. Store and savely restore autotuner states 
2. `ConfigSpaces` to explore a defined space exhaustively


