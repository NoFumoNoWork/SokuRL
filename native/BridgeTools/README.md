# Host Hash Helper

Optional **x64** DLL for the project's x64 Python consumer, not a game module.
Never inject this DLL into the x86 game. Build from the repository root:

```powershell
cmd /c native\BridgeTools\build_host.cmd
```

`bridge_shared.HASH_BACKEND` reports `cpp-fnv64` when the built helper is present,
otherwise `python`. Both implement the same FNV-1a-64 payload and support old
raw-frame layouts. Explicit `backend="cpp"` fails if the helper is absent.
Run `tests/test_bridge_host_cost.py` to verify hash and owned bulk-ring copies.
Build products stay in `build/`; this directory is reproducible, not authoritative.
