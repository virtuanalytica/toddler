"""CLI: keygen / revoke / list / serve.

  python3 -m jevserver keygen --label toddler   # prints the key once
  python3 -m jevserver revoke <key_id>
  python3 -m jevserver list
  python3 -m jevserver serve --port 8092         # binds 127.0.0.1 by default
"""

from __future__ import annotations

import argparse
import json

from jevserver.keys import KeyStore


def main() -> int:
    p = argparse.ArgumentParser(prog="jevserver")
    sub = p.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("keygen")
    k.add_argument("--label", required=True)
    r = sub.add_parser("revoke")
    r.add_argument("key_id")
    sub.add_parser("list")
    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--allow-remote", action="store_true",
                   help="allow a non-loopback bind; keys travel over plain HTTP, so only behind TLS")
    s.add_argument("--cache-decimals", type=int, default=2,
                   help="round state floats for the cache key (default 2; -1 = exact keys)")
    s.add_argument("--cache-ttl", type=float, default=300.0,
                   help="seconds a cached answer stays valid (default 300; freshness bound of every cached answer)")
    s.add_argument("--port", type=int, default=8092)
    s.add_argument("--calibration", default="", help="path to a calibration JSON from calibrate.py")
    a = p.parse_args()
    store = KeyStore()
    if a.cmd == "keygen":
        key_id, key = store.create(a.label)
        print(f"key id: {key_id}\nAPI key (shown once, store it yourself): {key}")
    elif a.cmd == "revoke":
        print("revoked" if store.revoke(a.key_id) else "not found")
    elif a.cmd == "list":
        print(json.dumps(store.list(), indent=1))
    else:
        import uvicorn

        from jevserver.app import create_app
        calibrator = None
        if a.calibration:
            from jevserver.calibrate import load_calibrator
            calibrator = load_calibrator(a.calibration)
        if a.host not in ("127.0.0.1", "localhost", "::1") and not a.allow_remote:
            raise SystemExit("refusing non-loopback bind without --allow-remote (bearer keys over plain HTTP)")
        from jevserver.app import TTLCache
        if a.cache_ttl <= 0:
            raise SystemExit("--cache-ttl must be positive")
        uvicorn.run(create_app(calibrator=calibrator, cache=TTLCache(ttl_s=a.cache_ttl),
                               cache_decimals=None if a.cache_decimals < 0 else a.cache_decimals),
                    host=a.host, port=a.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
