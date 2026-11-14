"""Create a VAPID key pair for Web Push (#22, P2) and write it into .env (repo root).

    backend/.venv/bin/python scripts/gen_vapid.py

- Never prints the private key. Refuses to overwrite keys that are already in .env.
- Then set VAPID_SUBJECT in .env to a contact the push services can reach (mailto: or https:).
- .env is gitignored and the pre-commit hook blocks it. A new key pair invalidates every existing
  subscription (devices must subscribe again), so keep the pair once the demo is set up.
"""

import base64
import re
import sys
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

ENV = Path(__file__).resolve().parents[1] / ".env"


def b64url(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


def main() -> int:
    text = ENV.read_text() if ENV.exists() else ""
    if re.search(r"^VAPID_PRIVATE_KEY=\S", text, re.M):
        print(f"{ENV} already has VAPID keys – not overwriting.")
        return 1
    key = ec.generate_private_key(ec.SECP256R1())
    private = b64url(key.private_numbers().private_value.to_bytes(32, "big"))
    public = b64url(key.public_key().public_bytes(
        serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint))
    lines = [ln for ln in text.splitlines() if not re.match(r"^#?\s*VAPID_(PUBLIC|PRIVATE)_KEY=", ln)]
    lines += [f"VAPID_PUBLIC_KEY={public}", f"VAPID_PRIVATE_KEY={private}"]
    if not re.search(r"^VAPID_SUBJECT=\S", text, re.M):
        lines.append("# VAPID_SUBJECT=mailto:   <- fill in a contact address, then uncomment")
    ENV.write_text("\n".join(lines) + "\n")
    ENV.chmod(0o600)
    print(f"Wrote a VAPID key pair to {ENV} (private key not shown).")
    print("Next: set VAPID_SUBJECT in .env, restart the backend, and GET /api/patients/1/push shows enabled: true.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
