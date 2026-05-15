"""
GEE Authentication Helper — run this script interactively in your terminal.

Usage:
    source venv/bin/activate
    python scripts/authenticate_gee.py

Steps:
    1. The script prints a URL.
    2. Open the URL in your browser.
    3. Log in with your personal Google account.
    4. Copy the verification code shown in the browser.
    5. Paste it here and press Enter.
    6. Done — credentials saved permanently.
"""
import sys

try:
    import ee
    from ee import oauth
except ImportError:
    print("ERROR: earthengine-api not installed. Run: pip install earthengine-api")
    sys.exit(1)

import secrets
import hashlib
import base64
import json
import os


def base64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def build_auth_url():
    """Build the OAuth2 authorization URL with PKCE."""
    code_verifier = base64url(secrets.token_bytes(32))
    code_challenge = base64url(
        hashlib.sha256(code_verifier.encode("ascii")).digest()
    )

    params = {
        "client_id": oauth.CLIENT_ID,
        "redirect_uri": oauth.REDIRECT_URI,
        "response_type": "code",
        "scope": " ".join(oauth.SCOPES),
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }

    from urllib.parse import urlencode
    url = f"https://accounts.google.com/o/oauth2/auth?{urlencode(params)}"
    return url, code_verifier


def exchange_code(auth_code: str, code_verifier: str) -> dict:
    """Exchange authorization code for tokens."""
    import urllib.request, urllib.parse

    data = urllib.parse.urlencode({
        "code": auth_code,
        "client_id": oauth.CLIENT_ID,
        "client_secret": oauth.CLIENT_SECRET,
        "redirect_uri": oauth.REDIRECT_URI,
        "grant_type": "authorization_code",
        "code_verifier": code_verifier,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://oauth2.googleapis.com/token",
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST"
    )

    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def save_credentials(token_data: dict):
    """Save refresh token in EE credentials format."""
    creds_path = oauth.get_credentials_path()
    os.makedirs(os.path.dirname(creds_path), exist_ok=True)

    creds = {
        "client_id": oauth.CLIENT_ID,
        "client_secret": oauth.CLIENT_SECRET,
        "refresh_token": token_data["refresh_token"],
        "token_uri": "https://oauth2.googleapis.com/token",
        "scopes": list(oauth.SCOPES),
    }

    with open(creds_path, "w") as f:
        json.dump(creds, f, indent=2)

    print(f"\nCredentials saved to: {creds_path}")
    return creds_path


def verify_credentials():
    """Test that ee.Initialize() works with saved credentials."""
    import ee
    try:
        ee.Initialize(project=None)  # No project needed for validation
        result = ee.Number(42).getInfo()
        if result == 42:
            print("GEE connection verified successfully!")
            return True
    except Exception as e:
        print(f"Verification failed: {e}")
        return False


def main():
    print("=" * 60)
    print(" Google Earth Engine — One-Time Authentication")
    print("=" * 60)

    # Step 1: Build URL
    url, code_verifier = build_auth_url()

    print("\nStep 1: Open this URL in your browser:\n")
    print(f"  {url}\n")
    print("Step 2: Log in with your personal Google account.")
    print("Step 3: Authorize the requested permissions.")
    print("Step 4: Copy the verification code shown.\n")

    auth_code = input("Step 5: Paste the verification code here and press Enter:\n> ").strip()

    if not auth_code:
        print("No code entered. Exiting.")
        sys.exit(1)

    print("\nExchanging code for refresh token...")
    try:
        token_data = exchange_code(auth_code, code_verifier)
    except Exception as e:
        print(f"ERROR: Token exchange failed: {e}")
        print("The code may have expired (they expire in ~10 minutes). Re-run the script.")
        sys.exit(1)

    if "refresh_token" not in token_data:
        print(f"ERROR: No refresh_token in response: {token_data}")
        sys.exit(1)

    creds_path = save_credentials(token_data)
    print("\nVerifying connection to Earth Engine...")
    verify_credentials()

    print("\n" + "=" * 60)
    print(" Authentication complete!")
    print(f" Credentials file: {creds_path}")
    print(" You can now run: python src/data_extraction/pipeline_runner.py ...")
    print("=" * 60)


if __name__ == "__main__":
    main()
