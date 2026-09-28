"""Create a user (bootstrap the first admin with this). Usage: python tools/create_user.py alice approver"""
import argparse
import getpass
import os
import sys

import psycopg
from psycopg.rows import dict_row

from meghprahari.governance import PERMISSIONS, hash_password
from meghprahari.store import audit_append


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("username")
    ap.add_argument("role", choices=sorted(PERMISSIONS))
    a = ap.parse_args()
    pw = getpass.getpass("Password (min 12 chars): ")
    if len(pw) < 12 or pw != getpass.getpass("Repeat: "):
        sys.exit("passwords differ or are shorter than 12 characters")
    with psycopg.connect(os.environ["MP_DATABASE_URL"], row_factory=dict_row) as c:
        c.execute("INSERT INTO app_user (username, pw_hash, role) VALUES (%s,%s,%s::user_role)", (a.username, hash_password(pw), a.role))
        audit_append(c, "cli", "user_create", {"username": a.username, "role": a.role})
    print("created", a.username, a.role)


if __name__ == "__main__":
    main()
