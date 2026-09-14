#!/usr/bin/env python3
"""Scarica gli ultimi n commit hash di una repo GitHub e li salva in un file txt."""

import argparse
import os
import re
import sys
import urllib.request
import urllib.error
import json


def parse_owner_repo(url: str) -> tuple[str, str]:
    match = re.search(r"github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$", url.strip())
    if not match:
        raise ValueError(f"URL non valido: {url}")
    return match.group(1), match.group(2)


def get_commit_hashes(owner: str, repo: str, n: int) -> list[str]:
    hashes = []
    page = 1
    per_page = 100
    while len(hashes) < n:
        url = (
            f"https://api.github.com/repos/{owner}/{repo}/commits"
            f"?per_page={per_page}&page={page}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "get-last-commits-script"})
        try:
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read())
        except urllib.error.HTTPError as e:
            sys.exit(f"Errore nella richiesta a GitHub: {e.code} {e.reason}")

        if not data:
            break

        hashes.extend(commit["sha"] for commit in data)
        page += 1

    if len(hashes) < n:
        print(f"Attenzione: trovati solo {len(hashes)} commit (richiesti {n}).")

    return hashes[:n]


def resolve_output_path(path: str) -> str:
    if os.path.dirname(path) == "":
        # solo nome file: deve essere .txt e va nella dir corrente
        if not path.endswith(".txt"):
            raise ValueError("Il nome file deve avere estensione .txt")
        return path
    return path


def main():
    parser = argparse.ArgumentParser(description="Salva gli ultimi n commit hash di una repo GitHub in un file txt.")
    parser.add_argument("repo_url", help="Link della repo GitHub, es. https://github.com/ReactiveX/RxJava")
    parser.add_argument("n", type=int, help="Numero di commit da recuperare")
    parser.add_argument("path", help="Path completo o solo nome file (.txt) dove salvare il risultato")
    args = parser.parse_args()

    owner, repo = parse_owner_repo(args.repo_url)
    output_path = resolve_output_path(args.path)

    hashes = get_commit_hashes(owner, repo, args.n)

    with open(output_path, "w") as f:
        f.write("\n".join(hashes) + "\n")

    print(f"Salvati {len(hashes)} commit hash in {output_path}")


if __name__ == "__main__":
    main()
