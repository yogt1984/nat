# Packaging `nat` as a password-gated `.deb`

Ship the `nat` terminal as a Debian package installable with
`sudo apt install nat` — but only on machines holding credentials to a private,
GPG-signed apt repository served behind HTTP basic-auth.

## 1. Build the `.deb`

```bash
bash packaging/build_deb.sh          # builds release Rust binaries first
# or, if rust/target/release is already built:
bash packaging/build_deb.sh --no-build
```

Output: `dist/nat_<version>_amd64.deb`. Inspect it:

```bash
dpkg-deb --info dist/nat_*.deb
dpkg-deb -c    dist/nat_*.deb        # file listing
```

Installed layout:

| Path | Contents |
|------|----------|
| `/usr/lib/nat/nat` | the CLI script (real file) |
| `/usr/lib/nat/scripts/` | Python analysis stack |
| `/usr/lib/nat/rust/target/release/` | `ing`, `api`, `natviz3d`, validators |
| `/etc/nat/*.toml` | configuration (symlinked back as `/usr/lib/nat/config`) |
| `/usr/bin/nat` | symlink → `/usr/lib/nat/nat` (created in `postinst`) |
| `/var/lib/nat/` | writable runtime data (symlinked as `/usr/lib/nat/data`) |

## 2. Sign + publish to a private apt repo

One-time: create a dedicated signing key.

```bash
gpg --quick-generate-key "NAT Repo <nat@diamondapps.ch>" rsa4096 sign never
gpg --armor --export "NAT Repo" > nat-archive-keyring.asc   # share with clients
```

Publish with **aptly** (recommended) — it produces a signed flat repo:

```bash
aptly repo create -distribution=stable -component=main nat
aptly repo add nat dist/nat_*.deb
aptly publish repo -gpg-key="NAT Repo" nat
# aptly's published tree lands in ~/.aptly/public/{dists,pool}
```

Serve `~/.aptly/public` over **HTTPS with basic-auth**. With Caddy (already in the
stack), a Caddyfile:

```
repo.example.com {
    root * /home/nat/.aptly/public
    file_server browse
    basicauth { alice JDJhJDE0... }   # htpasswd-style bcrypt hash
}
```

(`caddy hash-password` generates the hash. nginx equivalent: `auth_basic` +
`htpasswd`.)

## 3. Onboard a client (the "password")

```bash
sudo REPO_URL=https://repo.example.com \
     REPO_USER=alice REPO_PASS=secret \
     PUBKEY_URL=https://repo.example.com/nat-archive-keyring.asc \
     bash scripts/setup_apt_client.sh

sudo apt install nat
```

`setup_apt_client.sh` writes:
- `/etc/apt/keyrings/nat-archive-keyring.gpg` — trusted signing key
- `/etc/apt/sources.list.d/nat.list` — `deb [signed-by=…] …`
- `/etc/apt/auth.conf.d/nat.conf` (mode 600) — `machine/login/password`

Without those credentials, `apt update` returns **401 Unauthorized**; repository
metadata is additionally GPG-verified, so an attacker can neither install nor
tamper with the package.

## 4. From the CLI

`nat package deb` wraps `build_deb.sh` for convenience.
