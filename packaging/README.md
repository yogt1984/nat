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
| `/etc/nat/*.toml` | configuration (resolved directly by `nat_paths`) |
| `/usr/bin/nat` | symlink → `/usr/lib/nat/nat` (created in `postinst`) |

### Paths & relocation (`nat_paths`)

`nat` resolves data/config/logs/reports at runtime via `scripts/nat_paths.py`
(precedence: `NAT_HOME`/`NAT_*` env → source-checkout → installed XDG). Run
`nat config paths` to see the resolved locations and which rule was applied.

- **Installed** (no `.git` / `rust/Cargo.toml` in the tree): config from `/etc/nat`,
  data under `$NAT_HOME` or `~/.local/share/nat`, the ingestor honors
  `NAT_DATA_DIR`/`NAT_TRADE_DIR` (the `nat` CLI sets these automatically).
- **System-wide data:** set `NAT_HOME=/var/lib/nat` (e.g. in the service unit or
  `/etc/environment`) to keep data out of a user's home.
- **Dev checkout:** everything stays under the repo, byte-identical to before.

### Reboot-proof self-healing (`nat service`, systemd `--user`)

By default the ingestor + gap-alert daemon run in tmux kept alive by 5-min cron
watchdogs — which **die on reboot/terminal-close**. For durable supervision,
install systemd `--user` units (auto-restart on crash *and* start on boot):

```bash
nat service install      # writes ~/.config/systemd/user/nat-{ingestor,gap-alert}.service,
                         # enables linger, hands the running ingestor over to systemd
nat service status       # active/enabled + linger state
nat service uninstall    # remove units, restore the tmux+cron path
```

Once installed, `nat start/stop/status` and `nat log` transparently use systemd
(`nat log` follows `journalctl --user -u nat-ingestor`). `install` does a brief
(~seconds) ingestor restart, marked paused so the gap daemon doesn't false-page.
The gap-alert unit is **independent** of the ingestor (keeps monitoring while the
ingestor is down). Note: the gap unit pins the Python interpreter `nat` runs under
(it must have the analysis deps, e.g. pyarrow). Reboot survival needs
`loginctl enable-linger` (done by `install`).

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
