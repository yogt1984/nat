# /ship — Integrate one unit (inner-loop step 7, with git guardrails)

Definition-of-done for an increment: registered, contract-bound, dual-tested, **maturity-tagged**,
committed on a feat branch, CI green, merged `--no-ff`. This skill enforces that sequence and the
project's git guardrails. Run it only after the planted test and `/smoke` are green.

## Step 1: Re-check git state FIRST

Concurrent activity happens in this checkout — never assume the state from earlier in the session:
```bash
git status --short && git branch --show-current && git log --oneline -3
```
Note any files you did **not** create (the user may have uncommitted work) — you will stage only
your own files, never `git add -A` blindly.

## Step 2: Be on a feat branch

If on `master`, create one: `git checkout -b feat/<slug>`. Never commit a unit directly to master.

## Step 3: Confirm the maturity tag shipped in the SAME change

The methodology forbids a unit landing without its `SPEC→PRELIM→BETA→PROVEN→LIVE` tag (NAT9), set in
the same diff as the code + `nat` command. Verify the tag is present in the staged diff:
```bash
git diff --cached | grep -iE "maturity|PRELIM|BETA|PROVEN|\[SPEC\]" | head
```
A merged unit's honest default is **PRELIM** (planted + smoke pass). No tag → not shippable.

## Step 4: Run the fast CI-equivalent locally (green before commit)

Run the relevant subset; do not commit red:
```bash
# Rust unit (if rust/ touched):
cd rust && cargo fmt --check && cargo clippy -q 2>&1 | tail -5 && cargo test -q 2>&1 | tail -5
# Python (if scripts/ touched):
pytest scripts/tests/ -q 2>&1 | tail -5
```

## Step 5: Stage only your files + conventional commit

```bash
git add <your files explicitly>          # NOT -A; leave the user's concurrent work alone
git commit -m "feat(<scope>): <what> — <why>

<body: contract, tests, maturity tag>

Co-Authored-By: <the harness co-author trailer for the active model>"
```
Conventional type (`feat:`/`fix:`/`docs:`/`refactor:`), minimal diff, imperative subject.

## Step 6: Push the feat branch

```bash
git push -u origin feat/<slug>
```

## Step 7: Confirmation gate → merge --no-ff to master

The canonical flow merges `--no-ff` to master, but **pause and confirm with the user first** (they
often review on the branch). On confirmation:
```bash
git checkout master && git merge --no-ff feat/<slug> -m "Merge branch 'feat/<slug>'" && git push origin master
```
Then report: the merge commit, what advanced on origin, and the unit's maturity tag — so the next
step (the autonomous gate ladder, `/gate` → `/maturity`) is clear.
