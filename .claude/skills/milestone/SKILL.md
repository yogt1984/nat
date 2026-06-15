# /milestone — Context-first implementation workflow

Before writing any code, establish full project context. This prevents wrong patches, import errors, and wasted iterations.

## Step 1: Read project state

Read the following in parallel:
- `CLAUDE.md` (project conventions)
- `FEATURES.md` (feature manifest)
- `Makefile` (available targets)
- Any files the user references as the scope of work

## Step 2: Run current tests

Run the relevant test suite to establish a green baseline:
- Rust: `cd rust && cargo test --package ing 2>&1 | tail -20`
- Python: `pytest scripts/tests/ 2>&1 | tail -20`
- Or the specific test target if the user names one

Report: total tests, pass/fail count, any existing failures.

## Step 3: Plan before implementing

List the tasks to complete, with:
- Dependencies between tasks (what must come first)
- Which files each task touches
- Which existing tests cover the area, and what new tests are needed

Present this plan to the user and wait for approval before writing code.

## Step 4: Implement with test-after-each-task

For each task:
1. Implement the change
2. Write or update tests
3. Run the relevant test suite immediately
4. Fix any failures before moving to the next task
5. Commit with a conventional commit message (`feat:`, `fix:`, `refactor:`)

Never batch multiple tasks before running tests. Never defer testing to the end.

## Step 5: Final verification

After all tasks are complete:
1. Run the full relevant test suite one final time
2. Report final pass/fail counts vs the baseline from Step 2
3. List all commits made in this session
