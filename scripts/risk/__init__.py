"""Risk-control subsystem (T16).

The kill-switch daemon (`risk.kill_switch`) polls realised PnL / IC and halts
trading when a ROADMAP Step 9 threshold is breached. Halt state is published to
`data/risk/halt_state.json` — the IPC contract the signal bridge (T17) checks
before every cycle.
"""
