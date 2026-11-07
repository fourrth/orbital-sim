# orbit-sim

Simple N-body gravity simulation written in python

## Structure

| Module | Description |
|---|---|
| `orbitsim` | Main library |
| `test` | Simple test script |
| `display` | Matplotlib animation of the inner solar system. |

This project uses `uv` and `numpy`, but you could use python to run the scripts independently.
(SI units throughout — meters, kilograms)


## Test Script

```bash
uv run test
```

Verifies circular orbit stability, along with energy and momentum conservation.

## Display Script

```
uv run display
```
The inner solar system.

## Notes
- Set the softening term to prevent the acceleration from --> inf if you have a close encounter.