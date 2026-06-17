# Run ATOM — GPU Pre-flight Check

**Rule: Before starting any ATOM server, ALWAYS check GPU usage first. Never start a server on GPUs already in use by someone else.**

## Decision Flow

```
Check GPU usage
  ├─ GPUs free  → start the ATOM server
  └─ GPUs busy  → identify the owning container, report to user, DO NOT run, wait until free
```

## Step 1 — Check GPU usage

```bash
rocm-smi --showmemuse        # one-shot: VRAM% per GPU (0% = free)
# rocm-smi                   # full status (temp / power / util)
# watch -n 2 rocm-smi        # live view (interactive only, not for automation)
```

**Interpretation:**
- All GPUs at ~0% VRAM use → **free**, proceed to start the server.
- Any target GPU with VRAM in use → **busy**, go to Step 2.

## Step 2 — If GPUs are busy: find the owning container

```bash
# a. Get the PID(s) using the GPU
rocm-smi --showpids

# b. Map each PID to its docker container
docker ps -q | xargs -I {} sh -c \
  'if docker top {} 2>/dev/null | grep -q "<PID>"; then echo "Container ID: {}"; fi'

# c. Resolve the container name
docker inspect --format '{{.Name}}' <CONTAINER_ID>
```

## Step 3 — Report and wait

- Tell the user **which container** is occupying the GPU(s).
- **Do NOT start the server.** Wait until the other user finishes and the GPUs are free.
