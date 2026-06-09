#!/usr/bin/env python3
"""Drive the augustalabs.ai SSH TUI to (a) recon the menu, (b) submit a flag.

Usage:
    python3 arcus_drive.py recon                       # dump initial screen
    python3 arcus_drive.py submit "candidate string"   # navigate + submit
"""
import os, sys, pty, select, time, re, fcntl, termios, struct

HOST = 'augustalabs.ai'
SSH_ARGS = ['ssh', '-tt', '-o', 'StrictHostKeyChecking=accept-new',
            '-o', 'ConnectTimeout=15', '-o', 'BatchMode=yes', HOST]

def set_winsize(fd, rows=40, cols=120):
    try:
        fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack('HHHH', rows, cols, 0, 0))
    except OSError:
        pass

ANSI_RE = re.compile(rb'\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\x07\x1b]*[\x07\x1b\\]|\x1b[()][\w]')

def strip_ansi(buf: bytes) -> bytes:
    return ANSI_RE.sub(b'', buf)

def spawn_ssh():
    pid, fd = pty.fork()
    if pid == 0:
        os.environ['TERM'] = 'xterm-256color'
        os.environ['LANG'] = 'en_US.UTF-8'
        os.execvp(SSH_ARGS[0], SSH_ARGS)
    set_winsize(fd, 40, 120)
    return pid, fd

def read_for(fd, seconds, raw_log=None):
    """Read everything we can for `seconds`, return bytes."""
    deadline = time.time() + seconds
    out = bytearray()
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        try:
            r, _, _ = select.select([fd], [], [], min(0.2, remaining))
        except OSError:
            break
        if fd in r:
            try:
                chunk = os.read(fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            out.extend(chunk)
            if raw_log is not None:
                raw_log.extend(chunk)
    return bytes(out)

def read_until(fd, pattern: re.Pattern, timeout, raw_log=None):
    """Read until pattern is found in the *cleaned* stream or timeout."""
    deadline = time.time() + timeout
    out = bytearray()
    while time.time() < deadline:
        r, _, _ = select.select([fd], [], [], 0.2)
        if fd in r:
            try:
                chunk = os.read(fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            out.extend(chunk)
            if raw_log is not None:
                raw_log.extend(chunk)
            if pattern.search(strip_ansi(bytes(out))):
                return bytes(out), True
    return bytes(out), False

def write(fd, data: bytes):
    os.write(fd, data)
    time.sleep(0.15)

def recon(steps_down=0, dwell=4.0):
    """Connect, optionally press Down N times, dump initial screen."""
    pid, fd = spawn_ssh()
    try:
        initial = read_for(fd, dwell)
        print('--- initial ---')
        sys.stdout.write(strip_ansi(initial).decode('utf-8', errors='replace'))
        print('\n--- raw bytes (last 400) ---')
        sys.stdout.write(repr(initial[-400:]))
        for _ in range(steps_down):
            write(fd, b'\x1b[B')  # down arrow
        if steps_down:
            time.sleep(0.5)
            more = read_for(fd, 2.0)
            print(f'\n--- after {steps_down} downs ---')
            sys.stdout.write(strip_ansi(more).decode('utf-8', errors='replace'))
    finally:
        try: os.write(fd, b'\x03')  # ctrl-c
        except OSError: pass
        try: os.close(fd)
        except OSError: pass
        try: os.waitpid(pid, os.WNOHANG)
        except OSError: pass

FLAG_PROMPT_RE = re.compile(rb'(?i)flag\s*:[^\n]*$')
SUCCESS_RE = re.compile(b'(?i)correct|success|parab|right|blood|\xe2\x82\xac|ganh|vencedor|desbloque|congrat')
FAIL_RE    = re.compile(rb'(?i)wrong|incorrect|errado|incorreto|tente|try again')

def submit(candidate: str, menu_down=1, debug_path=None):
    """Submit one candidate. Returns dict with outcome."""
    pid, fd = spawn_ssh()
    log = bytearray()
    result = {'candidate': candidate, 'outcome': 'unknown', 'log_tail': ''}
    try:
        # let TUI fully render
        _ = read_for(fd, 4.0, raw_log=log)
        # navigate down (if needed) and Enter to select Ode Triunfal
        for _ in range(menu_down):
            write(fd, b'\x1b[B')
        if menu_down:
            _ = read_for(fd, 0.5, raw_log=log)
        # enter Ode Triunfal screen
        write(fd, b'\r')
        # wait for flag: prompt (let the screen settle)
        _ = read_for(fd, 2.0, raw_log=log)
        # send the candidate
        payload = candidate.encode('utf-8') + b'\r'
        write(fd, payload)
        # collect response — wait long enough for "checking..." -> answer
        _ = read_for(fd, 8.0, raw_log=log)
        cleaned = strip_ansi(bytes(log)).decode('utf-8', errors='replace')
        if SUCCESS_RE.search(bytes(log)):
            result['outcome'] = 'SUCCESS'
        elif FAIL_RE.search(bytes(log)):
            result['outcome'] = 'reject'
        else:
            result['outcome'] = 'ambiguous'
        result['log_tail'] = cleaned[-1500:]
    finally:
        try: os.write(fd, b'\x03')
        except OSError: pass
        try: os.close(fd)
        except OSError: pass
        try: os.waitpid(pid, os.WNOHANG)
        except OSError: pass
        if debug_path:
            with open(debug_path, 'wb') as f:
                f.write(bytes(log))
    return result

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage: arcus_drive.py recon [steps_down]')
        print('       arcus_drive.py submit "candidate"  [menu_down]')
        sys.exit(2)
    cmd = sys.argv[1]
    if cmd == 'recon':
        sd = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        recon(steps_down=sd, dwell=5.0)
    elif cmd == 'submit':
        cand = sys.argv[2]
        md = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        debug_path = f'/tmp/arcus_submit_{int(time.time())}.log'
        res = submit(cand, menu_down=md, debug_path=debug_path)
        print(f'\noutcome: {res["outcome"]}')
        print(f'debug log: {debug_path}')
        print('--- log tail ---')
        print(res['log_tail'])
    else:
        print(f'unknown command: {cmd}')
        sys.exit(2)
