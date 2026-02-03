# Copyright (c) Daniel Fusaro. All rights reserved.

# -----------------------------------------------------------------------------
# Upload a file to a MicroPython device (e.g., LEGO SPIKE Prime) over serial.
# -----------------------------------------------------------------------------

import time, os
import glob


def upload2hub(local_path, device="/dev/ttyACM0"):
    remote_path = "/flash/" + local_path

    print(f"Uploading file to device: from: {local_path} to:   {remote_path}")

    with open(local_path, "rb") as f:
        file_data = f.read()

    # --- Connect to hub ---
    with open(device, "rb+", buffering=0) as ser:
        # Enter raw REPL
        ser.write(b'\r\x03\x03')  # Ctrl-C x2 to interrupt
        time.sleep(0.2)
        ser.write(b'\r\x01')      # Ctrl-A to enter raw REPL
        time.sleep(0.2)
        ser.flush()

        # Prepare file creation command
        ser.write(f"f=open('{remote_path}','wb')\n".encode("utf-8"))

        # Write file contents in small chunks (to avoid buffer overflow)
        chunk_size = 512
        for i in range(0, len(file_data), chunk_size):
            chunk = file_data[i:i+chunk_size]
            # Send base64-like encoded chunk using repr() so binary data is safe
            cmd = f"f.write({repr(chunk)})\n".encode("utf-8")
            ser.write(cmd)
            time.sleep(0.02)  # small delay for safety

        ser.write(b"f.close()\n")
        ser.write(b"\x04")  # Ctrl-D to execute
        ser.flush()

        # Wait for REPL response
        data = b""
        while True:
            bts = ser.read(1)
            if not bts:
                break
            if bts == b'\x04':  # End of transmission
                break
            data += bts

        # Exit raw REPL
        ser.write(b'\r\x02')  # Ctrl-B: back to normal REPL

    print(f"Uploaded {len(file_data)} bytes to {remote_path}")

def mkdir(path, device="/dev/ttyACM0"):
    print(f"Uploading file to device: creating folder   {path}")
    # --- Connect to hub ---
    with open(device, "rb+", buffering=0) as ser:
        # Enter raw REPL
        ser.write(b'\r\x03\x03')  # Ctrl-C x2 to interrupt
        time.sleep(0.2)
        ser.write(b'\r\x01')      # Ctrl-A to enter raw REPL
        time.sleep(0.2)
        ser.flush()

        cmd = f"import os\ntry:\n os.mkdir('{path}')\nexcept OSError:\n pass\n\x04"
        ser.write(cmd.encode('utf-8'))
        time.sleep(0.2)

        ser.write(b"\x04")  # Ctrl-D to execute
        ser.flush()

        # Exit raw REPL
        ser.write(b'\r\x02')  # Ctrl-B: back to normal REPL

    print(f"Created folder {path}")

    
upload2hub("dataset.py")
upload2hub("linear.py")
upload2hub("model.py")

mkdir("utils")

for file in glob.glob("utils/*.py"):
    upload2hub(file)