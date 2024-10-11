#!/usr/bin/env python
import sys
import random

doc_string = """\
Usage:
    python3 get-random-ports.py <N>

    Where <N> is the number of unique random ports to generate.

    Example:
        python3 get-random-ports.py 5
        This will output 5 unique random ports from the valid range.

#---------------------------------------#
# Documentation:

    This script generates N unique random port numbers in the range 1025 to 65535.

    Internet sockets are described by both an address (IP address or hostname) and a port number.
    The IP address identifies the host on the network, while the port number identifies a specific
    process or service running on that host.

    Ports are represented by 16-bit integers between 1 and 65535. Lower-numbered ports (1-1024) are
    reserved for system services or well-known internet protocols such as HTTP (port 80), HTTPS (port 443),
    and SSH (port 22). For general-purpose simulations or non-privileged services, it is advisable to
    use ports in the range 1025-65535.

    This script ensures that the generated ports are within the range 1025-65535 to avoid conflicts
    with system-reserved ports.

#---------------------------------------#
# How to use the output in a bash script:

    You can capture the output of the Python script and store it in a variable in a Bash script as follows:

    ``` bash
    #!/bin/bash

    # Run the Python script and capture the output (the random ports)
    random_ports=$(python3 get-random-ports.py 10)

    # Print the captured ports
    echo "Random Ports: $random_ports"

    # Convert the string of ports into an array for individual access
    ports_array=($random_ports)

    # Access individual ports (for example, first and second port)
    echo "First port: ${ports_array[0]}"
    echo "Second port: ${ports_array[1]}"
    
    ```
"""

if len(sys.argv) != 2:
    # print("Usage: python3 get-random-ports.py <N>")
    print(doc_string)
    sys.exit(1)

try:
    N = int(sys.argv[1])
    if N <= 0:
        raise ValueError("N must be a positive integer.")
except ValueError as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)

# Ensure N does not exceed the number of available ports in the range 1025-65535
if N > (65535 - 1025 + 1):
    print("ERROR: N exceeds the number of available ports in the range.", file=sys.stderr)
    sys.exit(1)

# Generate N unique random ports in the range 1025-65535
ports = random.sample(range(1025, 65536), N)

# Output the ports as space-separated values
print(" ".join(map(str, ports)))



