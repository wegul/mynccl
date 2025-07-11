#!/bin/bash

# Test script for the control server
# This script demonstrates the reciprocal exchange functionality

echo "Building the control server and test client..."
make ctrl_server test_client

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

echo "Starting control server for 4 clients..."
./ctrl_server 4 &
SERVER_PID=$!

# Give the server time to start
sleep 1

echo "Starting test clients..."

# Start clients in background
./test_client 0 "Message from client 0" &
CLIENT0_PID=$!

./test_client 1 "Hello from client 1" &
CLIENT1_PID=$!

./test_client 2 "Data from client 2" &
CLIENT2_PID=$!

./test_client 3 "Buffer from client 3" &
CLIENT3_PID=$!

# Wait for all clients to complete
wait $CLIENT0_PID
wait $CLIENT1_PID
wait $CLIENT2_PID
wait $CLIENT3_PID

echo "All clients completed."

# Give server time to finish
sleep 2

# Kill server if still running
if kill -0 $SERVER_PID 2>/dev/null; then
    echo "Stopping server..."
    kill $SERVER_PID
fi

echo "Test completed!"
echo ""
echo "Expected exchanges:"
echo "  Client 0 should receive buffer from Client 3"
echo "  Client 1 should receive buffer from Client 2"
echo "  Client 2 should receive buffer from Client 1"
echo "  Client 3 should receive buffer from Client 0"
