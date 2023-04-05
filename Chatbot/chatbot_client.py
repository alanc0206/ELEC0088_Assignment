## Run this script to connect a client to the Oracle server. ##
# Import socket module
import socket
from uuid import uuid1


def Main():
    # local host IP ’127.0.0.1’
    host = '127.0.0.1'
    port = 1337
    # create socket object
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Connect to localhost:port
    s.connect((host, port))
    # Create unique session id for api
    sess_id = uuid1()
    # Send session ID
    s.send(str(sess_id).encode('ascii'))
    data = s.recv(1024)
    # Receive first response
    print('Received from Oracle :', str(data.decode()))
    # Loop forever
    while True:

        query = input('\nInput to Oracle: ')
        # Send input to server
        s.send(query.encode('ascii'))
        # Receive fulfillment from server
        data = s.recv(1024)
        # Detect empty transmission
        if not data:
            print("Server disconnected, closing connection.")
            # close connection
            s.close()
            quit()
        # Decode and print fulfillment
        print('\nReceived from Oracle :', str(data.decode()))

    # close the connection
    s.close()


if __name__ == '__main__':
    Main()