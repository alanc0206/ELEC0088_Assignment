# Import socket module
import socket
from uuid import uuid1


def Main():
    # local host IP ’127.0.0.1’
    host = '127.0.0.1'
    port = 1337
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    sess_id = uuid1()
    s.send(str(sess_id).encode('ascii'))
    data = s.recv(1024)
    print('Received from Oracle :', str(data.decode()))
    while True:

        query = input('\nInput to Oracle: ')

        s.send(query.encode('ascii'))

        data = s.recv(1024)

        print('\nReceived from Oracle :', str(data.decode()))

    # close the connection
    s.close()


if __name__ == '__main__':
    Main()