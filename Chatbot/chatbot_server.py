import socket
from _thread import *
import threading
from uuid import uuid4
import dialogflow_api

print_lock = threading.Lock()

# thread function
def threaded(c):

    # data received from client
    uuid = c.recv(1024)

    session_id = uuid.decode('ascii')

    # reverse the given string from client
    prompt = dialogflow_api.initiate(session_id)

    # send to client
    c.send(prompt.encode('ascii'))

    query = c.recv(1024)

    print(query.decode('ascii'))

    answer = "---DIAGNOSIS HERE---"

    c.send(answer.encode('ascii'))

    # connection closed
    c.close()


def Main():
    host = ""
    # reverse a port on your computer
    # in our case it is 12345 but it
    # can be anything
    port = 1337
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((host, port))
    print("socket binded to post", port)
    # put the socket into listening mode
    s.listen(5)
    print("socket is listening")
    # a forever loop until client wants to exit
    while True:
        # establish connection with client
        c, addr = s.accept()
        # lock acquired by client
        print_lock.acquire()
        print('Connected to :', addr[0], ':', addr[1])
        # Start a new thread and return its identifier
        start_new_thread(threaded, (c,))

    s.close()


if __name__ == '__main__':
    Main()