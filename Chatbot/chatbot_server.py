import socket
from _thread import *
import threading
import dialogflow_api

print_lock = threading.Lock()

# Thread function
def threaded(c):

    try:
        # Get uuid from client
        uuid = c.recv(1024)

        # Use uuid as unique session id for API
        session_id = uuid.decode('ascii')

        # Create Oracle API object
        Oracle = dialogflow_api.BotApi(session_id)

        # Initiate a greeting
        greet = Oracle.fulfill("Hi")

        # Send greeting to client
        c.send(greet.encode('ascii'))

        while True:
            # Receive query from client
            query = c.recv(1024)

            to_bot = query.decode('ascii')

            print(query.decode('ascii'))

            # Get fulfillment from dialogflow agent
            answer = Oracle.fulfill(to_bot)

            c.send(answer.encode())
    except socket.error as e:
        # Catch socket exceptions
        print("Socket error " + str(e))
        print_lock.release()
        c.close()
    # connection closed
    c.close()


def Main():
    host = ""
    port = 1337
    # Create socket object
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Bind socket into localhost:port
    s.bind((host, port))
    print("Socket bound to post", port)
    # put the socket into listening mode
    s.listen(5)
    print("Socket is listening")
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
