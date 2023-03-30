import socket
from _thread import *
import threading
import dialogflow_api

print_lock = threading.Lock()

# Thread function
def threaded(c):

    try:
        # Data received from client
        uuid = c.recv(1024)

        session_id = uuid.decode('ascii')

        # Create chatbot API object
        Chatbot = dialogflow_api.BotApi(session_id)

        greet = Chatbot.fulfill("Hi")

        # Send to client
        c.send(greet.encode('ascii'))

        while True:
            query = c.recv(1024)

            to_bot = query.decode('ascii')

            print(query.decode('ascii'))

            answer = Chatbot.fulfill(to_bot)

            c.send(answer.encode('ascii'))
    except socket.error as e:
        print("Socket error " + str(e))
        print_lock.release()
        c.close()
    # connection closed
    c.close()


def Main():
    host = ""
    port = 1337
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
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