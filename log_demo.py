import grpc
from proto.message_log_pb2 import LogRequest
from proto.message_log_pb2 import NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL
from proto.message_log_pb2_grpc import MessageLoggerStub

LOG_END_POINT = 'localhost:36000'

def request_log(request):
    with grpc.insecure_channel(LOG_END_POINT) as channel:
        stub = MessageLoggerStub(channel)
        response = stub.LogMessage(request)

        return response

if __name__ == '__main__':
    prompt = 'Hello'
    generated = 'Hello Bookathon~!'
    msg = 'Prompt: {}\nGenearted: {}'.format(prompt, generated)
    log_request = LogRequest(log_level=DEBUG, msg=msg)

    log_response = request_log(log_request)
    assert log_response.done is True
