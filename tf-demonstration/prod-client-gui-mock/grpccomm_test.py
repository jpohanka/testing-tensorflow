import unittest

import time
import grpc
import grpccomm_pb2 as pb
import grpccomm_pb2_grpc as pb_grpc
import concurrent.futures

GRPC_TEST_PORT = 50051
GRPC_TEST_REQUEST_MESSAGE = "success_request"
GRPC_TEST_RESPONSE_MESSAGE = "success_response"

SLEEP_TIME_SECONDS = 3

class TestGRPCServer(pb_grpc.TFCalcServicer):
    """
    A test class for the gRPC communication service.
    """
    def Calculate(self, request, context):
        return pb.CalcResponse(header=pb.CalcHeader(message=GRPC_TEST_RESPONSE_MESSAGE))    

        
def run_server_test(server):
    server.start()
    time.sleep(2 * SLEEP_TIME_SECONDS)
    server.stop(0)
        
        
def run_client_test():
    time.sleep(SLEEP_TIME_SECONDS)
    channel = grpc.insecure_channel('localhost:%d' % GRPC_TEST_PORT)
    stub = pb_grpc.TFCalcStub(channel)
    response = stub.Calculate(pb.CalcRequest(header=pb.CalcHeader(message=GRPC_TEST_REQUEST_MESSAGE)))
    try:
        if response.header.message == GRPC_TEST_RESPONSE_MESSAGE:
            return True
        else:
            return False
    except:
        return False


class TestGRPCComm(unittest.TestCase):

    def test_client_server_comm(self):
        """
        Tests a simple gRPC server-client connection. Sends dummy message with
        non-empty header.
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            server = grpc.server(executor)
            pb_grpc.add_TFCalcServicer_to_server(TestGRPCServer(), server)
            server.add_insecure_port('[::]:%d' % GRPC_TEST_PORT)
            future_list = {
                executor.submit(run_server_test, server) : "server",
                executor.submit(run_client_test) : "client"
            }
            for future in concurrent.futures.as_completed(future_list):
                if future_list[future] == "client":
                   self.assertTrue(future.result())