all:
	# Generation of the protobuf-generated Python scripts for gRPC.
	python -m grpc_tools.protoc \
		--proto_path=./tf-demonstration/protofiles/. \
		--python_out=./tf-demonstration/prod-client-gui-mock/. \
		--grpc_python_out=./tf-demonstration/prod-client-gui-mock/. \
		./tf-demonstration/protofiles/grpccomm.proto;
