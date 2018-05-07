# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: grpccomm.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='grpccomm.proto',
  package='grpccomm',
  syntax='proto3',
  serialized_pb=_b('\n\x0egrpccomm.proto\x12\x08grpccomm\"\x1d\n\nCalcHeader\x12\x0f\n\x07message\x18\x01 \x01(\t\"3\n\x0b\x43\x61lcRequest\x12$\n\x06header\x18\x01 \x01(\x0b\x32\x14.grpccomm.CalcHeader\"4\n\x0c\x43\x61lcResponse\x12$\n\x06header\x18\x01 \x01(\x0b\x32\x14.grpccomm.CalcHeader2F\n\x06TFCalc\x12<\n\tCalculate\x12\x15.grpccomm.CalcRequest\x1a\x16.grpccomm.CalcResponse\"\x00\x42\x30\n\x18io.grpccomm.tfcalculatorB\x0cTFCalculatorP\x01\xa2\x02\x03TFCb\x06proto3')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_CALCHEADER = _descriptor.Descriptor(
  name='CalcHeader',
  full_name='grpccomm.CalcHeader',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='message', full_name='grpccomm.CalcHeader.message', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=28,
  serialized_end=57,
)


_CALCREQUEST = _descriptor.Descriptor(
  name='CalcRequest',
  full_name='grpccomm.CalcRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='header', full_name='grpccomm.CalcRequest.header', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=59,
  serialized_end=110,
)


_CALCRESPONSE = _descriptor.Descriptor(
  name='CalcResponse',
  full_name='grpccomm.CalcResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='header', full_name='grpccomm.CalcResponse.header', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=112,
  serialized_end=164,
)

_CALCREQUEST.fields_by_name['header'].message_type = _CALCHEADER
_CALCRESPONSE.fields_by_name['header'].message_type = _CALCHEADER
DESCRIPTOR.message_types_by_name['CalcHeader'] = _CALCHEADER
DESCRIPTOR.message_types_by_name['CalcRequest'] = _CALCREQUEST
DESCRIPTOR.message_types_by_name['CalcResponse'] = _CALCRESPONSE

CalcHeader = _reflection.GeneratedProtocolMessageType('CalcHeader', (_message.Message,), dict(
  DESCRIPTOR = _CALCHEADER,
  __module__ = 'grpccomm_pb2'
  # @@protoc_insertion_point(class_scope:grpccomm.CalcHeader)
  ))
_sym_db.RegisterMessage(CalcHeader)

CalcRequest = _reflection.GeneratedProtocolMessageType('CalcRequest', (_message.Message,), dict(
  DESCRIPTOR = _CALCREQUEST,
  __module__ = 'grpccomm_pb2'
  # @@protoc_insertion_point(class_scope:grpccomm.CalcRequest)
  ))
_sym_db.RegisterMessage(CalcRequest)

CalcResponse = _reflection.GeneratedProtocolMessageType('CalcResponse', (_message.Message,), dict(
  DESCRIPTOR = _CALCRESPONSE,
  __module__ = 'grpccomm_pb2'
  # @@protoc_insertion_point(class_scope:grpccomm.CalcResponse)
  ))
_sym_db.RegisterMessage(CalcResponse)


DESCRIPTOR.has_options = True
DESCRIPTOR._options = _descriptor._ParseOptions(descriptor_pb2.FileOptions(), _b('\n\030io.grpccomm.tfcalculatorB\014TFCalculatorP\001\242\002\003TFC'))
try:
  # THESE ELEMENTS WILL BE DEPRECATED.
  # Please use the generated *_pb2_grpc.py files instead.
  import grpc
  from grpc.framework.common import cardinality
  from grpc.framework.interfaces.face import utilities as face_utilities
  from grpc.beta import implementations as beta_implementations
  from grpc.beta import interfaces as beta_interfaces


  class TFCalcStub(object):

    def __init__(self, channel):
      """Constructor.

      Args:
        channel: A grpc.Channel.
      """
      self.Calculate = channel.unary_unary(
          '/grpccomm.TFCalc/Calculate',
          request_serializer=CalcRequest.SerializeToString,
          response_deserializer=CalcResponse.FromString,
          )


  class TFCalcServicer(object):

    def Calculate(self, request, context):
      """Define the gRPC calculation service.
      """
      context.set_code(grpc.StatusCode.UNIMPLEMENTED)
      context.set_details('Method not implemented!')
      raise NotImplementedError('Method not implemented!')


  def add_TFCalcServicer_to_server(servicer, server):
    rpc_method_handlers = {
        'Calculate': grpc.unary_unary_rpc_method_handler(
            servicer.Calculate,
            request_deserializer=CalcRequest.FromString,
            response_serializer=CalcResponse.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        'grpccomm.TFCalc', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


  class BetaTFCalcServicer(object):
    """The Beta API is deprecated for 0.15.0 and later.

    It is recommended to use the GA API (classes and functions in this
    file not marked beta) for all further purposes. This class was generated
    only to ease transition from grpcio<0.15.0 to grpcio>=0.15.0."""
    def Calculate(self, request, context):
      """Define the gRPC calculation service.
      """
      context.code(beta_interfaces.StatusCode.UNIMPLEMENTED)


  class BetaTFCalcStub(object):
    """The Beta API is deprecated for 0.15.0 and later.

    It is recommended to use the GA API (classes and functions in this
    file not marked beta) for all further purposes. This class was generated
    only to ease transition from grpcio<0.15.0 to grpcio>=0.15.0."""
    def Calculate(self, request, timeout, metadata=None, with_call=False, protocol_options=None):
      """Define the gRPC calculation service.
      """
      raise NotImplementedError()
    Calculate.future = None


  def beta_create_TFCalc_server(servicer, pool=None, pool_size=None, default_timeout=None, maximum_timeout=None):
    """The Beta API is deprecated for 0.15.0 and later.

    It is recommended to use the GA API (classes and functions in this
    file not marked beta) for all further purposes. This function was
    generated only to ease transition from grpcio<0.15.0 to grpcio>=0.15.0"""
    request_deserializers = {
      ('grpccomm.TFCalc', 'Calculate'): CalcRequest.FromString,
    }
    response_serializers = {
      ('grpccomm.TFCalc', 'Calculate'): CalcResponse.SerializeToString,
    }
    method_implementations = {
      ('grpccomm.TFCalc', 'Calculate'): face_utilities.unary_unary_inline(servicer.Calculate),
    }
    server_options = beta_implementations.server_options(request_deserializers=request_deserializers, response_serializers=response_serializers, thread_pool=pool, thread_pool_size=pool_size, default_timeout=default_timeout, maximum_timeout=maximum_timeout)
    return beta_implementations.server(method_implementations, options=server_options)


  def beta_create_TFCalc_stub(channel, host=None, metadata_transformer=None, pool=None, pool_size=None):
    """The Beta API is deprecated for 0.15.0 and later.

    It is recommended to use the GA API (classes and functions in this
    file not marked beta) for all further purposes. This function was
    generated only to ease transition from grpcio<0.15.0 to grpcio>=0.15.0"""
    request_serializers = {
      ('grpccomm.TFCalc', 'Calculate'): CalcRequest.SerializeToString,
    }
    response_deserializers = {
      ('grpccomm.TFCalc', 'Calculate'): CalcResponse.FromString,
    }
    cardinalities = {
      'Calculate': cardinality.Cardinality.UNARY_UNARY,
    }
    stub_options = beta_implementations.stub_options(host=host, metadata_transformer=metadata_transformer, request_serializers=request_serializers, response_deserializers=response_deserializers, thread_pool=pool, thread_pool_size=pool_size)
    return beta_implementations.dynamic_stub(channel, 'grpccomm.TFCalc', cardinalities, options=stub_options)
except ImportError:
  pass
# @@protoc_insertion_point(module_scope)