# Generated by the Protocol Buffers compiler. DO NOT EDIT!
# source: grpc/generation.proto
# plugin: grpclib.plugin.main
import abc
import typing

import grpclib.const
import grpclib.client
if typing.TYPE_CHECKING:
    import grpclib.server

import stable_diffusion_krita.generation_pb2


class GenerationServiceBase(abc.ABC):

    @abc.abstractmethod
    async def Generate(self, stream: 'grpclib.server.Stream[grpc.generation_pb2.Request, grpc.generation_pb2.Answer]') -> None:
        pass

    @abc.abstractmethod
    async def ChainGenerate(self, stream: 'grpclib.server.Stream[grpc.generation_pb2.ChainRequest, grpc.generation_pb2.Answer]') -> None:
        pass

    def __mapping__(self) -> typing.Dict[str, grpclib.const.Handler]:
        return {
            '/gooseai.GenerationService/Generate': grpclib.const.Handler(
                self.Generate,
                grpclib.const.Cardinality.UNARY_STREAM,
                stable_diffusion_krita.generation_pb2.Request,
                stable_diffusion_krita.generation_pb2.Answer,
            ),
            '/gooseai.GenerationService/ChainGenerate': grpclib.const.Handler(
                self.ChainGenerate,
                grpclib.const.Cardinality.UNARY_STREAM,
                stable_diffusion_krita.generation_pb2.ChainRequest,
                stable_diffusion_krita.generation_pb2.Answer,
            ),
        }


class GenerationServiceStub:

    def __init__(self, channel: grpclib.client.Channel) -> None:
        self.Generate = grpclib.client.UnaryStreamMethod(
            channel,
            '/gooseai.GenerationService/Generate',
            stable_diffusion_krita.generation_pb2.Request,
            stable_diffusion_krita.generation_pb2.Answer,
        )
        self.ChainGenerate = grpclib.client.UnaryStreamMethod(
            channel,
            '/gooseai.GenerationService/ChainGenerate',
            stable_diffusion_krita.generation_pb2.ChainRequest,
            stable_diffusion_krita.generation_pb2.Answer,
        )