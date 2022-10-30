
from pathlib import Path
import sys,os
from typing import Dict,List
import random
from urllib.parse import urlparse
from PyQt5.QtGui  import QImage
import traceback
from krita import *

def append_sys_path(here=Path(__file__).parent):
    """
    add ./site-packages folder to sys.path, so that
    any package placed in there can be imported to pykrita.

    ie: ./site-packages/grpc/
    """
    site_packages = str(here.joinpath('site-packages'))
    if site_packages not in sys.path:
        sys.path.append(site_packages)


def register():
    # must be called before first import grpc
    append_sys_path()
register()
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from grpclib.client import Channel
import stable_diffusion_krita.generation_pb2 as generation  
import stable_diffusion_krita.generation_grpc as generation_grpc


import stable_diffusion_krita.engines_pb2 as engines  
import stable_diffusion_krita.engines_grpc as engines_grpc
diffusionMap: Dict[str, int] = {
    "DDIM": generation.SAMPLER_DDIM,
    "PLMS": generation.SAMPLER_DDPM,
    "Euler": generation.SAMPLER_K_EULER,
    "Euler a": generation.SAMPLER_K_EULER_ANCESTRAL,
    "Heun": generation.SAMPLER_K_HEUN,
    "DPM2": generation.SAMPLER_K_DPM_2,
    "DPM2 a": generation.SAMPLER_K_DPM_2_ANCESTRAL,
    "LMS": generation.SAMPLER_K_LMS,
}



class grpcClient:
    def __init__(self,SDConfig,endpoint):
        self.SDConfig = SDConfig
        self.endpoint=endpoint
    def errorMessage(self,text,detailed):
        msgBox= QMessageBox()
        msgBox.resize(500,200)
        msgBox.setWindowTitle("Stable Diffusion")
        msgBox.setText(text)
        msgBox.setDetailedText(detailed)
        msgBox.setStyleSheet("QLabel{min-width: 700px;}")
        msgBox.exec()     
    async def runSD(self,p):
        SDConfig=self.SDConfig
        prompts: List[generation.Prompt] = []
        prompt=generation.Prompt( text=p.prompt)
        prompts.append(prompt)
        if (p.negativePrompt):
            negativePromptParameters=generation.PromptParameters(weight=-1)
            negativePrompt = generation.Prompt( text=p.negativePrompt,
                parameters=negativePromptParameters
            )
            prompts.append(negativePrompt)
        if (p.imageBinary):
                artifact=generation.Artifact(
                    type=generation.ArtifactType.ARTIFACT_IMAGE,
                    binary=p.imageBinary
                )
                promptImage= generation.Prompt(
                    artifact=artifact,
                    parameters=generation.PromptParameters(init=True)
                )
                prompts.append(promptImage)
            
        if (p.mode=="inpainting" or p.mode=="inpainting_original"):
            if (p.maskImage64):                
                adjustments: List[generation.ImageAdjustment] = []

                adjustment=generation.ImageAdjustment(
                    levels=generation.ImageAdjustment_Levels(
                        input_low=0,
                        input_high=0.01,
                        output_low=0,
                        output_high=1
                    )
                )
                adjustments.append(adjustment)

                adjustment=generation.ImageAdjustment(
                    blur=generation.ImageAdjustment_Gaussian(
                        sigma=32,
                        direction=generation.GaussianDirection.DIRECTION_UP
                    )
                )
                adjustments.append(adjustment)
             #   adjustment=generation.ImageAdjustment(
            #            invert=generation.ImageAdjustment_Invert()
            #    )
             #   adjustments.append(adjustment)
                artifact=generation.Artifact(
                    type=generation.ArtifactType.ARTIFACT_MASK,
                    adjustments=adjustments,
                    binary=p.maskImageBinary
                )
                promptMask= generation.Prompt(artifact=artifact)
                prompts.append(promptMask)


        seed = [random.randrange(0, 4294967295)]
        if (p.seed):
            seed=[int(p.seed)]
        transform = generation.TransformType(
            diffusion=diffusionMap[p.sampling_method],
        )     
        if (p.imageBinary or p.maskImageBinary):
            start=1.0
            if (not p.maskImageBinary): start=1-float(p.strength)
            if (p.mode=="inpainting_original"):
                start=float(p.strength)            
            sp=generation.ScheduleParameters(start=start)
            step = generation.StepParameter(
                scaled_step=0,
                sampler=generation.SamplerParameters(cfg_scale=p.cfg_value),
                schedule=sp
            )     
        else:    
            step = generation.StepParameter(
                scaled_step=0,
                sampler=generation.SamplerParameters(cfg_scale=p.cfg_value)
            )           
        request_id = str(random.randrange(0, 4294967295))
        image = generation.ImageParameters(
            width=int(SDConfig.width),
            height=int(SDConfig.height),
            steps=int(p.steps),
            samples=int(p.num),
            seed=seed,
            transform=transform,
            parameters=[step]
        )
        request=generation.Request(
            engine_id=SDConfig.model,
            request_id=request_id,
            prompt=prompts,
            image=image
        )
        channel=self.getChannel()
        apiKey = ""
        if (SDConfig.token):    apiKey = SDConfig.token        
        metadata={"authorization": "Bearer " + apiKey}
        client = generation_grpc.GenerationServiceStub(channel)
        images = [0]*p.num
        p.seedList=[0]*p.num      
        imgNum=0  
        try:
            async with client.Generate.open(metadata=metadata)  as stream:
                await stream.send_message(request)
                await stream.end()
                replies = [reply async for reply in stream]
                for k in range(0,len(replies)):
                    reply=replies[k]
                    if (reply.artifacts):
                        for i in range(0,len(reply.artifacts)):                        
                            artifact=reply.artifacts[i]
                            if (artifact.type==generation.ArtifactType.ARTIFACT_IMAGE):
                                image = QImage()
                                image.loadFromData(artifact.binary)
                                images[imgNum]=image
                                p.seedList[imgNum]=str(int(artifact.seed))
                                imgNum+=1
        except Exception as e:
            error_message = traceback.format_exc() 
            self.errorMessage("Server Error","Endpoint: "+SDConfig.url+", Reason: "+error_message)        
            return None

        channel.close()
        return images
    def getChannel(self):
        url=self.SDConfig.url
        if self.endpoint: url=self.endpoint
        print("url",url)
        urlInfo=urlparse(url)
        host=urlInfo.hostname
        port=urlInfo.port
        ssl=False
        if (urlInfo.scheme=="https"): ssl=True
        if (port==443): ssl=True
        channel = Channel(
            host=host,
            port=port,
            ssl=ssl
        )        
        return channel

    async def getModels(self):
        SDConfig=self.SDConfig
        apiKey = SDConfig.token
        engineList=[]
        requestEngines=engines.ListEnginesRequest()
        if (SDConfig.token):    apiKey = SDConfig.token        
        metadata={"authorization": "Bearer " + apiKey}
        channel=self.getChannel()
        client=engines_grpc.EnginesServiceStub(channel)
        try:
            answer=await client.ListEngines(requestEngines, metadata=metadata)
            #print(answer.engine)
            if answer.engine:
                    res=[]
                    for engine in answer.engine:             
                        if engine.type==engines.EngineType.PICTURE:    res.append(engine.id)
                    engineList=res
        except Exception as e:
            error_message = traceback.format_exc() 
            self.errorMessage("Server Error","Endpoint: "+SDConfig.url+", Reason: "+error_message)        
            return None                    
        channel.close()
        return engineList     
