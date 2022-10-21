from PyQt5.QtGui  import QImage
import urllib.request
import http.client
import json
from PyQt5.Qt import QByteArray
from PyQt5.QtGui  import QImage, QPixmap
import traceback
from krita import *

class restClient:
    def __init__(self,SDConfig):
        self.SDConfig = SDConfig

    def errorMessage(self,text,detailed):
        msgBox= QMessageBox()
        msgBox.resize(500,200)
        msgBox.setWindowTitle("Stable Diffusion")
        msgBox.setText(text)
        msgBox.setDetailedText(detailed)
        msgBox.setStyleSheet("QLabel{min-width: 700px;}")
        msgBox.exec()        
    def getServerData(self,reqData):
        endpoint=self.SDConfig.url
        endpoint=endpoint.strip("/")
        endpoint+="/api/" 
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }    
        try:
            req = urllib.request.Request(endpoint, reqData, headers)
            with urllib.request.urlopen(req) as f:
                res = f.read()
                return res
        except http.client.IncompleteRead as e:
            print("Incomplete Read Exception - better restart Colab or ")
            res = e.partial 
            return res           
        except Exception as e:
            error_message = traceback.format_exc() 
            self.errorMessage("Server Error","Endpoint: "+endpoint+", Reason: "+error_message)        
            return None
    # convert image from server result into QImage
    def base64ToQImage(self,data):
    #   data=data.split(",")[1] # get rid of data:image/png,
        image64 = data.encode('ascii')
        imagen = QtGui.QImage()
        bytearr = QtCore.QByteArray.fromBase64( image64 )
        imagen.loadFromData( bytearr, 'PNG' )      
        return imagen
    def runSD(self,p):
        SDConfig=self.SDConfig
        if (not p.seed): seed=-1
        else: seed=int(p.seed)
        inpainting_fill_options= ['fill', 'original', 'latent noise', 'latent nothing',"g-diffusion"]
        inpainting_fill=inpainting_fill_options.index(SDConfig.inpaint_mask_content)
        print(inpainting_fill)
        j = {'prompt': p.prompt, 
            'mode': p.mode, 
            'initimage': {'image':p.image64, 'mask':p.maskImage64}, 
            'steps':p.steps, 
            'sampler':p.sampling_method, 
            'mask_blur': self.SDConfig.inpaint_mask_blur, 
            'inpainting_fill':inpainting_fill, 
            'tiling':p.tiling,
            'restore_faces':p.restore_faces,
            'use_gfpgan': False, 
            'batch_count': p.num, 
            'cfg_scale': p.cfg_value, 
            'denoising_strength': p.strength, 
            'seed':seed, 
            'height':self.SDConfig.height, 
            'width':self.SDConfig.width, 
            'resize_mode': 0, 
            'upscaler':'RealESRGAN', 
            'upscale_overlap':64, 
            'inpaint_full_res':True, 
            'inpainting_mask_invert': 0 
            }    
        data = json.dumps(j).encode("utf-8")
        res=self.getServerData(data)
        if not res: return    
        response=json.loads(res)
    #  print(response)
        images = [0]*p.num
        p.seedList=[0]*p.num
        s=response["info"]
        info=json.loads(s)

        firstSeed=int(info["seed"])
        if (p.num==1):
            data = response["images"][0] # first image
            p.seedList[0]=str(int(firstSeed))
            images[0]=self.base64ToQImage(data)
        else:
            for i in range(0,p.num):
                data = response["images"][i+1] # first image
                p.seedList[i]=str(int(firstSeed)+i)
                images[i]=self.base64ToQImage(data)
        return images