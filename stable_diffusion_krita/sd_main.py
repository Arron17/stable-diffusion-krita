from cmd import PROMPT
from krita import *
from PyQt5.Qt import QByteArray
from PyQt5.QtGui  import QImage, QPixmap
from copy import copy
from pathlib import Path
import sys
from pathlib import Path
import asyncio
import json



if 'stable_diffusion_krita.grpc' in sys.modules:
    del sys.modules["stable_diffusion_krita.grpc"]    
from stable_diffusion_krita.grpc import grpcClient

if 'stable_diffusion_krita.rest' in sys.modules:
    del sys.modules["stable_diffusion_krita.rest"]    
from stable_diffusion_krita.rest import restClient


host = 'localhost'
server_port = 50051
# instantiate a channel
#channel = grpc.insecure_channel(            '{}:{}'.format(host, server_port))
# Stable Diffusion Plugin fpr Krita
# (C) 2022, Nicolay Mausz
# MIT License
#
rPath= Path(Krita.instance().readSetting("","ResourceDirectory",""))
class ModifierData:    
    list=[]
    tags=[]
    def serialize(self):
        obj={"list":self.list,"tags":self.tags}
        return json.dumps(obj)
    def unserialize(self,str):
        obj=json.loads(str)
        self.list=obj["list"]
        self.tags=obj["tags"]
    def save(self):
        str=self.serialize(self)
        Krita.instance().writeSetting ("SDPlugin", "Modifiers", str)

#        with open(rPath / "krita_ai_modifiers.config", 'w', encoding='utf-8') as f_out:#
#            f_out.write(str)
    def load(self):
        str=Krita.instance().readSetting ("SDPlugin", "Modifiers",None)
        if (not str): return
        self.unserialize(self,str)    

#        if (not (rPath / "krita_ai_modifiers.config").exists()): return
#        with open(rPath / "krita_ai_modifiers.config", 'r', encoding='utf-8') as f_in:
 #            str=f_in.read()
  #      self.unserialize(self,str)    

class SDConfig:
    "This is Stable Diffusion Plugin Main Configuration"     
    url = "http://localhost:5000"
    type="SDGRPC"
    token=""
    model="stable-diffusion-v1-4"
    engines=[]
    inpaint_mask_blur=4
    inpaint_mask_content="latent noise"     
    width=512
    height=512
    fixed_res = 2   #New value used to detect if the fixed res box has been ticked
    dlgData={
        "mode": "txt2img",
        "prompt": "",
        "seed": "",
        "steps": 15,
        "steps_update": 50,
        "num": 2,
        "modifiers": "highly detailed\n",
        "cfg_value": 7.5,
        "guidance_strength": 0.618,     #Config value for the CLIP Guidance Strength
        "strength": .75,
        "sampling_method":"LMS"
    }


    def serialize(self):
        obj={"url":self.url,"type":self.type,"token":self.token,"model":self.model,"engines":self.engines,
        "inpaint_mask_blur":self.inpaint_mask_blur, "inpaint_mask_content":self.inpaint_mask_content,
        "width":self.width, "height":self.height,
        "fixed_res":self.fixed_res,
        "type": self.type,
        "dlgData":self.dlgData}
        return json.dumps(obj)
    def unserialize(self,str):
        obj=json.loads(str)
        self.url=obj.get("url","http://localhost:7860")
        self.type=obj.get("type","SDGRPC")
        self.token=obj.get("token","")
        self.model=obj.get("model","")
        self.engines=obj.get("engines",[])
        self.dlgData=obj["dlgData"]
        self.inpaint_mask_blur=obj.get("inpaint_mask_blur",4)
        self.inpaint_mask_content=obj.get("inpaint_mask_content","latent noise")
        self.width=obj.get("width",512)
        self.height=obj.get("height",512)
        self.fixed_res=obj.get("fixed_res",2)
    def save(self):
        str=self.serialize(self)
        Krita.instance().writeSetting ("SDPlugin", "Config", str)

#        with open(rPath / "krita_ai.config", 'w', encoding='utf-8') as f_out:
 #           f_out.write(str)
    def load(self):
        str=Krita.instance().readSetting ("SDPlugin", "Config",None)
        if (not str): return
        self.unserialize(self,str)
      #  if (not (rPath / "krita_ai_modifiers.config").exists()): return

#        with open(rPath / "krita_ai.config", 'r', encoding='utf-8') as f_in:
#            str=f_in.read()

SDConfig.load(SDConfig)

class SDParameters:
    "This is Stable Diffusion Parameter Class"     
    prompt = ""
    negativePrompt=""
    steps = 0
    seed = 0
    num =0
    strength=1.0
    model="stable-diffusion-v1-4"
    sampling_method="LMS",
    seedList =["","","",""]
    imageDialog = None
    regenerate = False
    image64=""
    imageBinary=None
    maskImage64=""
    maskImageBinary=None
    sampling_method="LMS"
    inpaint_mask_blur=4
    inpaint_mask_content="latent noise" 
    mode="txt2img"
    cfg_value=7.5
    strength = 1
    guidance_strength=0.618 #CLIP Guidance Strength
    tiling = False
    restore_faces =False
    cust_width=0    #Two custom values used for Text to Image to send a custom height and width to GRPC
    cust_height=0

def errorMessage(text,detailed):
    msgBox= QMessageBox()
    msgBox.resize(500,200)
    msgBox.setWindowTitle("Stable Diffusion")
    msgBox.setText(text)
    msgBox.setDetailedText(detailed)
    msgBox.setStyleSheet("QLabel{min-width: 700px;}")
    msgBox.exec()

engines=[]
class SDConfigDialog(QDialog):
    def __init__(self):
        super().__init__(None)
        if (not SDConfig.engines): SDConfig.engines=[]
        self.setWindowTitle("Stable Diffusion Configuration")
        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.layout = QVBoxLayout()
        self.type = QComboBox()
        self.layout.addWidget(QLabel('Type'))
        self.type.addItems(['SDGRPC', 'SDCloud',"SDLocal"])
        self.type.setCurrentText(SDConfig.type)
        self.layout.addWidget(self.type,stretch=1)              
        link_label=QLabel('Webservice URL (only SDGRPC or SDLocal)')
        link_label.setOpenExternalLinks(True)
        self.layout.addWidget(link_label)
        self.url = QLineEdit()
        self.url.setText(SDConfig.url)    
        self.layout.addWidget(self.url)
        self.layout.addWidget(QLabel('Auth-Key (only SDCloud or SDGRPC)'))
        self.token = QLineEdit()
        self.token.setText(SDConfig.token)    
        self.layout.addWidget(self.token)      
        self.layout.addWidget(QLabel(''))
        self.layout.addWidget(QLabel('Model (SDCloud or SDGRPC) - refresh list by reopen dialog'))
        self.model = QComboBox()
        self.model.addItems(SDConfig.engines)
        self.model.setCurrentText(SDConfig.model)
        self.layout.addWidget(self.model,stretch=1) 


#        self.layout.addWidget(QLabel('For local version you need this fork running <br> <a href="https://github.com/imperator-maximus/stable-diffusion-webui">imperator-maximus/stable-diffusion-webui</a><br>\n start api.bat there'))

        self.layout.addWidget(QLabel(''))
        

        inpainting_label=QLabel('Inpainting options (only SDLocal)')
        inpainting_label.setToolTip('You can play around with these two values. Default is 4 and "latent noise"')
        self.layout.addWidget(inpainting_label)
        h_layout_inpaint=QHBoxLayout()

        self.inpaint_mask_blur=QLineEdit()
        self.inpaint_mask_blur.setText(str(SDConfig.inpaint_mask_blur))
        h_layout_inpaint.addWidget(QLabel('Mask Blur:'),stretch=1)
        h_layout_inpaint.addWidget(self.inpaint_mask_blur,stretch=1)
        h_layout_inpaint.addWidget(QLabel('Masked Content:'),stretch=1)
        self.inpaint_mask_content = QComboBox()
        self.inpaint_mask_content.addItems(['fill', 'original', 'latent noise', 'latent nothing','g-diffusion'])
        self.inpaint_mask_content.setCurrentText(SDConfig.inpaint_mask_content)
        h_layout_inpaint.addWidget(self.inpaint_mask_content,stretch=1)      
        h_layout_inpaint.addWidget(QLabel(''),stretch=5)

        self.layout.addLayout(h_layout_inpaint)

        self.layout.addWidget(QLabel(''))
        self.layout.addWidget(QLabel('Please do not change these. Only if you really know what you do - otherwise let it to default 512x512'))
        h_layout_size=QHBoxLayout()
        h_layout_size.addWidget(QLabel('Width:'))
        self.width=QLineEdit()
        self.width.setText(str(SDConfig.width))       
        h_layout_size.addWidget(self.width)
        h_layout_size.addWidget(QLabel('Height:'))
        self.height=QLineEdit()
        self.height.setText(str(SDConfig.height))
        h_layout_size.addWidget(self.height)
        h_layout_size.addWidget(QLabel(''),stretch=5)

        self.layout.addLayout(h_layout_size)
        self.layout.addWidget(QLabel(''))
        
        #Add Checkbox to choose Fixed Res
        checkBoxLayout = QHBoxLayout()
        self.fixed_res=QCheckBox("Use Set Resolution (Unticking this box will cause SD to render images at the size of your selection box")
        self.fixed_res.setChecked(int(SDConfig.fixed_res)) #Get the current state from the Config
        self.fixed_res.setToolTip('Ticking this box will use the resolution set above, unticking will use the size of your selection box')
        checkBoxLayout.addWidget(self.fixed_res)
        self.layout.addLayout(checkBoxLayout)
        self.layout.addWidget(QLabel('Warning: If you do not use a fixed res setting your selection box too small will cause bad generations and setting the box too high may cause you to run out of VRAM'))
        self.layout.addWidget(QLabel(''))

        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)
        self.resize(500,200)
    def save(self):
        SDConfig.url=self.url.text()
        SDConfig.token=self.token.text()
        SDConfig.inpaint_mask_blur=int(self.inpaint_mask_blur.text())
        SDConfig.inpaint_mask_content=self.inpaint_mask_content.currentText()
        SDConfig.width=int(self.width.text())
        SDConfig.height=int(self.height.text())
        SDConfig.type=self.type.currentText()
        SDConfig.fixed_res=int(self.fixed_res.checkState())  #Save state of checkbox
        if (SDConfig.type=="SDGRPC"):
            client=grpcClient(SDConfig,"")
            model=self.model.currentText()
            if (not model): model="stable-diffusion-v1-4"
            SDConfig.model=model
            SDConfig.engines=asyncio.run(client.getModels())
        if (SDConfig.type=="SDCloud"):
            client=grpcClient(SDConfig,'https://grpc.stability.ai:443')
            model=self.model.currentText()
            if (not model): model="stable-diffusion-v1-5"
            SDConfig.model=model
            SDConfig.engines=asyncio.run(client.getModels())
        SDConfig.save(SDConfig)

class ModifierDialog(QDialog):
    def __init__(self):
        super().__init__(None)
        self.setWindowTitle("Stable Diffusion Modifier list")
        self.resize(800,500)
        QBtn =  QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.rejected.connect(self.reject)
        grid_layout = QVBoxLayout()
        self.layout=QVBoxLayout()
        self.setLayout(self.layout)

        ModifierData.load(ModifierData)
        self.layout.addLayout(grid_layout)
        for i in range(0,len(ModifierData.list)):
            entry=ModifierData.list[i]
            mod_layout=QVBoxLayout()
            mod_layout.addWidget(QLabel(entry["name"]))
            btn_h_layout=QHBoxLayout()
            btnSelect = QPushButton("")
            btnSelect.setIcon( Krita.instance().icon('select'))        
            btn_h_layout.addWidget(btnSelect,stretch=1)
            btnDelete = QPushButton("")
            btnDelete.setIcon( Krita.instance().icon('deletelayer'))   
            btnSelect.clicked.connect(lambda ch, num=i: self.selectModifier(num))
            btnDelete.clicked.connect(lambda ch, num=i: self.deleteModifier(num))

            btn_h_layout.addWidget(btnDelete,stretch=1)
            btn_h_layout.addWidget(QLabel(""),stretch=5)

            mod_layout.addLayout(btn_h_layout)
            grid_layout.addLayout(mod_layout)

        self.layout.addWidget(QLabel("Name"))
        self.name = QLineEdit()
#        self.name.setText(selected_mod["name"])       
        self.layout.addWidget(self.name) 
   
        self.layout.addWidget(QLabel("Modifiers"))
        self.modifiers=QPlainTextEdit()
        self.modifiers.setPlainText(SDConfig.dlgData.get("modifiers",""))      
        self.layout.addWidget(self.modifiers)     

        self.layout.addWidget(QLabel("Example Prompt"))
        self.example_prompt = QLineEdit()
        self.example_prompt.setText(SDConfig.dlgData.get("prompt",""))       
        self.layout.addWidget(self.example_prompt) 
        h_layout = QHBoxLayout()
        button_save=QPushButton("Add and Select") 
        h_layout.addWidget(button_save)
        self.layout.addLayout(h_layout)
        button_save.clicked.connect(lambda ch, : ModifierDialog.addModifier(self))
        self.layout.addWidget(QLabel(""))        
        self.layout.addWidget(self.buttonBox)
    def addModifier(self):        
        if (not self.name): return
        mod_info={"name":self.name.text(),"modifiers":self.modifiers.toPlainText()}
        ModifierData.list.append(mod_info)
        ModifierData.save(ModifierData)
        SDConfig.dlgData["modifiers"]=self.modifiers.toPlainText()
        self.accept()
    def selectModifier(self,num):
        mod_info=ModifierData.list[num]
        SDConfig.dlgData["modifiers"]=mod_info["modifiers"]
        self.accept()
    def deleteModifier(self,num):
        qm = QMessageBox()
        ret = qm.question(self,'', "Are you sure?", qm.Yes | qm.No)
        if ret == qm.Yes:
            del ModifierData.list[num]
            ModifierData.save(ModifierData)
            self.accept()

    def modifierInput(self,layout):
        layout.addWidget(QLabel("Modifiers"))
        modifiers=QPlainTextEdit()
        modifiers.setPlainText(SDConfig.dlgData.get("modifiers",""))      
        layout.addWidget(modifiers)
        h_layout = QHBoxLayout()
        button_presets=QPushButton("Presets...") 
        h_layout.addWidget(button_presets)
        button_copy_prompt=QPushButton("Copy full Prompt") 
        h_layout.addWidget(button_copy_prompt)        
        layout.addLayout(h_layout)
        button_presets.clicked.connect(lambda ch, : ModifierDialog.openModifierPresets(self))
        button_copy_prompt.clicked.connect(lambda ch, : ModifierDialog.copyPrompt(self))
        return modifiers        

    def copyPrompt(self):
        prompt=getFullPrompt(self)
        QApplication.clipboard().setText(prompt)

    def openModifierPresets(self):
        SDConfig.dlgData["modifiers"]=self.modifiers.toPlainText()
        dlg=ModifierDialog()
        if dlg.exec():            
            self.modifiers.setPlainText(SDConfig.dlgData["modifiers"])


# default dialog for image generation: txt2img, img2img and inpainting
class SDDialog(QDialog):
    def __init__(self,mode,image):
        super().__init__(None)
        SDConfig.dlgData["mode"]=mode
        data=SDConfig.dlgData

        self.setWindowTitle("Stable Diffusion "+data["mode"])

        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.layout = QHBoxLayout()
        formLayout= QVBoxLayout()
        self.layout.addLayout(formLayout)
        formLayout.addWidget(QLabel("Prompt"))
        self.prompt = QLineEdit()
        self.prompt.setText(data["prompt"])            
        formLayout.addWidget(self.prompt)
        self.modifiers= ModifierDialog.modifierInput(self,formLayout)
        if (data["mode"] in ("img2img", "inpainting")):
            if (not SDConfig.type=="SDGRPC" or data["mode"]=="img2img"):
                formLayout.addWidget(QLabel("Denoising Strength"))        
                self.strength=self.addSlider(formLayout,data["strength"]*100,0,100,1,100)

        steps_label=QLabel("Steps")
        steps_label.setToolTip("more steps = slower but often better quality. Recommendation start with lower step like 15 and update in image overview with higher one like 50")
       
        formLayout.addWidget(steps_label)        
        self.steps=self.addSlider(formLayout,data["steps"],1,250,5,1)
        if SDConfig.type=="SDLocal" and mode=="txt2img":
            checkBoxLayout = QHBoxLayout()
            self.restore_faces= QCheckBox("Restore Faces")
            self.restore_faces.setChecked(data.get("restore_faces",False))
            checkBoxLayout.addWidget(self.restore_faces)
            self.tiling= QCheckBox("Tiling")
            self.tiling.setChecked(data.get("tiling",False))
            checkBoxLayout.addWidget(self.tiling)
            formLayout.addLayout(checkBoxLayout)
        formLayout.addWidget(QLabel("Number images"))        
        self.num=self.addSlider(formLayout,data["num"],1,4,1,1)



        formLayout.addWidget(QLabel(""))        

        formLayout.addWidget(QLabel("Advanced"))        
        seed_label=QLabel("Seed (empty=random)")
        seed_label.setToolTip("same seed and same prompt = same image")
        formLayout.addWidget(seed_label)      
        self.seed = QLineEdit()
        self.seed.setText(data["seed"])                  
        formLayout.addWidget(self.seed)   
        cfg_label=QLabel("Guidance Scale")
        cfg_label.setToolTip("how strongly the image should follow the prompt")
        formLayout.addWidget(cfg_label)        
        self.cfg_value=self.addSlider(formLayout,data["cfg_value"]*10,10,300,5,10)
        
        #Adding CLIP Slider
        clip_label=QLabel("CLIP Guidance")
        clip_label.setToolTip("Set a value for CLIP Guidance Strength")
        formLayout.addWidget(clip_label)
        self.guidance_strength=self.addSlider(formLayout,data["guidance_strength"]*1000,0,1000,1,1000)

   
        cfg_label=QLabel("Sampling method")
        cfg_label.setToolTip("")
        formLayout.addWidget(cfg_label)           
        self.sampling_method = QComboBox()
        self.sampling_method.addItems(['LMS', 'Euler a', 'Euler', 'Heun','DPM2','DPM2 a','DDIM','PLMS'])
        self.sampling_method.setCurrentText(data.get("sampling_method","LMS"))
        formLayout.addWidget(self.sampling_method)           
        formLayout.addWidget(QLabel(""))        
        formLayout.addWidget(self.buttonBox)
        if (not data["mode"]=="txt2img"):
            imgLabel=QLabel()        
            self.layout.addWidget(imgLabel) 
            imgLabel.setPixmap(QPixmap.fromImage(image))  
        self.setLayout(self.layout)

    def addSlider(self,layout,value,min,max,steps,divider):
        h_layout =  QHBoxLayout()
        slider = QSlider(Qt.Orientation.Horizontal, self)
        slider.setRange(min, max)

        slider.setSingleStep(steps)
        slider.setPageStep(steps)
        slider.setTickInterval
        label = QLabel(str(value), self)
        h_layout.addWidget(slider, stretch=9)
        h_layout.addWidget(label, stretch=1)
        if (divider!=1):
            slider.valueChanged.connect(lambda: slider.setValue(slider.value()//steps*steps) or label.setText( str(slider.value()/divider)))
        else:
            slider.valueChanged.connect(lambda: slider.setValue(slider.value()//steps*steps) or label.setText( str(slider.value())))
        slider.setValue(int(value))
        layout.addLayout(h_layout)
        return slider
    # put data from dialog in configuration and save it        
    def setDlgData(self):
        SDConfig.dlgData["prompt"]=self.prompt.text()
        SDConfig.dlgData["steps"]=int(self.steps.value())
        SDConfig.dlgData["seed"]=self.seed.text()
        SDConfig.dlgData["num"]=int(self.num.value())
        SDConfig.dlgData["cfg_value"]=self.cfg_value.value()/10
        SDConfig.dlgData["guidance_strength"]=self.guidance_strength.value()/1000 #CLIP Slider
        SDConfig.dlgData["modifiers"]=self.modifiers.toPlainText()
        SDConfig.dlgData["sampling_method"]=self.sampling_method.currentText()
        if (SDConfig.type=="SDLocal" and SDConfig.dlgData["mode"]=="txt2img"):
            SDConfig.dlgData["tiling"]=self.tiling.isChecked()
            SDConfig.dlgData["restore_faces"]=self.restore_faces.isChecked()

        if SDConfig.dlgData["mode"] in ("img2img", "inpainting"):
            if (not SDConfig.type=="SDGRPC" or SDConfig.dlgData["mode"] =="img2img"):
                SDConfig.dlgData["strength"]=self.strength.value()/100
        SDConfig.save(SDConfig)
# put image in Krita on new layer or existing one
def selectImage(p: SDParameters,qImg):  
    d = Application.activeDocument()
    n = d.activeNode()
    s = d.selection()        
    root = d.rootNode()
    n = d.createNode(p.prompt, "paintLayer")
    root.addChildNode(n, None)

    if (p.mode=="img2img" or p.mode=="inpainting"):
        #Force selection box to be divisible by 8
        if SDConfig.fixed_res==0:
            w = s.width()    
            h = s.height()    
            w = int(w / 8 + 0.5) * 8
            h = int(h / 8 + 0.5) * 8
            qImg = qImg.scaled(w, h)
        elif SDConfig.fixed_res==2:
            qImg = qImg.scaled(s.width(), s.height())
    else:
        qImg = qImg.scaled(s.width(), s.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

    ptr = qImg.bits()
    ptr.setsize(qImg.byteCount())           
    n.setPixelData(QByteArray(ptr.asstring()),s.x(),s.y(),qImg.width(),qImg.height())
    d.waitForDone ()
    d.refreshProjection() 
class showTiling(QDialog):
    def __init__(self,qImg):
        super().__init__(None)
        self.setWindowTitle("Tiling Preview")
        layout=QVBoxLayout()
        imgLabel=QLabel()
        h_layout=QHBoxLayout()
        h_layout.addWidget(imgLabel) 
        canvas = QtGui.QPixmap(750, 750)
        imgLabel.setPixmap(canvas)
        painter = QtGui.QPainter(imgLabel.pixmap())
        qp=QPixmap.fromImage(qImg).scaled(250,250,Qt.KeepAspectRatio)
        height=qp.height()
        painter.drawPixmap(0,0,qp)
        painter.drawPixmap(250,0,qp)
        painter.drawPixmap(500,0,qp)
        painter.drawPixmap(0,height,qp)
        painter.drawPixmap(250,height,qp)
        painter.drawPixmap(500,height,qp)
        painter.drawPixmap(0,height*2,qp)
        painter.drawPixmap(250,height*2,qp)
        painter.drawPixmap(500,height*2,qp)
        layout.addLayout(h_layout)
        self.setLayout(layout)

# asking for image of result set and update option
class showImages(QDialog):
    def __init__(self,qImgs,p: SDParameters):
        super().__init__(None)
        self.qImgs=qImgs
        self.setWindowTitle("Result")
        QBtn =  QDialogButtonBox.Cancel
        self.SDParam=p
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        top_layout=QVBoxLayout()        
        prompt_layout=QHBoxLayout()        
        top_layout.addLayout(prompt_layout)
        self.prompt = QLineEdit()
        self.prompt.setText(SDConfig.dlgData.get("prompt",""))
        prompt_layout.addWidget(self.prompt,stretch=9)
        btn_regenerate=QPushButton("Generate with steps "+str(SDConfig.dlgData["steps"]))         
        btn_regenerate.clicked.connect(self.regenerateStart)
        prompt_layout.addWidget(btn_regenerate,stretch=1)
        self.modifiers= ModifierDialog.modifierInput(self,top_layout)

        layout=QHBoxLayout()
        top_layout.addLayout(layout)
        self.imgLabels=[0]*p.num
        self.seedLabel=[0]*p.num
        self.qImgs=qImgs
        i=0
        for qImg in qImgs:       
            v_layout = QVBoxLayout()
            layout.addLayout(v_layout)       
            imgLabel=QLabel()
            v_layout.addWidget(imgLabel) 
            imgLabel.setPixmap(QPixmap.fromImage(qImg).scaled(380,380,Qt.KeepAspectRatio))  
            if p.tiling:
                imgLabel.mousePressEvent = (lambda ch, num=i: self.clickImage(self.qImgs[num]))

            seedLabel=QLabel(p.seedList[i])
            seedLabel.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self.seedLabel[i]=seedLabel
            v_layout.addWidget(seedLabel)     
            self.imgLabels[i]=imgLabel
            h_layout = QHBoxLayout()
            v_layout.addLayout(h_layout)               
            btn=QPushButton("Select")          
            h_layout.addWidget(btn,stretch=1)
            btn.clicked.connect(lambda ch, num=i: selectImage(p,self.qImgs[num]))
            btnUpdate=QPushButton()  
            btnUpdate.setIcon( Krita.instance().icon('updateColorize'))        
            h_layout.addWidget(btnUpdate,stretch=1)
            btnUpdate.clicked.connect(lambda ch, num=i: self.updateImageStart(num))
            i=i+1
            
        #layout.addWidget(self.buttonBox)
        top_layout.addWidget(QLabel("Update one image with new Steps value"))
        self.steps_update=SDDialog.addSlider(self,top_layout,SDConfig.dlgData.get("steps_update",50),1,250,5,1)
        
        if (p.mode in ("img2img", "inpainting")):
            if (not SDConfig.type=="SDGRPC" or p.mode =="img2img"):
                top_layout.addWidget(QLabel("Update with new Strengths value"))
                self.strength_update=SDDialog.addSlider(self,top_layout,SDConfig.dlgData.get("strength",0.5)*100,0,100,1,100)

        self.setLayout(top_layout)

    def clickImage(self,qImg):
        dlg = showTiling(qImg)
        if dlg.exec():
            print("Tiling closed")        



    # start request for HQ version of one image
    def regenerateStart(self):
        p = copy(self.SDParam)
        #SDConfig.dlgData["steps_update"]=self.steps_update.value()
        #p.steps=SDConfig.dlgData["steps_update"]
        #SDConfig.save(SDConfig)
        SDConfig.dlgData["prompt"]=self.prompt.text()
        SDConfig.dlgData["modifiers"]=self.modifiers.toPlainText()
        SDConfig.save(SDConfig)
        p.prompt= getFullPrompt(self)        
        p.imageDialog=self
        p.regenerate=True
        runSD(p)

    def updateImages(self,qImgs,seeds):
        i=0
        self.qImgs=qImgs
        self.SDParam.seedList=seeds
        for qImg in qImgs:
            imgLabel=self.imgLabels[i]
            seedLabel=self.seedLabel[i]
            seedLabel.setText(seeds[i])
            imgLabel.setPixmap(QPixmap.fromImage(qImg).scaled(380,380,Qt.KeepAspectRatio))  
            i=i+1
    # update one single image with new parameters
    def updateImageStart(self,num):
        p = copy(self.SDParam)
        p.seed=p.seedList[num]
        if (p.mode in ("img2img", "inpainting")): 
            if (not SDConfig.type=="SDGRPC" or p.mode =="img2img"):
                p.strength=self.strength_update.value()/100
        SDConfig.dlgData["steps_update"]=self.steps_update.value()
        SDConfig.save(SDConfig)
        p.num=1
        p.steps=SDConfig.dlgData["steps_update"]
        SDConfig.dlgData["prompt"]=self.prompt.text()
        SDConfig.dlgData["modifiers"]=self.modifiers.toPlainText()
        p.prompt= getFullPrompt(self)        
        self.updateImageNum=num
        p.imageDialog=self
        runSD(p)

    # update image with HQ version       
    def updateImage(self,qImg):
        num=self.updateImageNum
        imgLabel=self.imgLabels[num]
        self.qImgs[num]=qImg
        imgLabel.setPixmap(QPixmap.fromImage(qImg).scaled(380,380,Qt.KeepAspectRatio))  


def imageResultDialog(qImgs,p):
    dlg = showImages(qImgs,p)
    if dlg.exec():
        print("HQ Update here")


def runSD(p: SDParameters):
    # finally the long waited interface change is here!
    if (SDConfig.type=="SDGRPC"):
        client=grpcClient(SDConfig,"")
        images=asyncio.run(client.runSD(p))
    elif (SDConfig.type=="SDCloud"):
        client=grpcClient(SDConfig,'https://grpc.stability.ai:443')
        images=asyncio.run(client.runSD(p))        
    else:
        client=restClient(SDConfig)
        images=client.runSD(p)
    if (p.imageDialog):                 # only refresh image
        if (p.regenerate):
            p.imageDialog.updateImages(images,p.seedList)
        else:  
            p.imageDialog.updateImage(images[0])
    return images


def getDocument():
    d = Application.activeDocument()
    if (d==None):  errorMessage("Please add a document","Needs document with a layer and selection.")
    return d

def getLayer():
    d=getDocument()
    if (d==None):  return
    n = d.activeNode()
    return n

def getSelection():
    d = getDocument()
    if (d==None): return
    s = d.selection()
    if (s==None):  errorMessage("Please make a selection","Operation runs on a selection only. Please use rectangle select tool.")
    return s      

def getFullPrompt(dlg):
    modifiers=""
    list=dlg.modifiers.toPlainText().split("\n")
    for i in range(0,len(list)):
        m=list[i]
        if (m and m[0]!="#" and m[0]!="-"): modifiers+=", "+m

   # modifiers=dlg.modifiers.toPlainText().replace("\n", ", ")
    prompt=dlg.prompt.text()
    if (not prompt):      
        errorMessage("Empty prompt","Type some text in prompt input box about what you want to see.")
        return ""
    prompt+=modifiers
    return prompt
def getNegativePrompt(dlg):
    negativePrompt=""
    list=dlg.modifiers.toPlainText().split("\n")
    for i in range(0,len(list)):
        m=list[i]
        if (m and m[0]=="-"): 
            m = m[1:]           #remove the first character (-) from the prompt
            negativePrompt+=m   #don't bother adding a (,) on as this is never going to be added anywhere but the negative
    return negativePrompt

def TxtToImage():
    s=getSelection()
    if (s==None):   return   
    SDConfig.load(SDConfig)
    dlg = SDDialog("txt2img",None)
    dlg.resize(700,200)
    if dlg.exec():
        dlg.setDlgData()
        p = SDParameters()
        p.prompt=getFullPrompt(dlg)
        if not p.prompt: return       
        p.negativePrompt=getNegativePrompt(dlg)
        p.mode="txt2img"
        data=SDConfig.dlgData
        p.steps=data["steps"]
        p.seed=data["seed"]
        p.num=data["num"]
        p.restore_faces=data.get("restore_faces",False)
        p.tiling=data.get("tiling",False)        
        p.sampling_method=data["sampling_method"]
        p.cfg_value=data["cfg_value"]
        p.guidance_strength=data["guidance_strength"]
        if SDConfig.fixed_res==0:
            w = s.width()    
            h = s.height()    
            w = int(w / 8 + 0.5) * 8
            h = int(h / 8 + 0.5) * 8
            p.cust_width=w
            p.cust_height=h
        images = runSD(p)
        imageResultDialog( images,p)


def ImageToImage():
    s=getSelection()
    if (s==None):   return    
    n=getLayer()     
    
    if SDConfig.fixed_res==0:
        #Force selection box to be divisible by 8
        w = s.width()    
        h = s.height()    
        w = int(w / 8 + 0.5) * 8
        h = int(h / 8 + 0.5) * 8
        
        data=n.pixelData(s.x(),s.y(),w,h)
        image=QImage(data.data(),w,h,QImage.Format_RGBA8888).rgbSwapped()
        image = image.scaled(w,h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
    elif SDConfig.fixed_res==2:
        data=n.pixelData(s.x(),s.y(),s.width(),s.height())
        image=QImage(data.data(),s.width(),s.height(),QImage.Format_RGBA8888).rgbSwapped()
        image = image.scaled(SDConfig.width,SDConfig.height, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        
    data = QByteArray()
    buf = QBuffer(data)
    image.save(buf, 'PNG')
    imageBinary=data.data()
    ba=data.toBase64()
    DataAsString=str(ba,"ascii")
    image64 = "data:image/png;base64,"+DataAsString
    image64 = DataAsString
    
    dlg = SDDialog("img2img",image)
    dlg.resize(900,200)

    if dlg.exec():
        dlg.setDlgData()
        p = SDParameters()
        p.prompt=getFullPrompt(dlg)
        if not p.prompt: return
        p.negativePrompt=getNegativePrompt(dlg)
        data=SDConfig.dlgData
        p.mode="img2img"        
        p.steps=data["steps"]
        p.seed=data["seed"]
        p.num=data["num"]
#        p.restore_faces=data["restore_faces"]
        p.sampling_method=data["sampling_method"]
#        p.tiling=data["tiling"]        
        p.cfg_value=data["cfg_value"]
        p.guidance_strength=data["guidance_strength"]
        p.image64=image64
        p.imageBinary=imageBinary
        p.strength=data["strength"]
        images = runSD(p)
        imageResultDialog( images,p)


def Inpainting():    
    #es = str(SDConfig.fixed_res)
    #errorMessage(es,"Value of fixed_res")
    n = getLayer()
    if (n==None):  return    
    s=getSelection()
    if (s==None):   return
    d = getDocument()
    
    if SDConfig.fixed_res==0:
        #Force selection box to be divisible by 8 (My Code)
        w = s.width()    
        h = s.height()    
        w = int(w / 8 + 0.5) * 8
        h = int(h / 8 + 0.5) * 8
        
        data=d.pixelData(s.x(),s.y(),w,h)
        image=QImage(data.data(),w,h,QImage.Format_RGBA8888).rgbSwapped()
        image = image.scaled(w, h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation) #My Change
    elif SDConfig.fixed_res==2:    
        data=d.pixelData(s.x(),s.y(),s.width(),s.height())
        image=QImage(data.data(),s.width(),s.height(),QImage.Format_RGBA8888).rgbSwapped()
        image = image.scaled(SDConfig.width,SDConfig.height, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)        # not using config here
    
    
    print(image.width(),image.height())        
    data = QByteArray()
    buf = QBuffer(data)
    image.save(buf, 'PNG')
    imageBinary=data.data()
    ba=data.toBase64()
    DataAsString=str(ba,"ascii")
    #image64 = "data:image/png;base64,"+DataAsString
    image64 = DataAsString

    maskImage=QPixmap(image.width(), image.height()).toImage()
    maskImage = maskImage.convertToFormat(QImage.Format_ARGB32)
    # generate mask image
    maskImage.fill(QColor(Qt.black).rgb())
    foundTrans=False
    foundPixel=False
    for i in range(image.width()):
        for j in range(image.height()):
            rgb = image.pixel(i, j)
            alpha = qAlpha(rgb)
            if (alpha !=255):
                foundTrans=True
                maskImage.setPixel(i, j, QColor(Qt.white).rgb())
            else: foundPixel=True

    if (foundTrans==False):
        errorMessage("No transparent pixels found","Needs content with part removed by eraser (Brush in Tool palette + Right click for Eraser selection)")
        return
    if (foundPixel==False):
        errorMessage("No  pixels found","Maybe wrong layer selected? Choose one with some content n it.")
        return        
    data = QByteArray()
    buf = QBuffer(data)
    maskImage.save(buf, 'PNG')
    ba=data.toBase64()
    maskImageBinary=data.data()
    DataAsString=str(ba,"ascii")
   # maskImage64 = "data:image/png;base64,"+DataAsString
    maskImage64 =DataAsString
    SDConfig.load(SDConfig)
    image = image.scaled(380,380, Qt.KeepAspectRatio, Qt.SmoothTransformation)  # preview smaller
    dlg = SDDialog("inpainting",image)
    dlg.resize(900,200)

    if dlg.exec():
        dlg.setDlgData()
        p = SDParameters()
        p.prompt=getFullPrompt(dlg)
        p.negativePrompt=getNegativePrompt(dlg)
        if not p.prompt: return        
        data=SDConfig.dlgData
        p.mode="inpainting"
        p.steps=data["steps"]
        p.seed=data["seed"]
        p.num=data["num"]
#        p.restore_faces=data.get("restore_faces",False)
#        p.tiling=data.get("tiling",False)
        p.cfg_value=data["cfg_value"]
        p.guidance_strength=data["guidance_strength"]
        p.strength=data["strength"]
        p.sampling_method=data["sampling_method"]
        p.image64=image64
        p.imageBinary=imageBinary
        p.maskImage64=maskImage64
        p.maskImageBinary=maskImageBinary
        images = runSD(p)
        imageResultDialog( images,p)

# config dialog
def Config():
    dlg=SDConfigDialog()
    if dlg.exec():
        dlg.save()

# expand selection to max size        
def expandSelection():
    d = getDocument()

    if (d==None): return
    s = d.selection()    
    if (not s):  x=0;y=0
    else: x=s.x(); y=s.y()     
    s2 = Selection()    
    s2.select(x, y, SDConfig.width, SDConfig.height, 1)
    d.setSelection(s2)
    d.refreshProjection()


#Inpainting()
#TxtToImage()
#ImageToImage()
#Config()
#expandSelection()

