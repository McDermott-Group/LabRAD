import Tkinter
from Tkinter import Tk, Frame, BOTH, OptionMenu, Spinbox, Canvas, Button, Checkbutton, Label, StringVar, IntVar, Entry
import labrad
import os

#checking how git handles usernames
# map from monitor bus name to allowed settings for that bus.
# for each bus, the settings are given as a map from name to numeric code.

with labrad.connect() as cxn:
    
    try:
       dc = cxn.dc_rack_server
    except:
       print 'dc rack server inopperable'
       
    class DCRackGui(Frame):

        w = 72
        h = 36
      
        def __init__(self, parent):
            Frame.__init__(self, parent, background="gray")   
             
            self.parent = parent
            
            self.initUI()
        
        def initUI(self):
            self.parent.title("DC Rack Controller")
            self.pack(fill=BOTH, expand=1)
            wScreen =720
            hScreen =540 #self.parent.winfo_screenwidth(), self.parent.winfo_screenheight()
            self.parent.geometry("%dx%d+0+0" % (wScreen, hScreen))
            self.parent.resizable(width=False, height=False)
            scriptPath = os.getcwd()
            self.parent.iconbitmap((scriptPath+r'\Qbucky.ico'))

            busSettings_width = 640
            busSettings_height = 200
            self.busSettingsCanvas =Canvas(self, width=busSettings_width, height=busSettings_height)
            self.busSettingsCanvas.place(x=40, y=20)

            busSettings_width = 640
            busSettings_height = 250
            self.busSettingsCanvas =Canvas(self, width=busSettings_width, height=busSettings_height)
            self.busSettingsCanvas.place(x=40, y=270)

            self.initializeDcRackDataStructures() 
            self.busSettingControls()

            self.InitDacVrefsButton = Button(self, text="Init DAC Vrefs", command=self.initDacVrefs)
            self.InitDacVrefsButton.place(x=DCRackGui.w*(1+1), y=DCRackGui.h*(4.5+2))
                
            self.SaveSettingsToRegistryButton = Button(self, text="Save To Registry", command=self.saveToRegistry)
            self.SaveSettingsToRegistryButton.place(x=DCRackGui.w*(1+3), y=DCRackGui.h*(4.5+2))
                
            self.UploadSettingsFromRegistryButton = Button(self, text="Upload From Registry", command=self.uploadFromRegistry)
            self.UploadSettingsFromRegistryButton.place(x=DCRackGui.w*(1+5), y=DCRackGui.h*(4.5+2))
            
            self.preampCardSelectorLabel = Label(self, text="Select Preamp Card:")
            self.preampCardSelectorLabel.place(x=DCRackGui.w*0.6, y = DCRackGui.h*(7.75))

            self.preampCard = StringVar()
            self.availablePreampCards = []
            for item in self.availableCards:
               if 'preamp' in item:
                  self.availablePreampCards.append(item)
            self.preampCardOptionMenu = OptionMenu(self,self.preampCard,*self.availablePreampCards, command = lambda event, card=self.preampCard: self.updatePreampCardSettings(card))
            self.preampCardOptionMenu.configure(width=10)
            self.preampCardOptionMenu.place(x=DCRackGui.w*(2.25), y=DCRackGui.h*(7.75))
            
            self.preampSettingControls(self.preampCard)
  
        def updatePreampCardSettings(self, card):
            cardStr = card.get()
            cardID = int(cardStr.split(",")[0].strip("("))
            dc.select_card(cardID)
            for chan in self.PreampChannels:
                for setting in self.PreampSettings:
                   if "High-Pass" in setting:
                      dc.change_high_pass_filter(chan, self.PreampSettingsData[cardStr][chan]["High-Pass \n Time Constant [us]"]) #set preamp as cards are intialized upon power up
                   elif "Low-Pass" in setting:
                      dc.change_low_pass_filter(chan, self.PreampSettingsData[cardStr][chan]["Low-Pass \n Time Constant [us]"])
                   elif "Polarity" in setting:
                      dc.change_polarity(chan, self.PreampSettingsData[cardStr][chan]["Polarity"])
                   elif "Offset" in setting:
                      dc.change_dc_offset(chan, self.PreampSettingsData[cardStr][chan]["Offset"])
                   self.PreampOptionMenusData[chan][setting].set(self.PreampSettingsData[card.get()][chan][setting])
            self.preampSettingControls(card)
                    
        def initializeDcRackDataStructures(self):
            availableRacks = dc.list_devices()
            if len(availableRacks)==1:
               dc.select_device(availableRacks[0][0])
               print "Device selected with ID=",availableRacks[0][0], " using server name=", availableRacks[0][1]
            elif len(availableRacks)>0:
               print "Multiple dc rack keys found in registry"
            elif len(availableRacks)==0:
               print "No dc rack keys found in registry"
               
            cards = dc.list_cards()
            self.availableCards = []#["(2,preamp)", "(13,fastbias)", "(17,preamp)", "(34,fastbias)"]
            for ii in range(0, len(cards)):
               self.availableCards.append("("+str(cards[ii][0])+","+str(cards[ii][1])+")")
               
            dc.init_dacs()

            self.BusChannelLabels={}
            self.BusSettingsLabels={}
            self.BusSettingsListDict={}
            self.BusOptionMenus={}
            self.BusOptionMenusData={}

            self.BusMostRecentSetting={}
            
            self.BusSettings = ["On/Off", "Card Address/Type", "Channel"]
            self.BusChannels = ["Dbus0", "Dbus1", "Abus0", "Abus1"]
            for jj in range(0, len(self.BusChannels)):
                self.BusOptionMenusData[self.BusChannels[jj]]={}
                self.BusOptionMenus[self.BusChannels[jj]]={}
                self.BusMostRecentSetting[self.BusChannels[jj]]=""
            
            self.BusSettingsListDict["On/Off"]=["On", "Off"]
            self.BusSettingsListDict["Card Address/Type"]=self.availableCards
            self.BusSettingsListDict["Channel"] = ["", ""]

            
            self.FOout = IntVar()
            self.FOflash = IntVar()
            self.RegLoadFlash = IntVar()
            
            self.PreampSettingsListDict={}
            self.PreampChannelLabels={}
            self.PreampSettingsLabels={}
            self.PreampOptionMenus={}
            self.PreampOptionMenusData={}
            self.PreampSettingsData={}
            self.PreampLedStateData={}
     
            self.PreampChannels= ["A", "B", "C", "D"]
            self.PreampSettings = ["High-Pass \n Time Constant [us]", "Low-Pass \n Time Constant [us]", "Polarity", "Offset"]
            self.PreampLedFlashOptions=["FOout", "FOflash", "RegLoadFlash"]
                
            for ii in range(0, len(self.availableCards)):
                self.PreampSettingsData[self.availableCards[ii]]={}
                self.PreampLedStateData[self.availableCards[ii]]={}
                for ll in range(0, len(self.PreampLedFlashOptions)):
                   self.PreampLedStateData[self.availableCards[ii]][self.PreampLedFlashOptions[ll]]=0
                for jj in range(0, len(self.PreampChannels)):
                    self.PreampSettingsData[self.availableCards[ii]][self.PreampChannels[jj]]={}
                    for kk in range(0, len(self.PreampSettings)):
                        if "High-Pass \n Time Constant [us]" in self.PreampSettings[kk]:
                           self.PreampSettingsData[self.availableCards[ii]][self.PreampChannels[jj]][self.PreampSettings[kk]]='DC'
                        elif "Low-Pass \n Time Constant [us]" in self.PreampSettings[kk]:
                           self.PreampSettingsData[self.availableCards[ii]][self.PreampChannels[jj]][self.PreampSettings[kk]]='0'
                        elif "Polarity" in self.PreampSettings[kk]:
                           self.PreampSettingsData[self.availableCards[ii]][self.PreampChannels[jj]][self.PreampSettings[kk]]='positive'
                        elif "Offset" in self.PreampSettings[kk]:
                           self.PreampSettingsData[self.availableCards[ii]][self.PreampChannels[jj]][self.PreampSettings[kk]]=0
                        else:
                           print "setting not found"
                
            for ii in range(0, len(self.PreampChannels)):
                self.PreampOptionMenus[self.PreampChannels[ii]]={}
                self.PreampOptionMenusData[self.PreampChannels[ii]]={}
                
            self.PreampSettingsListDict["High-Pass \n Time Constant [us]"] =['DC','3300','1000','330','100','33','10','3.3']
            self.PreampSettingsListDict["Low-Pass \n Time Constant [us]"] =['0','0.22','0.5','1.0','2.2','5','10','22']
            self.PreampSettingsListDict["Polarity"] =['positive', 'negative']


        def busSettingControls(self):
            configState = "normal"

            for jj in range(0, len(self.BusChannels)):
                self.BusChannelLabels[self.BusChannels[jj]] = Label(self, text=self.BusChannels[jj])
                self.BusChannelLabels[self.BusChannels[jj]].configure(state=configState)
                self.BusChannelLabels[self.BusChannels[jj]].place(x=DCRackGui.w, y = DCRackGui.h*(jj+2))
                for kk in range(0, len(self.BusSettings)):
                     if jj==0:
                        self.BusSettingsLabels[self.BusSettings[kk]] = Label(self, text=self.BusSettings[kk])
                        self.BusSettingsLabels[self.BusSettings[kk]].place(x=DCRackGui.w*(1+1.25+kk*2), y = DCRackGui.h)
                        self.BusSettingsLabels[self.BusSettings[kk]].configure(state=configState)
                     if self.BusSettings[kk] not in self.BusOptionMenusData[self.BusChannels[jj]].keys():
                        if "On/Off" not in self.BusSettings[kk]:
                           self.BusOptionMenusData[self.BusChannels[jj]][self.BusSettings[kk]]=StringVar()
                           self.BusOptionMenus[self.BusChannels[jj]][self.BusSettings[kk]]=OptionMenu(self,self.BusOptionMenusData[self.BusChannels[jj]][self.BusSettings[kk]],*self.BusSettingsListDict[self.BusSettings[kk]], command=lambda event, channel= self.BusChannels[jj],settingName=self.BusSettings[kk],settingVal=self.BusOptionMenusData[self.BusChannels[jj]][self.BusSettings[kk]]:self.setBusSetting(channel, settingName, settingVal))
                           self.BusOptionMenus[self.BusChannels[jj]][self.BusSettings[kk]].configure(width=12)
                           self.BusOptionMenus[self.BusChannels[jj]][self.BusSettings[kk]].place(x=DCRackGui.w*(2.25+2*kk), y=DCRackGui.h*(2+jj))
                           if 'Channel' in self.BusSettings[kk]:
                              self.BusOptionMenus[self.BusChannels[jj]][self.BusSettings[kk]].configure(state='disabled')
                        else:
                           self.BusOptionMenusData[self.BusChannels[jj]][self.BusSettings[kk]]=IntVar()
                           self.BusOptionMenus[self.BusChannels[jj]][self.BusSettings[kk]]=Checkbutton(self,variable=self.BusOptionMenusData[self.BusChannels[jj]][self.BusSettings[kk]],text="           ", command=lambda channel= self.BusChannels[jj],settingName=self.BusSettings[kk],settingVal=self.BusOptionMenusData[self.BusChannels[jj]][self.BusSettings[kk]]:self.setBusSetting(channel, settingName, settingVal))
                           self.BusOptionMenus[self.BusChannels[jj]][self.BusSettings[kk]].configure(width=10)
                           self.BusOptionMenus[self.BusChannels[jj]][self.BusSettings[kk]].place(x=DCRackGui.w*(2.25+2*kk), y=DCRackGui.h*(2+jj))
                     else:
                        if "Channel" in self.BusSettings[kk]:

                           self.BusOptionMenus[self.BusChannels[jj]][self.BusSettings[kk]].destroy()
                           associatedCard = self.BusOptionMenusData[self.BusChannels[jj]]["Card Address/Type"].get()
                           newList = self.getChannelOptionsList(self.BusChannels[jj],associatedCard)
         
                           self.BusOptionMenus[self.BusChannels[jj]][self.BusSettings[kk]]=OptionMenu(self,self.BusOptionMenusData[self.BusChannels[jj]][self.BusSettings[kk]],*newList, command=lambda event, channel= self.BusChannels[jj],settingName=self.BusSettings[kk],settingVal=self.BusOptionMenusData[self.BusChannels[jj]][self.BusSettings[kk]]:self.setBusSetting(channel, settingName, settingVal))
                           if len(associatedCard)==0:
                              self.BusOptionMenus[self.BusChannels[jj]][self.BusSettings[kk]].configure(state='disabled')
                           self.BusOptionMenus[self.BusChannels[jj]][self.BusSettings[kk]].configure(width=12)
                           self.BusOptionMenus[self.BusChannels[jj]][self.BusSettings[kk]].place(x=DCRackGui.w*(2.25+2*kk), y=DCRackGui.h*(2+jj))
                           
        
        def setBusSetting(self, busName, settingName, settingVal):
            if "On/Off" in settingName:
                self.setBusState(busName, settingVal)
            elif "Card Address/Type" in settingName:
                self.setBusCardMapping(busName, settingVal)
            elif "Channel" in settingName:
                self.setBusChannelMapping(busName, self.BusOptionMenusData[busName]["Card Address/Type"] ,settingVal)
            else:
               print "Unknown bus setting encountered, please call a bug exterminator immeadiately."

        def preampSettingControls(self, cardSelection):
            card = cardSelection.get()
            if 'fastbias' in card or len(card)==0:
                card = 'fastbias'
         
            if 'fastbias' in card:
                configState = "disabled"
            else:
                configState = "normal"

            if not hasattr(self, 'FOoutButton'):
               self.FOoutButton = Checkbutton(self, variable=self.FOout, text="FO Out", command=lambda ledType="FO Out Switch":self.setLEDs(ledType))
               self.FOoutButton.place(x=DCRackGui.w*(4), y=DCRackGui.h*(7.75))
            else:
               self.FOout.set(self.PreampLedStateData[card]["FOout"])
            self.FOoutButton.configure(state=configState)

            if not hasattr(self, 'FOflashButton'):
               self.FOflashButton = Checkbutton(self, variable=self.FOflash, text="FO Flash", command=lambda ledType="FO Flash Switch":self.setLEDs(ledType))
               self.FOflashButton.place(x=DCRackGui.w*(5.5), y=DCRackGui.h*(7.75))
            else:
               self.FOflash.set(self.PreampLedStateData[card]["FOflash"])
            self.FOflashButton.configure(state=configState)

            if not hasattr(self, 'RegLoadFlashButton'):
               self.RegLoadFlashButton = Checkbutton(self, variable=self.RegLoadFlash, text="Reg Load Flash", command=lambda ledType="Reg Load Flash Switch":self.setLEDs(ledType))
               self.RegLoadFlashButton.place(x=DCRackGui.w*(7), y=DCRackGui.h*(7.75))
            else:
               self.RegLoadFlash.set(self.PreampLedStateData[card]["RegLoadFlash"])
            self.RegLoadFlashButton.configure(state=configState)
                
            for jj in range(0, len(self.PreampChannels)):
                self.PreampChannelLabels[self.PreampChannels[jj]] = Label(self, text=self.PreampChannels[jj])
                self.PreampChannelLabels[self.PreampChannels[jj]].configure(state=configState)
                self.PreampChannelLabels[self.PreampChannels[jj]].place(x=DCRackGui.w*(1.25), y=DCRackGui.h*(10+jj))
                for kk in range(0, len(self.PreampSettings)):
                    if jj==0: # make labels once
                        self.PreampSettingsLabels[self.PreampSettings[kk]] = Label(self, text=self.PreampSettings[kk])
                        self.PreampSettingsLabels[self.PreampSettings[kk]].place(x=DCRackGui.w*(2+2*kk), y=DCRackGui.h*(9))
                        self.PreampSettingsLabels[self.PreampSettings[kk]].configure(state=configState)
                    if self.PreampSettings[kk] not in self.PreampOptionMenusData[self.PreampChannels[jj]].keys():
                        if "Offset" not in self.PreampSettings[kk]:
                           self.PreampOptionMenusData[self.PreampChannels[jj]][self.PreampSettings[kk]]=StringVar()
                           self.PreampOptionMenus[self.PreampChannels[jj]][self.PreampSettings[kk]]=OptionMenu(self,self.PreampOptionMenusData[self.PreampChannels[jj]][self.PreampSettings[kk]],*self.PreampSettingsListDict[self.PreampSettings[kk]], command=lambda event, channel= self.PreampChannels[jj],settingName=self.PreampSettings[kk],settingVal=self.PreampOptionMenusData[self.PreampChannels[jj]][self.PreampSettings[kk]]:self.setPreampSetting(channel, settingName, settingVal))
                        else:
                           self.PreampOptionMenusData[self.PreampChannels[jj]][self.PreampSettings[kk]]=IntVar()
                           self.PreampOptionMenus[self.PreampChannels[jj]][self.PreampSettings[kk]]=Spinbox(self, textvariable=self.PreampOptionMenusData[self.PreampChannels[jj]][self.PreampSettings[kk]],from_=0, to=2**16-1,command=lambda channel= self.PreampChannels[jj],settingName=self.PreampSettings[kk],settingVal=self.PreampOptionMenusData[self.PreampChannels[jj]][self.PreampSettings[kk]]:self.setPreampSetting(channel, settingName, settingVal))
                           self.PreampOptionMenus[self.PreampChannels[jj]][self.PreampSettings[kk]].bind("<Return>",lambda event, channel= self.PreampChannels[jj],settingName=self.PreampSettings[kk],settingVal=self.PreampOptionMenusData[self.PreampChannels[jj]][self.PreampSettings[kk]]:self.setPreampOffsetReturn(channel, settingName, settingVal))
          
                    self.PreampOptionMenus[self.PreampChannels[jj]][self.PreampSettings[kk]].configure(width=10)
                    self.PreampOptionMenus[self.PreampChannels[jj]][self.PreampSettings[kk]].place(x=DCRackGui.w*(2+2*kk), y=DCRackGui.h*(10+jj))
                    
                    self.PreampOptionMenus[self.PreampChannels[jj]][self.PreampSettings[kk]].configure(state=configState)

        def setLEDs(self,ledType):
            card = self.preampCard.get()
            cardID = int(card.split(",")[0].strip("("))
            dc.select_card(cardID)
            self.PreampLedStateData[card]["FOout"]=self.FOout.get()
            self.PreampLedStateData[card]["FOflash"]=self.FOflash.get()
            self.PreampLedStateData[card]["RegLoadFlash"]=self.RegLoadFlash.get()
            dc.leds(bool(self.FOout.get()),bool(self.FOflash.get()),bool(self.RegLoadFlash.get()))


        def setPreampSetting(self, channel, settingName, settingVal):
            preampSetting = settingVal.get()
            card = self.preampCard.get()
            cardID = int(card.split(",")[0].strip("("))
            if "High-Pass" in settingName:
               self.changeHighPassFilterSetting(cardID, channel, preampSetting)
            elif "Low-Pass" in settingName:
               self.changeLowPassFilterSetting(cardID, channel, preampSetting)
            elif "Offset" in settingName:
               self.changeOffsetSetting(cardID, channel, preampSetting)
            elif "Polarity" in settingName:
               self.changePolaritySetting(cardID, channel, preampSetting)
            else:
               print "Error: Unknown setting name encountered in setPreampSetting()"
            self.PreampSettingsData[card][channel][settingName]=settingVal.get()
               

        def changeHighPassFilterSetting(self, cardID, channel, highPassSetting):
           
            dc.select_card(cardID)
            dc.change_high_pass_filter(channel, highPassSetting)

        def changeLowPassFilterSetting(self, cardID, channel, lowPassSetting):
           
            dc.select_card(cardID)
            dc.change_low_pass_filter(channel, lowPassSetting)

        def changeOffsetSetting(self, cardID, channel, offsetSetting):
           
            dc.select_card(cardID)
            dc.change_dc_offset(channel, offsetSetting)

        def changePolaritySetting(self, cardID, channel, polaritySetting):
           
            dc.select_card(cardID)
            dc.change_polarity(channel, polaritySetting)
           

        def setPreampOffsetReturn(self, channel, settingName, settingVal):
           
            card = self.preampCard.get()
            offset = settingVal.get()
            if offset<0 or offset>2**16-1: # set to whatever it was before because user sucks
               settingVal.set(self.PreampSettingsData[card][channel][settingName])
               offset = self.PreampSettingsData[card][channel][settingName]
            else:
               cardID = int(card.split(",")[0].strip("("))
               self.changeOffsetSetting(cardID, channel, offset)
               self.PreampSettingsData[card][channel][settingName] = offset
            self.focus()
        
        def initDacVrefs(self):
            dc.init_dacs()
        
        def saveToRegistry(self):
           
            dc.commit_monitor_state_to_registry()
            for cards in self.availablePreampCards:
               cardID = int(cards.split(",")[0].strip("("))
               dc.select_card(cardID)
               dc.commit_led_state_to_registry((bool(self.PreampLedStateData[cards]["FOout"]),bool(self.PreampLedStateData[cards]["FOflash"]),bool(self.PreampLedStateData[cards]["RegLoadFlash"])))
               dc.commit_to_registry()
               
        def uploadFromRegistry(self):

            for cards in self.availablePreampCards:
               cardID = int(cards.split(",")[0].strip("("))
               dc.select_card(cardID)
               dc.load_from_registry()
               for channel in ['A', 'B', 'C', 'D']:
                  preampState = dc.get_preamp_state(cardID, channel)
                  self.PreampSettingsData[cards][channel]["High-Pass \n Time Constant [us]"] = preampState[0]
                  dc.change_high_pass_filter(channel, preampState[0]) #set preamp as cards are intialized upon power up
                  self.PreampSettingsData[cards][channel]["Low-Pass \n Time Constant [us]"] = preampState[1]
                  dc.change_low_pass_filter(channel, preampState[1])
                  self.PreampSettingsData[cards][channel]["Polarity"] = preampState[2]
                  dc.change_polarity(channel, preampState[2])
                  self.PreampSettingsData[cards][channel]["Offset"] = int(preampState[3])
                  dc.change_dc_offset(channel, int(preampState[3]))
                  if channel == 'D':
                     ledState = dc.get_led_state_from_registry()
                     if ledState !=-1:
                        self.PreampLedStateData[cards]["FOout"]=int(ledState[0])
                        self.PreampLedStateData[cards]["FOflash"]=int(ledState[1])
                        self.PreampLedStateData[cards]["RegLoadFlash"]=int(ledState[2])
                     else:
                        print "LED State not found in registry. After saving settings to registry once, this message will no longer appear."
            
            currentCard = self.preampCard.get()
            if len(currentCard)>0:
               self.updatePreampCardSettings(self.preampCard)

            dc.load_monitor_state_from_registry()
            monitorStateArray = dc.get_monitor_state()
            self.reconfigureMonitorState(monitorStateArray)

        def reconfigureMonitorState(self, monitorStateArray):

            for ii in range(0, len(monitorStateArray)):

               bus = self.BusChannels[ii]
               did = int(monitorStateArray[ii][0])
               channel = monitorStateArray[ii][1]
               if did !=0:
                  cardSelection = self.getCardNameFromCardId(did)
                  self.BusOptionMenusData[bus]["Card Address/Type"].set(cardSelection)
                  self.setBusSetting(bus, "Card Address/Type", self.BusOptionMenusData[bus]["Card Address/Type"])
                  if "Abus" not in bus:
                     self.BusOptionMenusData[bus]["Channel"].set(channel)
                  else:
                     self.BusOptionMenusData[bus]["Channel"].set(channel[0])
                  self.setBusSetting(bus, "Channel", self.BusOptionMenusData[bus]["Channel"])
                  self.BusOptionMenusData[bus]["On/Off"].set(1)
                  self.setBusSetting(bus, "On/Off", self.BusOptionMenusData[bus]["On/Off"])
               else:
                  self.BusOptionMenusData[bus]["On/Off"].set(0)
                  self.setBusSetting(bus, "On/Off", self.BusOptionMenusData[bus]["On/Off"])
                  self.BusOptionMenusData[bus]["Card Address/Type"].set("")
                  self.BusOptionMenusData[bus]["Channel"].set("")
                  
            self.busSettingControls()
              
        def getCardNameFromCardId(self, did):
            for ii in range(0, len(self.availableCards)):
               if did ==int(self.availableCards[ii].split(",")[0].strip("(")):
                  return self.availableCards[ii]
      

        def setBusState(self, busName, busState):
            state = busState.get()
            channelSelection = self.BusOptionMenusData[busName]["Channel"].get()
            cardSelection = self.BusOptionMenusData[busName]["Card Address/Type"].get()
            defaultSettingsList = self.getChannelOptionsList(busName, 'preamp')
            defaultSetting = defaultSettingsList[0]
            
            if state==0 or len(channelSelection)==0: # i.e. off
               dc.select_card(0)
               if 'Abus0' in busName:
                  defaultSetting = defaultSetting+'0'
               elif 'Abus1' in busName:
                  defaultSetting = defaultSetting+'1'
               dc.change_monitor(busName,defaultSetting)
            else:
               cardId = int(cardSelection.split(",")[0].strip("("))
               dc.select_card(cardId)
               if 'Abus0' in busName:
                  channelSelection = channelSelection+'0'
               elif 'Abus1' in busName:
                  channelSelection = channelSelection+'1'
               dc.change_monitor(busName,channelSelection)


        def setBusCardMapping(self, busName, cardSelection):
            card = cardSelection.get()
            if card != self.BusMostRecentSetting[busName]:
               newList = self.getChannelOptionsList(busName,card)
               self.updateChannelOptionsList(self.BusOptionMenusData[busName]["Channel"], self.BusOptionMenus[busName]["Channel"], newList)
               #initialize bus as new card was selected which requires a channel selection as well
               self.BusMostRecentSetting[busName] = card
               defaultSettingsList = self.getChannelOptionsList(busName, 'preamp')
               defaultSetting = defaultSettingsList[0]
               dc.select_card(0)
               if 'Abus0' in busName:
                  defaultSetting = defaultSetting+'0'
               elif 'Abus1' in busName:
                  defaultSetting = defaultSetting+'1'
               dc.change_monitor(busName,defaultSetting)
            self.BusOptionMenus[busName]["Channel"].configure(state='normal')

        
        def setBusChannelMapping(self, busName, cardSelection, channelSelection):
            selection = channelSelection.get()
            card = cardSelection.get()
            state = self.BusOptionMenusData[busName]["On/Off"].get()
            cardId = int(card.split(",")[0].strip("("))
            if state==1:
               dc.select_card(cardId)
               if 'Abus0' in busName:
                  selection = selection+'0'
               elif 'Abus1' in busName:
                  selection = selection+'1'
               dc.change_monitor(busName,selection)
            else:
               dc.select_card(0)
               defaultSettingsList = self.getChannelOptionsList(busName, 'preamp')
               defaultSetting = defaultSettingsList[0]
               dc.select_card(0)
               if 'Abus0' in busName:
                  defaultSetting = defaultSetting+'0'
               elif 'Abus1' in busName:
                  defaultSetting = defaultSetting+'1'
               dc.change_monitor(busName,defaultSetting)
            
            
            

        def updateChannelOptionsList(self, optionMenuVar, optionMenuObject, newList):
            optionMenuVar.set('')
            optionMenuObject['menu'].delete(0, 'end')
            for items in newList:
               optionMenuObject['menu'].add_command(label=items, command=Tkinter._setit(optionMenuVar, items))
            self.busSettingControls()

        def getChannelOptionsList(self, busName, cardSelection):
            if len(cardSelection)==0:
                return [""]
            elif "Dbus0" in busName:
                if 'preamp' in cardSelection:
                    return ['trigA','trigB','trigC','trigD', 'dadata', 'done', 'strobe', 'clk']
                elif 'fastbias' in cardSelection:
                    return ['Pbus0','clk','clockon', 'cardsel', 'clk1', 'clk2', 'clk3', 'clk4']
            elif "Dbus1" in busName:
                if 'preamp' in cardSelection:
                    return ['FOoutA','FOoutB','FOoutC', 'FOoutD', 'dasyn', 'cardsel', 'Pbus0', 'Clockon']
                elif 'fastbias' in cardSelection:
                    return ['foin1', 'foin2', 'foin3', 'foin4', 'on1', 'on2', 'on3', 'on4']
            elif "Abus" in busName:
                return ["A", "B", "C", "D"]
            else:
                return ["Error"]

    def main():
      
        root = Tk()
        app = DCRackGui(root)
        root.mainloop()  


    if __name__ == '__main__':
        main()  
