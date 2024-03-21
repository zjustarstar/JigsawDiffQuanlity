"""this is a class for each patch
    now it has the following attributes:
        object：0,1
        edge：0,1(common),2(vertical),3(horizontal)
"""

class Cpatch():

    def __init__(self,num):
        self.attribute={
            "patchID":num,
            "type":"Unknow",
            "value":{},
            "SimPatchInObj":0,
            "SimPatchOutObj":0
        }


    def change_Type(self,type_num):
        if self.attribute["type"]== "Unknow":
            if type_num == -1:
                self.attribute["type"]="Edge"
            elif type_num >0:
                self.attribute["type"]="Object"
                self.attribute["value"][int(type_num)]=1


    def change_Value(self,value):
        if self.attribute["type"] !="Object":
            self.attribute["value"]=value


    def change_SimPatchInObj(self,simpatchInobj):
        self.attribute["SimPatchInObj"]=simpatchInobj


    def change_SimPatchOutObj(self,simpatchOutobj):
        self.attribute["SimPatchOutObj"]=simpatchOutobj


