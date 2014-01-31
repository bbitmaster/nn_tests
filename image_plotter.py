# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 15:47:27 2013

@author: Ben
"""
import cv2
import numpy as np

default_width = 640;
default_height = 480;

class NoImageDefined(Exception):
    pass
    
class image_plotter(object):
    def __init__(self):
        self.windowname = "plot"
        pass
    
    #Image and window init
    def setAxis(self,x1,x2,y1,y2):
        self.axis = (x1,x2,y1,y2);
    def setWindowName(self,windowname):
        self.windowname = "plot"
    def setImg(self,img):
        if(type(img) == np.ndarray):
            self.img = img; #should be a numpy array
    def initImg(self,width=default_width,height=default_height):
        self.img = np.zeros((height,width,3))
        
    #Show/Hide
    def show(self):
        if(self.img is None):
            raise NoImageDefined()
        cv2.namedWindow(self.windowname,0)
        
        cv2.imshow(self.windowname, self.img)
        cv2.startWindowThread()
    def hide(self):
        cv2.destroyWindow(self.windowname)
    def update(self):
        cv2.imshow(self.windowname, self.img)
    
    #save img
    def save_plot(self,filename):
        cv2.imwrite(filename,self.img*255)
        
    def processEvents(self):
        return cv2.waitKey(1);

    #TODO: Write This
    def convertToAxisCoords(self,coords):
        pass
    def convertToImgCoords(self,coords):
        img_coords = [0,0];
        img_width = self.img.shape[1]
        img_height = self.img.shape[0]

        img_coords[0] = float(((coords[0] - self.axis[0])/float(self.axis[1] - self.axis[0]))*float(img_width))
        img_coords[1] = float(((coords[1] - self.axis[2])/float(self.axis[3] - self.axis[2]))*float(img_height))
        return tuple(img_coords);
        
        
    #Drawing fuunctions
    def drawLine(self,x1,x2,y1,y2,color=(255,0,0)):
        #need to convert coordinates of axis to coordinates within the image.
        if type(x1) == np.ndarray:
            length = x1.shape[0];
            for i in range(length):
                if(type(color) == np.ndarray):
                    c= (float(color[i,0]),float(color[i,1]),float(color[i,2]))
                else:
                    c = color
                #cv2.line(self.img,self.convertToImgCoords((x1[i],y1[i])),self.convertToImgCoords((x2[i],y2[i])),c)
                self._drawLine(x1[i],x2[i],y1[i],y2[i],c)
        else:
            #cv2.line(self.img,self.convertToImgCoords((x1,y1)),self.convertToImgCoords((x2,y2)),color)
            self._drawLine(x1,x2,y1,y2,color)
            
    #non-vectorized version
    def _drawLine(self,x1,x2,y1,y2,color=(255,0,0)):
        (img_x1,img_y1) = self.convertToImgCoords((x1,y1));
        (img_x2,img_y2) = self.convertToImgCoords((x2,y2));
        
        cv2.line(self.img,(int(img_x1),int(img_y1)),(int(img_x2),int(img_y2)),color)

    def drawRect(self,x1,x2,y1,y2,color=(255,0,0),thickness=-1,use_image_coords=False):
        if type(x1) == np.ndarray:
            length = x1.shape[0]
            for i in range(length):
                if(type(color) == np.np.ndarray):
                    c = (float(color[i,0]),float(color[i,1]),float(color[i,2]))
                else:
                    c = color
                self._drawRect(x1[i],x2[i],y1[i],y2[i],c,use_image_coords=use_image_coords)
        self._drawRect(x1,x2,y1,y2,color,use_image_coords=use_image_coords)

    def _drawRect(self,x1,x2,y1,y2,color=(255,0,0),thickness=-1,use_image_coords=False):
        if(use_image_coords):
            (img_x1,img_y1) = (x1,y1)
            (img_x2,img_y2) = (x2,y2)
        else:
            (img_x1,img_y1) = self.convertToImgCoords((x1,y1));
            (img_x2,img_y2) = self.convertToImgCoords((x2,y2));
        cv2.rectangle(self.img,(int(img_x1),int(img_y1)),(int(img_x2),int(img_y2)),color,thickness=thickness)

    def drawPoint(self,x,y,color=(1,0,0),size=1,plotType='x'):
        if type(x) == np.ndarray:
            length = x.shape[0];
            for i in range(length):
                if(type(color) == np.ndarray):
                    c= (float(color[i,0]),float(color[i,1]),float(color[i,2]))
                else:
                    c = color
                self._drawPoint(x[i],y[i],c,size,plotType)
        else:
            self._drawPoint(x,y,color,size,plotType)
                    
    def _drawPoint(self,x,y,color=(255,0,0),size=1,plotType='x'):
        if(plotType == 'x'):
            coords = list(self.convertToImgCoords((x,y)))
            #draw the \
            coords1 = (int(coords[0] - size*2),int(coords[1] - size*2))
            coords2 = (int(coords[0] + size*2),int(coords[1] + size*2))
            cv2.line(self.img,coords1,coords2,color,thickness=size)
            #draw the /
            coords1 = (int(coords[0] - size*2),int(coords[1] + size*2))
            coords2 = (int(coords[0] + size*2),int(coords[1] - size*2))
            cv2.line(self.img,coords1,coords2,color,thickness=size)

    def drawText(self,x,y,txt,size=1,thickness=1,color=(1,1,1),use_image_coords=False):
        if(use_image_coords):
            (img_x,img_y) = (x,y)
        else:
            (img_x,img_y) = self.convertToImgCoords((x,y));

        img_y = img_y + cv2.getTextSize(txt,cv2.FONT_HERSHEY_PLAIN,fontScale=size,thickness=thickness)[0][1];
        cv2.putText(self.img,txt,(int(img_x),int(img_y)),cv2.FONT_HERSHEY_PLAIN,fontScale=size,color=color)

    #draw text using image coordinates
    #def drawTextImg(self,x,y,txt,size=1,color=(1,1,1)):
    #    img_y = img_y + cv2.getTextSize(txt,cv2.FONT_HERSHEY_PLAIN,fontScale=size,thickness=1)[0][1];
    #    cv2.putText(self.img,txt,(int(img_x),int(img_y)),cv2.FONT_HERSHEY_PLAIN,fontScale=size,color=color)
